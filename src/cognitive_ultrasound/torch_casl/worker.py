"""Separate Torch process: parity first, synchronized same-work timing second."""

import time
from pathlib import Path

import numpy as np
import torch

from ..preparation.common import atomic_json, atomic_npz, read_json
from ..provenance import sha256
from .native import CapturedFrame, FrozenGraph, NativeCASL, nchw


def comparison(actual, expected, atol=2e-4, rtol=2e-4):
    a, b = np.asarray(actual), np.asarray(expected)
    return dict(
        passed=bool(np.isfinite(a).all() and np.allclose(a, b, atol=atol, rtol=rtol)),
        max_abs=float(np.max(np.abs(a - b))),
        mean_abs=float(np.mean(np.abs(a - b))),
    )


def validate_probes(model, export, device):
    with np.load(Path(export) / "probes.npz", allow_pickle=False) as z:
        x, y, m, n, s = [
            nchw(z[k], device) for k in ("x", "measurement", "mask", "noise", "signal")
        ]
        prediction = model.network(x, n * n)
        grad, _ = model.grad_fn(x, y, m, n, s)
        # Exported entropy probe has particle channels packed as frames for select().
        particles = torch.from_numpy(z["particles"][0].copy()).to(device)
        selected, entropy = model.select(particles[:, None])
        result = dict(
            network=comparison(prediction.detach().permute(0, 2, 3, 1).cpu(), z["prediction"]),
            dps_gradient=comparison(grad.detach().permute(0, 2, 3, 1).cpu(), z["gradient"]),
            entropy=comparison(entropy.cpu(), z["entropy"][0]),
            selected=bool(np.array_equal(selected.cpu(), z["selected"][0])),
        )
    result["passed"] = all(v["passed"] if isinstance(v, dict) else v for v in result.values())
    return result


def timed(fn, args):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    result = fn(*args)
    torch.cuda.synchronize()
    return result, time.perf_counter() - tick


def run_torch(cfg, output, mode):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU Torch missing; no CPU speed fallback")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("highest")
    device = "cuda"
    export = output / "export"
    spec = read_json(export / "network.json")
    if sha256(export / "ema.npz") != spec["ema_sha256"]:
        raise ValueError("Export checksum mismatch")
    model = NativeCASL(FrozenGraph(export), cfg["budget"]).to(
        device=device, memory_format=torch.channels_last
    )
    checks = validate_probes(model, export, device)
    atomic_json(output / f"{mode}.probes.json", checks)
    if not checks["passed"]:
        raise ValueError("Native EMA/DPS/selection parity failed; speed is not admissible")
    reference = read_json(output / "reference.json")
    if not reference["completed"]:
        raise ValueError("Incomplete JAX reference")
    if mode in ("compile", "graph"):
        model.step_fn = torch.compile(model.dps_step, fullgraph=True, mode="default")
        model.select = torch.compile(model.select, fullgraph=True, mode="default")
    rows, functions, setup = [], {}, {}
    # Identical history/noise/masks isolate framework changes from trajectory divergence.
    for row in reference["rows"]:
        with np.load(output / "fixtures" / f"{row['id']}.npz", allow_pickle=False) as z:
            args = tuple(
                nchw(z[k], device) for k in ("measurement", "mask", "previous", "initial_noise")
            )
            steps = row["steps"]
            if row["cold"]:
                # Cold 500-step path is measured once and excluded from steady FPS.
                result, first_s = timed(lambda *a: model.frame(*a, steps=500, cold=True), args)
                samples = []
            else:
                if steps not in functions:
                    tick = time.perf_counter()
                    if mode == "graph":
                        functions[steps] = CapturedFrame(model, args, steps)
                    else:
                        functions[steps] = lambda *a, _s=steps: model.frame(*a, steps=_s)
                    for _ in range(2):
                        timed(functions[steps], args)
                    setup[steps] = time.perf_counter() - tick
                result, first_s = timed(functions[steps], args)
                samples = []
                for _ in range(cfg["repeats"]):
                    result, elapsed = timed(functions[steps], args)
                    samples.append(elapsed)
            arrays = [x.detach().cpu().numpy() for x in result]
            if not all(np.isfinite(a).all() for a in arrays):
                raise FloatingPointError("Nonfinite native frame; reject this execution mode")
            parity = comparison(arrays[0].transpose(0, 2, 3, 1), z["samples"])
            matched_parity = comparison(arrays[0].transpose(0, 2, 3, 1), z["matched_samples"])
            selected = bool(np.array_equal(arrays[2], z["selected"]))
            target = z["target"]
            quality = dict(
                mae=float(np.abs(arrays[1] - target).mean()),
                reference_mae=float(np.abs(z["prediction"] - target).mean()),
            )
            rows.append(
                dict(
                    id=row["id"],
                    steps=steps,
                    cold=row["cold"],
                    seconds=samples,
                    first_call_s=first_s,
                    parity=parity,
                    matched_parity=matched_parity,
                    matched_selected_equal=bool(np.array_equal(arrays[2], z["matched_selected"])),
                    selected_equal=selected,
                    selected_count=int(arrays[2].sum()),
                    **quality,
                )
            )
            # Compact explanatory figures can be made from these fixed-index arrays.
            if row["frame"] in (0, 2):
                atomic_npz(
                    output / mode / f"{row['id']}.npz",
                    prediction=arrays[1],
                    reference=z["prediction"],
                    target=target,
                    selected=arrays[2],
                    reference_selected=z["selected"],
                )
            atomic_json(
                output / f"{mode}.json", dict(rows=rows, completed=False, setup_seconds=setup)
            )
            print(
                f"TORCH {mode} {row['id']} parity={parity['passed']} "
                f"lines={selected} seconds={samples or [first_s]}",
                flush=True,
            )
    # Causal rollout uses its OWN posterior and future mask, with common random numbers.
    trajectory = []
    previous = mask = measurement = None
    for row in reference["rows"]:
        with np.load(output / "fixtures" / f"{row['id']}.npz", allow_pickle=False) as z:
            # Decompress fixture arrays before timing: separately exclude disk/NPZ I/O.
            host_noise, host_target = z["initial_noise"], z["target"]
            if row["cold"]:
                host_mask, host_measurement, host_previous = (
                    z["mask"],
                    z["measurement"],
                    z["previous"],
                )
            torch.cuda.synchronize()
            tick = time.perf_counter()
            noise = nchw(host_noise, device)
            target = torch.from_numpy(host_target.copy()).to(device)
            if row["cold"]:
                mask = nchw(host_mask, device)
                measurement = torch.zeros_like(nchw(host_measurement, device))
                previous = nchw(host_previous, device)
            obs = target[None, None] * mask[:, -1:]
            measurement = torch.cat(
                (measurement[:, 1:], obs.expand(2, -1, -1, -1)), dim=1
            ).contiguous(memory_format=torch.channels_last)
            args = (measurement, mask, previous, noise)
            if row["cold"]:
                result = model.frame(*args, steps=500, cold=True)
            else:
                result = functions[row["steps"]](*args)
            previous, prediction, selected, _ = result
            mask = torch.cat(
                (mask[:, 1:], selected[None, None, None, :].expand(1, 1, 112, 112)), dim=1
            ).contiguous(memory_format=torch.channels_last)
            pred = prediction.detach().cpu().numpy()  # includes return transfer and sync
            seconds = time.perf_counter() - tick
            trajectory.append(
                dict(
                    id=row["id"],
                    steps=row["steps"],
                    cold=row["cold"],
                    host_wall_s=seconds,
                    mae=float(np.abs(pred - z["target"]).mean()),
                    reference_mae=float(np.abs(z["prediction"] - z["target"]).mean()),
                    selected_equal=bool(np.array_equal(selected.cpu(), z["selected"])),
                )
            )
            atomic_json(output / f"{mode}.trajectory.json", dict(rows=trajectory, completed=False))
    atomic_json(output / f"{mode}.trajectory.json", dict(rows=trajectory, completed=True))
    atomic_json(
        output / f"{mode}.json",
        dict(
            rows=rows,
            completed=True,
            setup_seconds=setup,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            tf32=False,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        ),
    )
