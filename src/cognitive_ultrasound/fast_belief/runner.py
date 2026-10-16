"""Finite three-stage pipeline with exact checkpoints, isolated probe and bounded storage."""

import copy
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset

from .data import Clips, atomic, digest, file_lock, load_config, prepare, read, sha, verify
from .evaluation import final_evaluation, sync, validate
from .learning import update
from .model import ArrayOracle, BeliefSystem

REPO = Path(__file__).resolve().parents[3]
_storage_checked = {}


def emit(**values):
    print(json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


def environment(device):
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        driver = None
    p = psutil.Process()
    limits = {}
    for key, path in {
        "cpu_max": "/sys/fs/cgroup/cpu.max",
        "memory_max": "/sys/fs/cgroup/memory.max",
    }.items():
        if Path(path).is_file():
            limits[key] = Path(path).read_text().strip()
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        cuda=torch.version.cuda,
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        vram=torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else None,
        cudnn=torch.backends.cudnn.version(),
        driver=driver,
        cpus=psutil.cpu_count(),
        affinity=len(p.cpu_affinity()) if hasattr(p, "cpu_affinity") else None,
        ram_bytes=psutil.virtual_memory().total,
        cgroup_v2=limits,
    )


def configure(cpu, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20261016)
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; no silent CPU fallback. --cpu is for local functional tests"
        )
    return torch.device("cpu" if cpu else "cuda")


def save(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def seed(cfg, stage):
    return cfg["seed"] + {"codec": 0, "prior": 1, "filter": 2}[stage] * 10000


def checkpoint(model, optimizer, state):
    return dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        state=state,
        rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
    )


def restore(payload, model, optimizer=None):
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"].cpu())
    if payload["cuda_rng"] and next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])
    return payload["state"]


def training_loader(manifest, cfg, stage, epoch, start_index, workers):
    s = cfg["training"][stage]
    ds = Clips(manifest, manifest["cohorts"]["train"], s["frames"], seed(cfg, stage), epoch)
    kwargs = dict(
        batch_size=s["batch"],
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=torch.Generator().manual_seed(seed(cfg, stage) + epoch),
    )
    if workers:
        kwargs.update(prefetch_factor=cfg["runtime"]["prefetch"], persistent_workers=False)
    return DataLoader(Subset(ds, range(start_index, len(ds))), **kwargs)


def frozen_signature(model, stage):
    names = {
        "codec": ["prior", "updater"],
        "prior": ["codec", "updater"],
        "filter": ["codec", "prior"],
    }[stage]
    h = __import__("hashlib").sha256()
    for name in names:
        for k, v in getattr(model, name).state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def short_trace(model, clip, starts):
    memory = model.initial(len(clip), clip.device)
    trace = []
    for t in range(min(8, clip.shape[1])):
        memory, state = model.frame(ArrayOracle(clip[:, t]), memory, starts + t)
        trace.append(
            [
                v.detach().cpu().clone()
                for v in [
                    memory.z,
                    memory.image,
                    memory.uncertainty,
                    memory.values,
                    *state["actions"],
                ]
            ]
        )
    return trace


def probe(cfg, manifest, device, output):
    """Fresh models only. Real greedy workload; never advances production weights."""
    results = []
    stage = "filter"
    ds = Clips(
        manifest, manifest["cohorts"]["train"], cfg["training"][stage]["frames"], cfg["seed"], 0
    )
    b = cfg["training"][stage]["batch"]
    values = [ds[i % len(ds)] for i in range(b)]
    x = torch.from_numpy(np.stack([v[0] for v in values])).to(device)
    starts = torch.tensor([v[1] for v in values], device=device)
    original_rng = torch.get_rng_state()
    torch.manual_seed(cfg["seed"])
    template = BeliefSystem(cfg)
    initial = copy.deepcopy(template.state_dict())
    stage_smoke = []
    for name in ["codec", "prior"]:
        emit(event="PROBE_STAGE", stage=name, updates=1)
        model = BeliefSystem(cfg).to(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.Adam(model.stage(name), lr=cfg["training"]["learning_rate"])
        frozen = frozen_signature(model, name)
        batch, indexes = next(iter(training_loader(manifest, cfg, name, 0, 0, 0)))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        started = time.perf_counter()
        record = update(model, optimizer, batch.to(device), indexes.to(device), name)
        sync(device)
        elapsed = time.perf_counter() - started
        if frozen_signature(model, name) != frozen:
            raise AssertionError("Stage probe changed frozen components")
        stage_smoke.append(
            dict(
                stage=name,
                first_update_seconds=elapsed,
                **record,
                projected_cap_hours=elapsed
                * math.ceil(len(ds) / cfg["training"][name]["batch"])
                * cfg["training"][name]["epochs"]
                / 3600,
                vram_peak=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            )
        )
        del model, optimizer, batch, indexes
        if device.type == "cuda":
            torch.cuda.empty_cache()
    expected = None
    expected_parameters, expected_trace = None, None
    for threads in cfg["runtime"]["calibrate_threads"]:
        emit(event="PROBE_CANDIDATE", threads=threads, updates=5)
        torch.set_num_threads(threads)
        model = BeliefSystem(cfg).to(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.Adam(model.stage(stage), lr=cfg["training"]["learning_rate"])
        frozen = frozen_signature(model, stage)
        times, losses = [], []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for i in range(5):
            sync(device)
            start = time.perf_counter()
            record = update(model, optimizer, x, starts, stage)
            sync(device)
            times.append(time.perf_counter() - start)
            losses.append(record["total"])
        parameters = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        trace = short_trace(model, x[:2], starts[:2])
        replay_matches = True
        if expected is None:
            expected = losses
            expected_parameters, expected_trace = parameters, trace
        else:
            replay_matches = all(
                torch.allclose(v, expected_parameters[k], atol=2e-4, rtol=2e-4)
                for k, v in parameters.items()
            )
            replay_matches &= all(
                torch.equal(a, b)
                if not a.is_floating_point()
                else torch.allclose(a, b, atol=2e-4, rtol=2e-4)
                for row, ref in zip(trace, expected_trace)
                for a, b in zip(row, ref)
            )
        if not replay_matches or not np.allclose(losses, expected, atol=2e-4, rtol=2e-4):
            results.append(dict(threads=threads, passed=False, reason="numerical replay differs"))
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        if frozen_signature(model, stage) != frozen:
            raise AssertionError("Probe changed frozen ancestors")
        results.append(
            dict(
                threads=threads,
                passed=True,
                steady_update_s=float(np.median(times[2:])),
                losses=losses,
                vram_peak=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            )
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    torch.set_rng_state(original_rng)
    candidates = [r for r in results if r["passed"]]
    if not candidates:
        raise RuntimeError("No numerically verified runtime configuration")
    chosen = min(candidates, key=lambda x: x["steady_update_s"])
    # Measure actual deterministic HDF5 batches; parallel loaders cannot reorder data.
    loaders = []
    expected_inputs = None
    torch.set_num_threads(chosen["threads"])
    for workers in sorted({0, cfg["runtime"]["workers"]}):
        start = time.perf_counter()
        signatures = []
        iterator = iter(training_loader(manifest, cfg, "codec", 0, 0, workers))
        for _ in range(min(4, math.ceil(len(ds) / cfg["training"]["codec"]["batch"]))):
            clip, indexes = next(iterator)
            signatures.append(
                digest(
                    dict(
                        pixels=__import__("hashlib").sha256(clip.numpy().tobytes()).hexdigest(),
                        starts=indexes.tolist(),
                    )
                )
            )
        elapsed = time.perf_counter() - start
        del iterator
        if expected_inputs is None:
            expected_inputs = signatures
        if signatures != expected_inputs:
            raise AssertionError("Prefetch changed input content/order")
        loaders.append(dict(workers=workers, first_batches_seconds=elapsed))
    selected_workers = min(loaders, key=lambda x: x["first_batches_seconds"])["workers"]
    value = dict(
        status="completed",
        fresh_models_only=True,
        scientific_config_unchanged=True,
        batch=b,
        frames=cfg["training"][stage]["frames"],
        selection="actual10+4 greedy",
        precision="fp32",
        candidates=results,
        stage_smoke=stage_smoke,
        throughput=loaders,
        chosen_threads=chosen["threads"],
        chosen_workers=selected_workers,
        projected_filter_hours=chosen["steady_update_s"]
        * math.ceil(len(ds) / b)
        * cfg["training"][stage]["epochs"]
        / 3600,
        projection_scope="fixed training cap only, excludes validation, loading, setup and convergence uncertainty",
    )
    atomic(output / "probe.json", value)
    emit(event="PROBE_COMPLETED", **value)
    return value


def guard(output, cfg):
    if (output / "STOP").exists():
        raise InterruptedError("Stop requested; checkpoint preserved")
    if shutil.disk_usage(output).free < 2 * 2**30:
        raise OSError("Less than2GiB free; preserve evidence and stop")
    now = time.monotonic()
    if now - _storage_checked.get(str(output), -1e10) > 30:
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        _storage_checked[str(output)] = now
        if size > cfg["runtime"]["max_output_gib"] * 2**30:
            raise OSError("Output exceeds configured disk cap; no automatic evidence deletion")


def fit(cfg, manifest, output, stage, device, identity, runtime, max_updates=None):
    folder = output / stage
    folder.mkdir(exist_ok=True)
    torch.manual_seed(seed(cfg, stage))
    model = BeliefSystem(cfg).to(device)
    optimizer = torch.optim.Adam(model.stage(stage), lr=cfg["training"]["learning_rate"])
    last, best = folder / "last.pt", folder / "best.pt"
    parent = {"prior": "codec", "filter": "prior"}.get(stage)
    state = dict(
        identity=identity,
        stage=stage,
        epoch=0,
        next_index=0,
        updates=0,
        best_mse=None,
        bad_checks=0,
        completed=False,
        epoch_loss_sum=0.0,
        epoch_examples=0,
        validation_history=[],
    )
    if last.exists():
        state = restore(load(last, device), model, optimizer)
        if state["identity"] != identity or state["stage"] != stage:
            raise ValueError("Checkpoint identity mismatch")
        if parent and state.get("parent_best_sha256") != sha(output / parent / "best.pt"):
            raise ValueError("Parent checkpoint changed; preserve this run and use a fresh output")
        if state["completed"]:
            if not best.exists():
                raise ValueError("Completed stage missing best weights")
            return
    else:
        if parent:
            payload = load(output / parent / "best.pt", device)
            if payload["state"]["identity"] != identity:
                raise ValueError("Parent identity mismatch")
            model.load_state_dict(payload["model"])
            state["parent_best_sha256"] = sha(output / parent / "best.pt")
        save(last, checkpoint(model, optimizer, state))
    frozen = frozen_signature(model, stage)
    total_epochs = cfg["training"][stage]["epochs"]
    start_updates = state["updates"]
    times = []
    # Logs after a lost, uncommitted update are rolled back to the last checkpoint.
    log = folder / "training.jsonl"
    if log.exists():
        records = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        temp = log.with_suffix(".tmp")
        temp.write_text(
            "".join(json.dumps(r) + "\n" for r in records if r["update"] <= state["updates"]),
            encoding="utf-8",
        )
        temp.replace(log)
    loader = None
    try:
        while state["epoch"] < total_epochs and not state["completed"]:
            guard(output, cfg)
            model.stage(stage)
            loader = training_loader(
                manifest, cfg, stage, state["epoch"], state["next_index"], runtime["chosen_workers"]
            )
            for x, starts in loader:
                guard(output, cfg)
                step_started = time.perf_counter()
                x = x.to(device, non_blocking=True)
                starts = starts.to(device, non_blocking=True)
                record = update(model, optimizer, x, starts, stage)
                sync(device)
                elapsed = time.perf_counter() - step_started
                times.append(elapsed)
                state.update(
                    next_index=state["next_index"] + len(x),
                    updates=state["updates"] + 1,
                    epoch_loss_sum=state["epoch_loss_sum"] + record["total"] * len(x),
                    epoch_examples=state["epoch_examples"] + len(x),
                )
                row = dict(
                    stage=stage,
                    epoch=state["epoch"] + 1,
                    update=state["updates"],
                    next_video=state["next_index"],
                    videos=len(manifest["cohorts"]["train"]),
                    seconds=elapsed,
                    ram_rss=psutil.Process().memory_info().rss,
                    vram_peak=torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                    **record,
                )
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                emit(event="TRAIN", **row)
                stop_after = (
                    max_updates is not None and state["updates"] - start_updates >= max_updates
                )
                if (
                    state["updates"] % cfg["training"]["checkpoint_updates"] == 0
                    or stop_after
                    or (output / "STOP").exists()
                ):
                    save(last, checkpoint(model, optimizer, state))
                    if frozen_signature(model, stage) != frozen:
                        raise AssertionError("Frozen components changed")
                    if stop_after:
                        return
                atomic(
                    output / "progress.json",
                    dict(
                        phase=stage,
                        epoch=state["epoch"] + 1,
                        epoch_cap=total_epochs,
                        update=state["updates"],
                        videos=state["next_index"],
                        total_videos=len(manifest["cohorts"]["train"]),
                        recent_update_seconds=float(np.median(times[-20:])),
                    ),
                )
            del loader
            loader = None
            epoch = state["epoch"] + 1
            if (
                epoch % cfg["training"]["validate_every"] == 0
                or epoch == total_epochs
                or not best.exists()
            ):
                save(last, checkpoint(model, optimizer, state))
                emit(
                    event="VALIDATE",
                    stage=stage,
                    epoch=epoch,
                    scope="full-video recurrent" if stage == "filter" else "component diagnostic",
                )
                value = validate(model, manifest, stage, device, lambda: guard(output, cfg))
                improved = (
                    state["best_mse"] is None
                    or value["mse"] < state["best_mse"] - cfg["training"]["min_delta"]
                )
                state["bad_checks"] = 0 if improved else state["bad_checks"] + 1
                if improved:
                    state["best_mse"] = value["mse"]
                state["validation_history"].append(
                    dict(
                        epoch=epoch,
                        training_loss=state["epoch_loss_sum"] / max(1, state["epoch_examples"]),
                        **value,
                    )
                )
                if improved:
                    save(best, checkpoint(model, optimizer, state))
                atomic(folder / "validation.json", state["validation_history"])
                emit(
                    event="VALIDATION_END",
                    stage=stage,
                    epoch=epoch,
                    mse=value["mse"],
                    psnr=value["psnr"],
                    best=improved,
                    bad_checks=state["bad_checks"],
                )
            state.update(
                epoch=epoch,
                next_index=0,
                epoch_loss_sum=0.0,
                epoch_examples=0,
                completed=epoch >= total_epochs
                or state["bad_checks"] >= cfg["training"]["patience"],
            )
            if frozen_signature(model, stage) != frozen:
                raise AssertionError("Frozen components changed")
            save(last, checkpoint(model, optimizer, state))
            atomic(folder / "status.json", state)
        emit(
            event="STAGE_COMPLETED",
            stage=stage,
            updates=state["updates"],
            epochs=state["epoch"],
            best_mse=state["best_mse"],
        )
    except BaseException:
        # A failed numerical update is not committed. Last verified checkpoint remains.
        if sys.exc_info()[0] is InterruptedError:
            save(last, checkpoint(model, optimizer, state))
        raise
    finally:
        del loader, optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_ef(cfg, device):
    if not cfg["evaluation"]["ef_enabled"]:
        return None
    from ..config import load as read_config
    from ..task_budget.ef_worker import EFModel
    from ..task_budget.protocol import clip_indices

    task = read_config(cfg["evaluation"]["ef_config"])
    with np.load(cfg["evaluation"]["coordinates"], allow_pickle=False) as z:
        coordinates = z["coordinates"]
    for key in ["ef_weights", "ef_stats"]:
        if not Path(task[key]).exists():
            raise FileNotFoundError(f"Explicit EF enabled, missing {key}")
    task["task_device"] = str(device)
    evaluator = EFModel(task, coordinates)

    def predict(images, truth):
        clips = clip_indices(len(images), 32, 2, 1)
        values = []
        for indexes in clips:
            value = evaluator.evaluate(images[indexes][None], False, "polar")
            values.append(float(value["predictions"][0]))
        prediction = float(np.mean(values))
        return dict(
            prediction=prediction,
            absolute_error=abs(prediction - truth),
            clips=len(clips),
            readout="all_starts_v1",
            weights_sha256=sha(task["ef_weights"]),
            stats_sha256=sha(task["ef_stats"]),
            coordinate_sha256=sha(cfg["evaluation"]["coordinates"]),
        )

    return predict


def ef_identity(cfg):
    if not cfg["evaluation"]["ef_enabled"]:
        return None
    from ..config import load as read_config

    path = cfg["evaluation"]["ef_config"]
    task = read_config(path)
    return {
        "config": sha(path),
        "coordinates": sha(cfg["evaluation"]["coordinates"]),
        "weights": sha(task["ef_weights"]),
        "stats": sha(task["ef_stats"]),
    }


def run(config, output, cpu=False, probe_only=False, resume=False):
    from .report import report

    cfg = load_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = configure(cpu, cfg["runtime"]["threads"])
    with file_lock(output / ".coordinator.lock"):
        env = environment(device)
        source = {str(p.relative_to(REPO)): sha(p) for p in Path(__file__).parent.glob("*.py")}
        identity = dict(
            config=cfg,
            source=source,
            runtime={
                k: env[k]
                for k in [
                    "python",
                    "torch",
                    "numpy",
                    "cuda",
                    "gpu",
                    "vram",
                    "cudnn",
                    "driver",
                    "affinity",
                    "cgroup_v2",
                ]
            },
            cpu_functional=cpu,
            ef_assets=ef_identity(cfg),
        )
        if (output / "identity.json").exists():
            if read(output / "identity.json") != identity:
                raise ValueError(
                    "Code/config/device changed; use a fresh output, never edit identity"
                )
            manifest = read(output / "manifest.json")
            verify(manifest)
        else:
            if any(p.name not in [".coordinator.lock"] for p in output.iterdir()):
                raise ValueError(
                    "Unrecognized existing output; preserve files and use a fresh directory"
                )
            if shutil.disk_usage(output).free < cfg["runtime"]["free_gib"] * 2**30:
                raise OSError("Insufficient initial disk space")
            manifest = prepare(cfg)
            atomic(output / "manifest.json", manifest)
            atomic(output / "identity.json", identity)
            atomic(output / "config.json", cfg)
            atomic(output / "environment.json", env)
        if (output / "complete.json").exists():
            done = read(output / "complete.json")
            if done["selected_checkpoint_sha256"] != sha(output / "filter/best.pt"):
                raise ValueError("Completed run's selected checkpoint changed")
            emit(event="ALREADY_COMPLETED")
            report(output)
            atomic(
                output / "status.json",
                dict(
                    status="completed",
                    phase="export_verified",
                    pid=os.getpid(),
                    created=psutil.Process().create_time(),
                    finished=time.time(),
                ),
            )
            return
        if resume and (output / "STOP").exists():
            (output / "STOP").unlink()
        state = dict(
            status="running",
            pid=os.getpid(),
            created=psutil.Process().create_time(),
            cpu_functional=cpu,
            phase="probe",
        )
        atomic(output / "status.json", state)
        try:
            if (output / "probe.json").exists():
                runtime = read(output / "probe.json")
            else:
                runtime = probe(cfg, manifest, device, output)
            torch.set_num_threads(runtime["chosen_threads"])
            if probe_only:
                atomic(output / "status.json", dict(state, status="probe_completed"))
                return
            identity_hash = digest(identity)
            for stage in ["codec", "prior", "filter"]:
                state["phase"] = stage
                atomic(output / "status.json", state)
                fit(cfg, manifest, output, stage, device, identity_hash, runtime)
            state["phase"] = "final_evaluation"
            atomic(output / "status.json", state)
            model = BeliefSystem(cfg).to(device)
            payload = load(output / "filter/best.pt", device)
            restore(payload, model)
            ef = load_ef(cfg, device)
            evaluation_id = digest(
                dict(
                    identity=identity_hash,
                    checkpoint_sha256=sha(output / "filter/best.pt"),
                    ef=cfg["evaluation"]["ef_enabled"],
                )
            )
            rows = final_evaluation(
                model,
                manifest,
                output / "evaluation",
                device,
                evaluation_id,
                lambda: guard(output, cfg),
                ef,
            )
            atomic(output / "results.json", rows)
            atomic(
                output / "complete.json",
                dict(
                    status="completed",
                    identity=identity_hash,
                    selected_checkpoint_sha256=sha(output / "filter/best.pt"),
                    ef_evaluated=ef is not None,
                    full_train_dataset=len(manifest["cohorts"]["train"]),
                    cpu_functional=cpu,
                ),
            )
            state.update(phase="report")
            atomic(output / "status.json", state)
            bundles = report(output)
            state.update(status="completed", phase="export_verified", finished=time.time())
            atomic(output / "status.json", state)
            emit(
                event="COMPLETED",
                bundles={
                    key: {k: r[k] for k in ["archive", "bytes", "sha256", "verified"]}
                    for key, r in bundles.items()
                },
            )
        except BaseException as error:
            state.update(
                status="stopped" if isinstance(error, InterruptedError) else "failed",
                error=f"{type(error).__name__}: {error}",
            )
            atomic(output / "status.json", state)
            raise
