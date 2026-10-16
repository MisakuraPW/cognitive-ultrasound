"""Full-video recurrent checks; bounded samples, no full trajectory cache."""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import atomic, frames, read, sha
from .model import ArrayOracle, ssim


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def measures(target, prediction, mask=None):
    error = (target - prediction).float()
    mse = error.square().mean()
    value = dict(
        mse=float(mse),
        mae=float(error.abs().mean()),
        psnr=float(10 * torch.log10(4 / mse.clamp_min(1e-12))),
        ssim=float(ssim(target, prediction).mean()),
    )
    if mask is not None:
        value["unobserved_mae"] = float(
            (error.abs() * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1)
        )
        value["observed_max_error"] = float((error.abs() * mask).max())
    if not np.isfinite(list(value.values())).all():
        raise FloatingPointError("Nonfinite evaluation")
    return value


def average(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def interpolate(observed, lines):
    x = torch.arange(112, device=observed.device)
    right = torch.searchsorted(lines, x).clamp(0, len(lines) - 1)
    left = (right - 1).clamp_min(0)
    left = torch.where(x > lines[-1], right, left)
    weight = ((x - lines[left]) / (lines[right] - lines[left]).clamp_min(1)).clamp(0, 1)
    return (
        observed.index_select(-1, lines[left]) * (1 - weight)
        + observed.index_select(-1, lines[right]) * weight
    )


def save_sample(directory, index, truth, prediction, mask, uncertainty):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        target=truth[0, 0].cpu().numpy(),
        reconstruction=prediction[0, 0].cpu().numpy(),
        mask=mask[0, 0].cpu().numpy(),
        uncertainty=uncertainty[0, 0].cpu().numpy(),
    )
    np.savez_compressed(directory / f"frame_{index:05d}.npz", **arrays)
    pictures = dict(
        target=(arrays["target"] + 1) / 2,
        reconstruction=(arrays["reconstruction"] + 1) / 2,
        acquired=arrays["mask"],
        error=np.abs(arrays["target"] - arrays["reconstruction"]) / 2,
        uncertainty=arrays["uncertainty"] / 4.1,
    )
    for key, value in pictures.items():
        Image.fromarray((np.clip(value, 0, 1) * 255).round().astype("uint8")).save(
            directory / f"frame_{index:05d}_{key}.png"
        )


@torch.no_grad()
def evaluate_case(
    model,
    manifest,
    name,
    stage,
    device,
    policy="greedy",
    output=None,
    stop=None,
    full=True,
    retain=False,
    save_samples=True,
):
    started = time.perf_counter()
    n = manifest["files"][name]["frames"]
    length = n if full else min(n, model.cfg["evaluation"]["component_frames"])
    cpu = frames(manifest, name, length=length)
    data = torch.from_numpy(cpu).to(device)
    sync(device)
    load_seconds = time.perf_counter() - started
    model.eval()
    memory = model.initial(1, device)
    rows, predictions, timings, controls = [], [], [], []
    selected = set(np.linspace(0, length - 1, 4, dtype=int).tolist())
    fixed = torch.linspace(0, 111, 14, device=device).round().long()
    for t in range(length):
        if stop:
            stop()
        target = data[t : t + 1]
        sync(device)
        t0 = time.perf_counter()
        if stage == "codec":
            prediction = model.codec.decoder(model.codec.encoder(target))
            baseline = F.interpolate(
                F.interpolate(target, size=(28, 28), mode="bilinear", align_corners=False),
                size=(112, 112),
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.zeros_like(target)
            uncertainty = torch.zeros_like(target)
        elif stage == "prior":
            if t == 0:
                # Explicit privileged input for prediction diagnostic, never an online BF frame.
                memory.z = model.codec.encoder(target)
                continue
            memory.z = model.prior(memory.z, t)
            rollout_prediction = model.codec.decoder(memory.z)
            prediction = model.codec.decoder(model.prior(model.codec.encoder(data[t - 1 : t]), t))
            baseline = data[t - 1 : t]
            mask = torch.zeros_like(target)
            uncertainty = torch.zeros_like(target)
        elif policy == "interpolation":
            oracle = ArrayOracle(target)
            observed = oracle.acquire(fixed[None])
            prediction = interpolate(observed, fixed)
            mask = (
                torch.zeros((1, 112), device=device)
                .scatter_(1, fixed[None], 1)[:, None, None]
                .expand_as(target)
            )
            uncertainty = torch.zeros_like(target)
        else:
            memory, state = model.frame(ArrayOracle(target), memory, t, policy=policy)
            prediction, mask, uncertainty = state["image"], state["mask"], state["uncertainty"]
        sync(device)
        timings.append(time.perf_counter() - t0)
        value = measures(target, prediction, mask if stage == "filter" else None)
        if stage == "filter" and policy != "interpolation":
            acquired = torch.cat(state["actions"], 1)[0].sort().values
            matched = interpolate(state["observed"], acquired)
            value["same_observation_interpolation_mse"] = float((matched - target).square().mean())
        if stage == "prior":
            value["privileged_first_frame_rollout_mse"] = float(
                (target - rollout_prediction).square().mean()
            )
        row = dict(frame=t, frame_seconds=timings[-1], **value)
        if stage == "filter":
            if not torch.equal(mask[:, 0, 0].sum(1), torch.full((1,), 14.0, device=device)):
                raise AssertionError("Duplicate/missing physical lines")
            if value["observed_max_error"] != 0:
                raise AssertionError("Hard projection failed")
            row.update(actual_lines=14, k1=10, k2=4)
        else:
            controls.append(measures(target, baseline))
        rows.append(row)
        if retain:
            predictions.append(prediction[0, 0].cpu().numpy())
        if output is not None and save_samples and t in selected:
            save_sample(Path(output) / "samples", t, target, prediction, mask, uncertainty)
    warm = np.array(timings[1:] if len(timings) > 1 else timings)
    metric_keys = list(value)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in metric_keys}
    third = max(1, len(rows) // 3)
    summary.update(
        case=name,
        frames=length,
        evaluated_frames=len(rows),
        full_video=full,
        early_mse=float(np.mean([r["mse"] for r in rows[:third]])),
        late_mse=float(np.mean([r["mse"] for r in rows[-third:]])),
        cold_frame_seconds=timings[0],
        warm_frame_seconds=float(warm.mean()),
        steady_fps=float(1 / warm.mean()),
        warm_p50_s=float(np.percentile(warm, 50)),
        warm_p95_s=float(np.percentile(warm, 95)),
        load_transfer_seconds=load_seconds,
        total_seconds=time.perf_counter() - started,
        timing_scope="synchronized frame model/selection/oracle; excludes quality and file output; full wall time separate",
        policy=policy,
        stage=stage,
    )
    if stage == "prior":
        summary["control_scope"] = (
            "one-step prior vs copy-last share true previous frame; recurrent diagnostic uses only privileged first frame; neither is an online BF result"
        )
    if controls:
        summary["control"] = average(controls)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "frames.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary, np.stack(predictions) if retain else None


def validate(model, manifest, stage, device, stop=None):
    names = manifest["cohorts"]["development"]
    rows = [
        evaluate_case(model, manifest, n, stage, device, stop=stop, full=stage == "filter")[0]
        for n in names
    ]
    return dict(
        mse=float(np.mean([r["mse"] for r in rows])),
        mae=float(np.mean([r["mae"] for r in rows])),
        psnr=float(np.mean([r["psnr"] for r in rows])),
        ssim=float(np.mean([r["ssim"] for r in rows])),
        cases=rows,
    )


def final_evaluation(model, manifest, output, device, identity, stop, ef=None):
    rows = []
    output = Path(output)
    for cohort in ["development", "test"]:
        for policy in ["greedy", "uniform", "interpolation"]:
            for i, name in enumerate(manifest["cohorts"][cohort]):
                folder = output / cohort / policy / Path(name).stem
                marker = folder / "complete.json"
                if marker.exists():
                    value = read(marker)
                    if value["identity"] != identity:
                        raise ValueError("Evaluation identity mismatch")
                    for relative, checksum in value["artifacts"].items():
                        if sha(folder / relative) != checksum:
                            raise ValueError("Committed evaluation artifact changed")
                else:
                    metrics, images = evaluate_case(
                        model,
                        manifest,
                        name,
                        "filter",
                        device,
                        policy,
                        folder,
                        stop,
                        full=True,
                        retain=ef is not None,
                        save_samples=i < 3 and policy != "interpolation",
                    )
                    if ef is not None:
                        metrics["ef"] = ef(images, manifest["files"][name]["ef"])
                    artifacts = {
                        str(p.relative_to(folder)).replace("\\", "/"): sha(p)
                        for p in folder.rglob("*")
                        if p.is_file() and p != marker
                    }
                    value = dict(identity=identity, cohort=cohort, artifacts=artifacts, **metrics)
                    atomic(marker, value)
                rows.append(value)
                print(
                    __import__("json").dumps(
                        dict(
                            event="EVALUATE",
                            cohort=cohort,
                            policy=policy,
                            case=name,
                            progress=i + 1,
                            total=len(manifest["cohorts"][cohort]),
                            mse=value["mse"],
                            psnr=value["psnr"],
                        )
                    ),
                    flush=True,
                )
    return rows
