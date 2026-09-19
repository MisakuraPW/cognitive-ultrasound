"""Bounded thesis-inspired stages and independent simple-baseline qualification."""

import json
import time
from pathlib import Path

import numpy as np

from ..config import ROOT, load
from .common import atomic_json, atomic_npz, emit, metrics, read_frames, read_json


def configuration(cfg, manifest):
    result = load(ROOT / "configs/belief_filter/pilot.yaml")
    result.update(
        data_root=cfg["data_root"],
        split_manifest=cfg["split_manifest"],
        seed=cfg["seed"],
        steps=cfg["bf"]["steps"],
        training_policy="uniform",
        checkpoint_every=cfg["bf"]["chunk_steps"],
        validation_every=cfg["bf"]["chunk_steps"],
        case_names={
            "train": manifest["cohorts"]["train"],
            "val": manifest["cohorts"]["development"],
        },
    )
    return result


def train_stage(task, cfg, manifest, output, root):
    from ..belief_filter.runner import train

    stage = task["stage"]
    model_cfg = configuration(cfg, manifest)
    directory = output / "training"
    parent = {"prior": "codec", "filter": "prior"}.get(stage)
    initialize = root / "jobs" / ("bf_" + parent) / "training/checkpoint.npz" if parent else None
    while True:
        checkpoint = directory / "checkpoint.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as archive:
                saved = json.loads(str(archive["metadata"]))
            if saved["status"] == "completed":
                break
        train(
            model_cfg,
            directory,
            stage,
            initialize,
            resume=checkpoint.exists(),
            cpu=task.get("cpu", False),
            stop_after=cfg["bf"]["chunk_steps"],
        )
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            stage=stage,
            steps=model_cfg["steps"][stage],
            full_reproduction=False,
        ),
    )


def qualify(task, cfg, manifest, output, root):
    from ..belief_filter.runner import load_checkpoint, runtime

    tf = runtime(task.get("cpu", False))
    from ..belief_filter.core import ArrayOracle, Models, initial_memory, step

    stage = task["stage"]
    cohort = task.get("cohort", "development")
    model_cfg = configuration(cfg, manifest)
    models = Models(model_cfg)
    metadata = load_checkpoint(root / "jobs" / ("bf_" + stage) / "training/checkpoint.npz", models)
    if metadata["stage"] != stage or metadata["status"] != "completed":
        raise ValueError("Qualification requires a completed matching stage")
    records = []
    for name in manifest["cohorts"][cohort]:
        directory = output / Path(name).stem
        complete = directory / "complete.json"
        if complete.exists():
            records.append(read_json(complete))
            continue
        frames = read_frames(cfg, "val", name, cfg["frames"][cohort])
        sums = {}
        times = []
        memory = initial_memory(models)
        # Full first-frame encoding is only allowed in the explicitly privileged prior diagnostic.
        prior_z = (
            models.encoder(frames[:1])
            if stage == "prior"
            else tf.zeros((1, 28, 28, model_cfg["latent_channels"]))
        )
        baseline_z = tf.zeros_like(prior_z)
        for index, target in enumerate(frames):
            started = time.perf_counter()
            comparisons = {}
            mask = np.zeros_like(target)
            if stage == "codec":
                prediction = np.asarray(models.decoder(models.encoder(target[None])))[0]
                small = tf.image.resize(target[None], (28, 28), method="bilinear")
                comparisons["resize"] = np.asarray(tf.image.resize(small, (112, 112)))[0]
            elif stage == "prior":
                if index == 0:
                    continue
                old = models.encoder(frames[index - 1 : index])
                prediction = np.asarray(models.decoder(models.predict(old, index)))[0]
                comparisons["copy_last"] = frames[index - 1]
                comparisons["copy_codec"] = np.asarray(models.decoder(old))[0]
                prior_z = models.predict(prior_z, index)
                comparisons["privileged_rollout"] = np.asarray(models.decoder(prior_z))[0]
            else:
                memory, state = step(
                    models, ArrayOracle(target), memory, index, model_cfg, "uniform"
                )
                prediction = np.asarray(state["reconstruction"])[0]
                mask = state["mask"][0]
                observation = state["observation"][0]
                # Independent baseline state: it never consumes learned-filter output or truth.
                baseline_prior = models.predict(baseline_z, index)
                baseline_prediction = np.asarray(models.decoder(baseline_prior))[0]
                for count in np.cumsum(model_cfg["groups"]):
                    lines = np.linspace(0, 111, sum(model_cfg["groups"]), dtype=int)[:count]
                    acquired = np.zeros_like(mask)
                    acquired[:, lines, :] = 1
                    baseline_prediction = np.where(acquired > 0, observation, baseline_prediction)
                    baseline_z = models.encoder(baseline_prediction[None])
                    if count < sum(model_cfg["groups"]):
                        baseline_prediction = np.asarray(models.decoder(baseline_z))[0]
                comparisons["prior_projection"] = baseline_prediction
                lines = np.flatnonzero(mask[0, :, 0])
                comparisons["spatial_interpolation"] = np.stack(
                    [np.interp(np.arange(112), lines, observation[r, lines, 0]) for r in range(112)]
                )[..., None].astype("float32")
            times.append(time.perf_counter() - started)
            comparisons = {"model": prediction, **comparisons}
            for label, image in comparisons.items():
                if not np.isfinite(image).all():
                    raise FloatingPointError(label)
                score = metrics(target, image, mask)
                sums.setdefault(label, []).append(score["unobserved_mae"])
            if index < 3:
                atomic_npz(
                    directory / f"frame_{index:04d}.npz", target=target, mask=mask, **comparisons
                )
        record = dict(
            case=name,
            errors={k: float(np.mean(v)) for k, v in sums.items()},
            median_comparison_wall_s=float(np.median(times)),
            frames=len(times),
        )
        atomic_json(complete, record)
        records.append(record)
        emit("bf_qualification", stage=stage, case=name, errors=record["errors"])
    means = {k: float(np.mean([r["errors"][k] for r in records])) for k in records[0]["errors"]}
    if stage == "codec":
        eligible = means["model"] <= cfg["bf"]["codec_mae_limit"]
        reason = "Absolute codec reconstruction tolerance; resize baseline reported separately"
    elif stage == "prior":
        eligible = means["model"] <= means["copy_codec"] * cfg["bf"]["prior_mae_ratio"]
        reason = "Teacher-forced prediction must not materially lose to codec persistence; rollout is privileged diagnostic"
    else:
        baseline = min(means["prior_projection"], means["spatial_interpolation"])
        eligible = means["model"] <= baseline * cfg["bf"]["filter_mae_ratio"]
        reason = "Equal fixed observations; learned update compared with independent projection and interpolation"
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            eligible=bool(eligible),
            stage=stage,
            cohort=cohort,
            patient_mean_unobserved_mae=means,
            cases=records,
            interpretation=reason,
            caveat="Thesis-inspired bounded pilot; failure is not proof the thesis direction fails",
        ),
    )
