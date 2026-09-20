"""One matched BF training-history experiment, with frozen ancestors and fixed exposure."""

import time
from pathlib import Path

import numpy as np

from ..provenance import sha256
from .belief import configuration
from .common import atomic_json, atomic_npz, emit, read_frames, read_json
from .followup import interpolate_observed


def model_config(cfg, manifest, interval):
    result = configuration(cfg, manifest)
    result.update(
        clip_frames=cfg["closure"]["bf_clip_frames"],
        training_reset_interval=interval,
        filter_finetune=True,
        execution="auto",
        checkpoint_every=25,
        validation_every=50,
    )
    result["steps"] = dict(result["steps"], filter=cfg["closure"]["bf_steps"])
    return result


def train(task, cfg, manifest, output, root):
    from ..belief_filter.runner import train as run_train
    from ..belief_filter.runner import validate

    interval = task["reset_interval"]
    model_cfg = validate(model_config(cfg, manifest, interval))
    parent = Path(cfg["closure"]["bf_source"]) / "jobs/bf_filter/training/checkpoint.npz"
    before = sha256(parent)
    dest = output / "training"
    run_train(
        model_cfg,
        dest,
        "filter",
        initialize=parent,
        resume=(dest / "checkpoint.npz").exists(),
        cpu=cfg["closure"].get("cpu_only", False),
    )
    frozen = []
    with (
        np.load(parent, allow_pickle=False) as a,
        np.load(dest / "checkpoint.npz", allow_pickle=False) as b,
    ):
        for k in a.files:
            if k.startswith(("encoder_", "decoder_", "prior_")):
                if not np.array_equal(a[k], b[k]):
                    raise AssertionError("Frozen component changed: " + k)
                frozen.append(k)
    assert sha256(parent) == before
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            reset_interval=interval,
            updates=model_cfg["steps"]["filter"],
            clip_frames=model_cfg["clip_frames"],
            frame_exposure=model_cfg["steps"]["filter"] * model_cfg["clip_frames"],
            parent_sha256=before,
            frozen_arrays_checked=len(frozen),
            note="Same parent weights, new Adam, sample order, frames and update count; only state reset interval differs; no best-checkpoint selection",
        ),
    )


def qualify(task, cfg, manifest, output, root):
    from ..belief_filter.runner import load_checkpoint, runtime

    tf = runtime(cfg["closure"].get("cpu_only", False))
    from ..belief_filter.core import ArrayOracle, Models, initial_memory, step

    cohort = task["cohort"]
    model_cfg = model_config(cfg, manifest, 0)
    paths = {
        "frozen": Path(cfg["closure"]["bf_source"]) / "jobs/bf_filter/training/checkpoint.npz",
        "reset3": root / "jobs/bf_history_reset3/training/checkpoint.npz",
        "continuous12": root / "jobs/bf_history_continuous12/training/checkpoint.npz",
    }
    records = []
    for mode, checkpoint in paths.items():
        models = Models(model_cfg)
        meta = load_checkpoint(checkpoint, models)
        if meta["stage"] != "filter" or meta["status"] != "completed":
            raise ValueError("Incomplete BF weights")
        # Independent inference parity/speed probe; never updates model weights.
        original = models.update
        z = tf.TensorSpec((1, 28, 28, model_cfg["latent_channels"]), tf.float32)
        im = tf.TensorSpec((1, 112, 112, 1), tf.float32)
        graph = tf.function(original, input_signature=[z, im, im], autograph=False)
        clip = read_frames(cfg, "val", manifest["cohorts"]["debug"][0], 3)
        probe = {}
        times = {}
        for kind, fn in [("eager", original), ("graph", graph)]:
            models.update = fn
            memory = initial_memory(models)
            images = []
            tt = []
            for i, target in enumerate(clip):
                t = time.perf_counter()
                memory, state = step(models, ArrayOracle(target), memory, i, model_cfg, "uniform")
                images.append(np.asarray(state["reconstruction"]))
                tt.append(time.perf_counter() - t)
            probe[kind] = np.stack(images)
            times[kind] = tt
        parity = bool(np.allclose(probe["eager"], probe["graph"], rtol=1e-4, atol=1e-5))
        accelerated = parity and np.median(times["graph"][1:]) < 0.9 * np.median(times["eager"][1:])
        models.update = graph if accelerated else original
        atomic_json(
            output / (mode + "_preflight.json"),
            dict(
                parity=parity, execution="graph_update" if accelerated else "eager", times_s=times
            ),
        )
        for name in manifest["cohorts"][cohort]:
            directory = output / mode / Path(name).stem
            receipt = directory / "complete.json"
            if receipt.exists():
                records.append(read_json(receipt))
                continue
            frames = read_frames(
                cfg, "val", name, min(cfg["closure"]["bf_eval_frames"], cfg["frames"][cohort])
            )
            memory = initial_memory(models)
            rows = []
            for i, target in enumerate(frames):
                memory, state = step(models, ArrayOracle(target), memory, i, model_cfg, "uniform")
                prediction = np.asarray(state["reconstruction"])[0]
                mask = state["mask"][0]
                lines = np.flatnonzero(mask[0, :, 0])
                interp = interpolate_observed(state["observation"][0], lines)
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(mode)
                rows.append(
                    dict(
                        frame=i,
                        all_pixel_mae=float(abs(prediction - target).mean()),
                        unobserved_mae=float(abs(prediction - target)[mask == 0].mean()),
                        interpolation_mae=float(abs(interp - target)[mask == 0].mean()),
                    )
                )
                if i in (0, len(frames) - 1):
                    atomic_npz(
                        directory / f"frame_{i:04d}.npz",
                        target=target,
                        model=prediction,
                        interpolation=interp,
                        mask=mask,
                    )
            record = dict(
                case=name,
                mode=mode,
                rows=rows,
                mean_unobserved_mae=float(np.mean([r["unobserved_mae"] for r in rows])),
                late_unobserved_mae=float(
                    np.mean([r["unobserved_mae"] for r in rows[len(rows) // 2 :]])
                ),
            )
            atomic_json(receipt, record)
            records.append(record)
            emit(
                "BF_HISTORY_EVAL",
                cohort=cohort,
                case=name,
                mode=mode,
                error=record["mean_unobserved_mae"],
            )
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            records=records,
            checkpoint_hashes={k: sha256(p) for k, p in paths.items()},
            means={
                k: float(np.mean([r["mean_unobserved_mae"] for r in records if r["mode"] == k]))
                for k in paths
            },
            note="Both fixed final checkpoints evaluated; no gate-driven extra training. Frozen codec/prior, matched exposure; limited pilot does not reproduce senior thesis.",
        ),
    )
