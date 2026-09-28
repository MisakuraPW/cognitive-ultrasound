"""Isolated real 200-update pilots; fixed batches and explicit checkpoint inventory.

This does not change the legacy trainer's epoch-boundary resume guarantee.
"""

import json
import time
from pathlib import Path

import numpy as np

from ..config import load
from ..preparation.common import atomic_json, atomic_npz, read_json
from .protocol import numeric


def prepare_batches(cfg, stage, output):
    from ..belief_filter.runner import Clips

    base = load(Path(cfg["bf_config"]))
    base.update(data_root=cfg["data_root"], split_manifest=cfg["split_manifest"], seed=42)
    clips = Clips(base, "train", base["train_cases"])
    batch = cfg["training_batch_size"] if stage == "casl" else 1
    batches = []
    rng = np.random.default_rng(42)
    for _ in range(200):
        batches.append(
            [
                (int(i), int(rng.integers(clips.lengths[i] - clips.frames + 1)))
                for i in rng.integers(len(clips.names), size=batch)
            ]
        )
    atomic_json(
        output,
        dict(
            batches=batches,
            names=clips.names,
            identity=clips.identity,
            batch_size=batch,
            clip_frames=clips.frames,
            updates=200,
        ),
    )


def worker(task, cfg, root, output):
    from ..belief_filter.runner import Clips, runtime

    tf = runtime(cpu=task.get("cpu_functional_test", False))
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(42)
    tf.keras.mixed_precision.set_global_policy("float32")
    stage, mode = task["workload"], task["mode"]
    spec = load(Path(cfg["bf_config"]))
    spec.update(
        data_root=cfg["data_root"],
        split_manifest=cfg["split_manifest"],
        seed=42,
        training_policy=cfg.get("bf_training_policy", "uniform"),
    )
    clips = Clips(spec, "train", spec["train_cases"])
    batches = read_json(root / "training" / (stage + ".batches.json"))
    if clips.identity != batches["identity"] or clips.names != batches["names"]:
        raise ValueError("Training input identity changed")
    if stage == "casl":
        from ..official import activate

        activate("tensorflow")
        from zea.models.diffusion import DiffusionModel

        spec = load(Path(cfg["casl_training_config"]))
        model = DiffusionModel(
            (112, 112, 3),
            input_range=(-1, 1),
            min_signal_rate=spec["min_signal_rate"],
            max_signal_rate=spec["max_signal_rate"],
            network_kwargs=spec["network_kwargs"],
            ema_val=spec["ema"],
            guidance="dps",
            operator="inpainting",
        )
        dummy = tf.zeros((1, 112, 112, 3))
        model.denoise(
            dummy, tf.ones((1, 1, 1, 1)) * 0.5, tf.ones((1, 1, 1, 1)) * 0.5, training=True
        )
        model.denoise(
            dummy, tf.ones((1, 1, 1, 1)) * 0.5, tf.ones((1, 1, 1, 1)) * 0.5, training=False
        )
        opt = tf.keras.optimizers.AdamW(spec["learning_rate"], weight_decay=spec["weight_decay"])
        variables = model.network.trainable_variables
        groups = dict(network=model.network.weights, ema=model.ema_network.weights)
        frozen = {}

        def objective(data, start, noise, times):
            nr, sr = model.diffusion_schedule(times)
            pn, pi = model.denoise(sr * data + nr * noise, nr, sr, training=True)
            return tf.reduce_mean(tf.abs(noise - pn))

        def ema():
            for w, e in zip(model.network.weights, model.ema_network.weights):
                e.assign(spec["ema"] * e + (1 - spec["ema"]) * w)
    else:
        from ..belief_filter.core import Models
        from ..belief_filter.runner import load_checkpoint, loss_for_clip

        models = Models(spec)
        parent = cfg.get("bf_initial", {}).get(stage)
        if parent:
            load_checkpoint(Path(parent), models)
        variables = models.configure_stage(stage)
        opt = tf.keras.optimizers.Adam(spec["learning_rate"])
        groups = {name: m.weights for name, m in models.all().items()}
        train_ids = {id(v) for v in variables}
        frozen = {
            f"{name}_{i}": v.numpy().copy()
            for name, values in groups.items()
            for i, v in enumerate(values)
            if id(v) not in train_ids
        }
        if mode == "graph":
            from ..belief_filter.kernels import make_kernels

            _, loss_fn = make_kernels(models, opt, stage, spec)
        else:

            def loss_fn(data, start):
                return loss_for_clip(models, data, int(start), stage, spec)

        def objective(data, start, noise, times):
            return loss_fn(data, start)

        def ema():
            pass

    opt.build(variables)
    groups["optimizer"] = opt.variables

    def snapshot():
        return {
            f"{name}_{i}": v.numpy()
            for name, values in groups.items()
            for i, v in enumerate(values)
        }

    def restore(file):
        with np.load(file, allow_pickle=False) as z:
            for name, values in groups.items():
                for i, v in enumerate(values):
                    v.assign(z[f"{name}_{i}"])
            return int(z["step"])

    initial = root / "training" / (stage + ".initial.npz")
    if task.get("initialize"):
        atomic_npz(initial, step=np.array(0), **snapshot())
        atomic_json(
            output / "result.json",
            dict(
                status="completed",
                initialized=True,
                source_parent=cfg.get("bf_initial", {}).get(stage),
                components={k: len(v) for k, v in groups.items()},
            ),
        )
        return
    begin = restore(Path(task["resume_from"]) if task.get("resume_from") else initial)
    # Frozen check must use the restored snapshot, not a newly initialized network.
    restored = snapshot()
    frozen = {k: restored[k].copy() for k in frozen}

    def update(data, start, noise, times):
        with tf.GradientTape() as tape:
            loss = objective(data, start, noise, times)
        grads = tape.gradient(loss, variables)
        if any(g is None for g in grads):
            raise ValueError("Missing training gradient")
        tf.debugging.assert_all_finite(loss, "Nonfinite loss")
        for g in grads:
            tf.debugging.assert_all_finite(g, "Nonfinite gradient")
        used = tf.clip_by_global_norm(grads, 1.0)[0] if stage != "casl" else grads
        opt.apply_gradients(zip(used, variables))
        ema()
        return loss, grads

    fn = tf.function(update, autograph=False, jit_compile=False) if mode == "graph" else update
    records, setup_s = [], 0.0
    for step in range(begin, task["until"]):
        read_start = time.perf_counter()
        batch = [clips.read(i, start, clips.frames) for i, start in batches["batches"][step]]
        if stage == "casl":
            data = np.stack([np.moveaxis(x[..., 0], 0, -1) for x in batch])
        else:
            data = batch[0]
        rng = np.random.default_rng(420000 + step)
        noise = rng.standard_normal(data.shape).astype(np.float32)
        times = rng.uniform(0.0, 1.0, size=(len(data), 1, 1, 1)).astype(np.float32)
        inputs = (
            tf.constant(data),
            tf.constant(batches["batches"][step][0][1]),
            tf.constant(noise),
            tf.constant(times),
        )
        io_s = time.perf_counter() - read_start
        tick = time.perf_counter()
        loss, grads = fn(*inputs)
        value = float(loss.numpy())
        gradient_arrays = [g.numpy() for g in grads]  # sync and validate full gradient
        elapsed = time.perf_counter() - tick
        if step == begin:
            setup_s = elapsed
        state = snapshot()
        if any(not np.isfinite(v).all() for v in state.values()):
            raise FloatingPointError("Nonfinite optimizer/parameter/EMA")
        if any(not np.array_equal(state[k], v) for k, v in frozen.items()):
            raise ValueError("Frozen module changed")
        records.append(
            dict(
                step=step + 1,
                loss=value,
                seconds=elapsed,
                io_s=io_s,
                gradient_norm=float(
                    np.sqrt(sum(np.square(g.astype(float)).sum() for g in gradient_arrays))
                ),
            )
        )
        if (step + 1) % 25 == 0:
            print(
                json.dumps(dict(event="train", workload=stage, mode=mode, **records[-1])),
                flush=True,
            )
    atomic_npz(output / "checkpoint.npz", step=np.array(task["until"]), **snapshot())
    atomic_npz(output / "gradient.npz", **{f"g{i}": g for i, g in enumerate(gradient_arrays)})
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            workload=stage,
            mode=mode,
            begin=begin,
            end=task["until"],
            records=records,
            first_update_with_compile_s=setup_s,
            frozen_verified=True,
            state_components=list(groups),
            rng="explicit per-global-step numpy RNG, fixed batch manifest; no mutable training RNG",
            calibration_isolated=True,
            production_training_advanced=False,
            ema="included" if stage == "casl" else "not part of BF workload",
            precision="fp32",
            convergence_equivalence_claim=False,
        ),
    )


def compare(a, b):
    checks = {}
    for file in ("checkpoint.npz", "gradient.npz"):
        with np.load(a / file, allow_pickle=False) as x, np.load(b / file, allow_pickle=False) as y:
            if set(x.files) != set(y.files):
                return dict(passed=False, reason="checkpoint inventory differs")
            checks[file] = {k: numeric(x[k], y[k], k == "step") for k in x.files}
    passed = all(c["passed"] for group in checks.values() for c in group.values())
    return dict(
        passed=passed,
        bitwise=all(c["bitwise"] for g in checks.values() for c in g.values()),
        checks=checks,
        status="tolerance_equivalent" if passed else "not_equivalent",
    )
