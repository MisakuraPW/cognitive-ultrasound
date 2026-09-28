"""Separate instrumentation and finite capacity probes; never used for FPS ranking."""

import time

import h5py
import numpy as np

from ..preparation.common import atomic_json, read_json
from .data import read_one
from .inference import cache_path, read_group
from .protocol import Profile


def profile(task, cfg, root, output):
    from .engines import JaxEngine, TorchEngine

    p = Profile(**task["profile"])
    manifest = read_json(root / "manifest.json")
    name = manifest["cohorts"]["debug"][0]
    with h5py.File(cache_path(root, "debug", 42, 14, name)) as source:
        previous, current = read_group(source, 1), read_group(source, 2)
    from pathlib import Path

    target = read_one(Path(cfg["data_root"]) / "val" / name, 2)
    cls = JaxEngine if p.backend == "jax" else TorchEngine
    engine = cls(cfg, p, 14, root / "export")
    for _ in range(3):
        engine.restore(previous)
        engine.step(target, current["noise"])
    engine.restore(previous)
    if p.backend == "jax":
        import jax

        with jax.profiler.trace(str(output / "trace"), create_perfetto_link=False):
            engine.step(target, current["noise"])
    else:
        import torch

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            engine.step(target, current["noise"])
        prof.export_chrome_trace(str(output / "trace.json"))
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            instrumented=True,
            excluded_from_speed_ranking=True,
            profile=p.record(),
        ),
    )


def capacity(task, cfg, root, output):
    """Real CASL objective/optimizer probe; advisory batch never updates scientific cfg."""
    from pathlib import Path

    from ..belief_filter.runner import runtime
    from ..config import load
    from ..official import activate

    tf = runtime()
    activate("tensorflow")
    from zea.models.diffusion import DiffusionModel

    spec = load(Path(cfg["casl_training_config"]))
    spec_data = read_json(root / "training/casl.batches.json")
    from ..belief_filter.runner import Clips

    base = load(Path(cfg["bf_config"]))
    base.update(data_root=cfg["data_root"], split_manifest=cfg["split_manifest"], seed=42)
    clips = Clips(base, "train", base["train_cases"])
    model = DiffusionModel(
        (112, 112, 3),
        input_range=(-1, 1),
        network_kwargs=spec["network_kwargs"],
        guidance="dps",
        operator="inpainting",
    )
    batch = task["batch"]
    choices = spec_data["batches"][0]
    data = tf.constant(
        np.stack(
            [
                np.moveaxis(clips.read(*choices[i % len(choices)], 3)[..., 0], 0, -1)
                for i in range(batch)
            ]
        )
    )
    opt = tf.keras.optimizers.AdamW(spec["learning_rate"], weight_decay=spec["weight_decay"])
    noise = tf.random.stateless_normal(data.shape, [42, 1])
    rates = tf.ones((batch, 1, 1, 1)) * 0.5
    tick = time.perf_counter()
    with tf.GradientTape() as tape:
        pn, _ = model.denoise(data + noise, rates, rates, training=True)
        loss = tf.reduce_mean(tf.abs(noise - pn))
    grads = tape.gradient(loss, model.network.trainable_variables)
    for g in grads:
        tf.debugging.assert_all_finite(g, "capacity gradient")
    opt.apply_gradients(zip(grads, model.network.trainable_variables))
    value = float(loss.numpy())
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            advisory_only=True,
            batch=batch,
            loss=value,
            seconds=time.perf_counter() - tick,
            memory=tf.config.experimental.get_memory_info("GPU:0"),
            scientific_batch_size=cfg["training_batch_size"],
            adopted=False,
            limitation="single update capacity, not convergence or throughput; inputs repeated when batch > 32",
        ),
    )
