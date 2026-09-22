"""Isolated JAX worker: export EMA and exact-input reference fixtures."""

import time
from pathlib import Path

import numpy as np

from ..official import activate
from ..preparation.common import atomic_json, atomic_npz, read_frames
from ..provenance import sha256


def export_network(model, checkpoint, output):
    import keras

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    nodes, arrays = [], {}
    allowed = {
        "InputLayer",
        "Conv2D",
        "BatchNormalization",
        "Lambda",
        "UpSampling2D",
        "AveragePooling2D",
        "Concatenate",
        "Add",
    }
    for layer in model.ema_network.layers:
        kind, cfg = type(layer).__name__, layer.get_config()
        if kind not in allowed:
            raise ValueError(f"Unsupported EMA layer: {kind}")
        if kind == "Conv2D":
            assert cfg["activation"] in ("linear", "swish", "silu")
            assert cfg["strides"] == (1, 1) or cfg["strides"] == [1, 1]
            assert cfg["groups"] == 1 and cfg["use_bias"]
            assert tuple(cfg["dilation_rate"]) == (1, 1)
        if kind == "BatchNormalization":
            assert not cfg["center"] and not cfg["scale"] and cfg["axis"] == -1
        if kind == "AveragePooling2D":
            assert cfg["padding"] == "valid"
        if kind == "Concatenate":
            assert cfg["axis"] == -1
        if kind == "Lambda":
            assert layer.function.__name__ == "_sinusoidal_embedding"
            cfg = {}  # Never serialize executable Lambda bytecode.
        if cfg.get("data_format") not in (None, "channels_last"):
            raise ValueError("Expected channels_last Keras export")
        weights = []
        for i, value in enumerate(layer.get_weights()):
            key = f"{layer.name}_{i}"
            arrays[key] = value
            weights.append(key)
        inbound = (
            []
            if kind == "InputLayer"
            else [x._keras_history.operation.name for x in keras.tree.flatten(layer.input)]
        )
        nodes.append(dict(name=layer.name, kind=kind, config=cfg, inputs=inbound, weights=weights))
    cfg = model.get_config()
    if cfg["network_name"] != "unet_time_conditional" or list(cfg["input_shape"]) != [112, 112, 3]:
        raise ValueError("Only pinned CASL 112x112 W=3 UNet is supported")
    atomic_npz(output / "ema.npz", **arrays)
    atomic_json(
        output / "network.json",
        dict(
            nodes=nodes,
            inputs=[x._keras_history.operation.name for x in model.ema_network.inputs],
            output=model.ema_network.output._keras_history.operation.name,
            model_config=cfg,
            checkpoint_sha256=sha256(Path(checkpoint) / "model.weights.h5"),
            ema_sha256=sha256(output / "ema.npz"),
            frozen=True,
        ),
    )


def load_model(checkpoint, cpu=False):
    if cpu:
        import os

        os.environ["JAX_PLATFORMS"] = "cpu"
    activate("jax")
    from zea.models.diffusion import DiffusionModel

    return DiffusionModel.from_preset(
        str(checkpoint),
        guidance={"name": "dps", "params": {"disable_jit": True}},
        operator={"name": "inpainting", "params": {"min_val": 0}},
    )


def probes(model, output):
    import jax
    import jax.numpy as jnp
    from zea.agent.selection import GreedyEntropy

    rng = np.random.default_rng(721)
    x = rng.normal(size=(2, 112, 112, 3)).astype("float32")
    mask = np.zeros((1, 112, 112, 3), "float32")
    mask[:, :, ::8] = 1
    measurement = rng.uniform(-1, 1, size=x.shape).astype("float32") * mask
    noise, signal = model.diffusion_schedule(jnp.full((2, 1, 1, 1), 0.08))
    pred = model.ema_network([jnp.asarray(x), noise**2], training=False)

    def grad_one(a, y, n, s):
        return model.guidance_fn(
            a[None],
            measurements=y[None],
            mask=jnp.asarray(mask),
            noise_rates=n[None],
            signal_rates=s[None],
            omega=10.0,
        )[0][0]

    gradient = jax.jit(jax.vmap(grad_one))(jnp.asarray(x), jnp.asarray(measurement), noise, signal)
    selector = GreedyEntropy(14, 112, 112, 112)
    particles = jnp.asarray(rng.normal(size=(1, 2, 112, 112)).astype("float32"))
    selected, _ = selector.sample(particles)
    entropy = selector.compute_pixelwise_entropy(particles)
    atomic_npz(
        Path(output) / "probes.npz",
        x=x,
        measurement=measurement,
        mask=mask,
        noise=np.asarray(noise),
        signal=np.asarray(signal),
        prediction=np.asarray(pred),
        gradient=np.asarray(gradient),
        particles=np.asarray(particles),
        selected=np.asarray(selected),
        entropy=np.asarray(entropy),
    )


def replay_check(samples, expected, selected, expected_selected):
    """Numerical rejection is an experimental result, not a worker exception."""
    samples, expected = np.asarray(samples), np.asarray(expected)
    if not np.isfinite(samples).all() or not np.isfinite(expected).all():
        raise FloatingPointError("Nonfinite JAX replay; no valid speed comparison")
    numerical = bool(np.allclose(samples, expected, rtol=2e-4, atol=2e-4))
    actions = bool(np.array_equal(selected, expected_selected))
    return dict(
        passed=numerical and actions,
        numerical=numerical,
        selected_equal=actions,
        max_abs=float(np.max(np.abs(samples - expected))),
        mean_abs=float(np.mean(np.abs(samples - expected))),
    )


def run_reference(cfg, output):
    activate("jax")
    import jax
    import jax.numpy as jnp
    import keras
    from zea.func import split_seed

    from ..preparation.casl import Adapter

    if jax.default_backend() != "gpu":
        raise RuntimeError("Reference performance measurement requires GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    model = load_model(cfg["checkpoint"])
    export_network(model, cfg["checkpoint"], output / "export")
    probes(model, output / "export")
    rows = []
    # Both sides exclude RNG from matched-input timing.
    kernel = jax.jit(matched_frame(model, cfg["budget"]), static_argnames=("steps",))
    from ..data import read_splits

    names = (
        np.random.default_rng(cfg["seed"])
        .permutation(read_splits(cfg["split_manifest"])["val"])
        .tolist()[: cfg["cases"]]
    )
    # Lock cases before inspecting lengths/outcomes. Never replace a bad case with a better one.
    atomic_json(output / "cases.json", dict(split="val", cases=names, seed=cfg["seed"]))
    for steps in cfg["steps"]:
        adapter = Adapter(cfg, "reference" if steps == 50 else "official25")
        for ci, name in enumerate(names):
            targets = read_frames(cfg, "val", name, cfg["frames"])
            adapter.reset(cfg["seed"] + ci)
            for index, target in enumerate(targets):
                state = adapter.clone()
                seed1 = split_seed(state.seed, 3)[0]
                seeds = split_seed(seed1, 2)
                z = np.stack(
                    [
                        np.asarray(keras.random.normal((1, 112, 112, 3), seed=split_seed(k, 2)[0]))[
                            0
                        ]
                        for k in seeds
                    ]
                )
                mask = np.asarray(state.mask)[None]
                obs = target * mask[0, ..., -1, None]
                history = np.concatenate(
                    (np.asarray(state.measurement_buffer.buffer)[..., 1:], obs), axis=-1
                )
                measurement = np.repeat(history[None], 2, axis=0)
                previous = np.zeros_like(z) if index == 0 else np.asarray(state.posterior_samples)
                original_row, arrays = adapter.step(target)
                expected = np.asarray(adapter.state.posterior_samples)
                inputs = tuple(map(jnp.asarray, (measurement, mask, previous, z)))
                matched_times, setup_s = [], None
                check = None
                matched_samples, matched_selected = expected, arrays["next_action"]
                if index:
                    tick = time.perf_counter()
                    result = kernel(*inputs, steps=steps)
                    jax.block_until_ready(result)
                    setup_s = time.perf_counter() - tick
                    matched_samples, matched_selected = map(np.asarray, (result[0], result[2]))
                    check = replay_check(
                        matched_samples, expected, matched_selected, arrays["next_action"]
                    )
                    for _ in range(cfg["repeats"]):
                        tick = time.perf_counter()
                        result = kernel(*inputs, steps=steps)
                        jax.block_until_ready(result)
                        matched_times.append(time.perf_counter() - tick)
                name_out = f"s{steps}_c{ci}_f{index:03d}"
                atomic_npz(
                    output / "fixtures" / f"{name_out}.npz",
                    measurement=measurement,
                    mask=mask,
                    previous=previous,
                    initial_noise=z,
                    target=target[..., 0],
                    samples=expected,
                    matched_samples=matched_samples,
                    matched_selected=matched_selected,
                    prediction=arrays["prediction"][..., 0],
                    selected=arrays["next_action"],
                    entropy=arrays["uncertainty"],
                )
                row = dict(
                    id=name_out,
                    steps=steps,
                    case=name,
                    frame=index,
                    cold=index == 0,
                    original=original_row,
                    matched_seconds=matched_times,
                    compile_or_first_s=setup_s,
                    replay_check=check,
                )
                rows.append(row)
                atomic_json(output / "reference.json", dict(rows=rows, completed=False))
                print(
                    f"JAX {name_out}: adapter={original_row['algorithm_s']:.3f}s replay={check}",
                    flush=True,
                )
    atomic_json(
        output / "reference.json",
        dict(rows=rows, completed=True, jax=jax.__version__, device=str(jax.devices()[0])),
    )


def matched_frame(model, budget):
    import jax
    import jax.numpy as jnp
    from zea.agent.selection import GreedyEntropy

    selector = GreedyEntropy(budget, 112, 112, 112)

    def frame(measurement, mask, previous, noise, steps):
        def one(y, p, z):
            return model.reverse_conditional_diffusion(
                y[None],
                z[None],
                500,
                p[None],
                500 - steps,
                seed=jax.random.PRNGKey(0),
                # Upstream defaults to NumPy progress snapshots, which cannot run
                # inside this jitted/vmapped numerical comparison.
                track_progress_type=None,
                verbose=False,
                mask=mask,
                omega=10.0,
            )[0]

        samples = jax.vmap(one)(measurement, previous, noise)
        selected, _ = selector.sample(samples[None, ..., -1])
        entropy = selector.compute_pixelwise_entropy(samples[None, ..., -1])[0]
        obs = measurement[0, ..., -1]
        projected = jnp.where(obs != 0, obs, samples[0, ..., -1])
        return samples, projected, selected[0], entropy

    return frame
