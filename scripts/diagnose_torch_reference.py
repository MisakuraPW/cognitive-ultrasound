"""Bounded, read-only two-frame diagnosis of official versus explicit-noise replay."""
# ruff: noqa: E402

import inspect
import json
import sys
from pathlib import Path

import numpy as np

from cognitive_ultrasound.official import activate
from cognitive_ultrasound.preparation.common import atomic_json, atomic_npz, read_frames, read_json

activate("jax")
import jax
import jax.numpy as jnp
import keras
from zea.func import split_seed

from cognitive_ultrasound.preparation.casl import Adapter
from cognitive_ultrasound.torch_casl.reference import load_model, matched_frame

jax.config.update("jax_default_matmul_precision", "highest")
root = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
cfg = read_json(root / "config.json")
case = read_json(root / "cases.json")["cases"][0]
targets = read_frames(cfg, "val", case, 2)
adapter = Adapter(cfg)
adapter.reset(cfg["seed"])
adapter.step(targets[0])
state = adapter.clone()
mask = np.asarray(state.mask)[None]
obs = targets[1] * mask[0, ..., -1, None]
history = np.concatenate((np.asarray(state.measurement_buffer.buffer)[..., 1:], obs), -1)
measurement = np.repeat(history[None], 2, 0)
previous = np.asarray(state.posterior_samples)
_, arrays = adapter.step(targets[1])
expected = np.asarray(adapter.state.posterior_samples)
results = {}


def compare(label, actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    result = dict(
        max_abs=float(np.max(np.abs(a - b))),
        mean_abs=float(np.mean(np.abs(a - b))),
        passed=bool(np.allclose(a, b, atol=2e-4, rtol=2e-4)),
    )
    results[label] = result
    print(label, json.dumps(result), flush=True)


def noise_fn(seed):
    seeds = split_seed(split_seed(seed, 3)[0], 2)
    return jax.vmap(lambda k: keras.random.normal((1, 112, 112, 3), seed=split_seed(k, 2)[0])[0])(
        seeds
    )


seeds = split_seed(split_seed(state.seed, 3)[0], 2)
eager_noise = np.stack(
    [np.asarray(keras.random.normal((1, 112, 112, 3), seed=split_seed(k, 2)[0]))[0] for k in seeds]
)
jit_noise = jax.jit(noise_fn)(state.seed)
compare("noise_eager_vs_jit", eager_noise, jit_noise)
posterior_fn = adapter.agent.recover._fun.keywords["posterior_sample"]
official_model = inspect.getclosurevars(posterior_fn).nonlocals["model"]
model = load_model(cfg["checkpoint"])
compare(
    "weights",
    np.concatenate([x.reshape(-1) for x in model.ema_network.get_weights()]),
    np.concatenate([x.reshape(-1) for x in official_model.ema_network.get_weights()]),
)
posterior = jax.jit(posterior_fn)(
    jnp.asarray(history), state.mask, state.posterior_samples, split_seed(state.seed, 3)[0]
)
compare("official_posterior_vs_recover", posterior, expected)


def inside_rng(y, m, p, seed):
    keys = split_seed(split_seed(seed, 3)[0], 2)

    def one(a, b, k):
        return model.posterior_sample(
            a[None],
            mask=m,
            n_steps=500,
            initial_step=450,
            initial_samples=b[None, None],
            seed=k,
            track_progress_type=None,
            omega=10.0,
        )[0, 0]

    return jax.vmap(one)(y, p, keys)


inside = jax.jit(inside_rng)(
    jnp.asarray(measurement), jnp.asarray(mask), jnp.asarray(previous), state.seed
)
compare("rng_inside_graph_vs_recover", inside, expected)
outside = posterior_fn(
    jnp.asarray(history), state.mask, state.posterior_samples, split_seed(state.seed, 3)[0]
)
compare("no_outer_jit_vs_recover", outside, expected)

# A barrier isolates the rounding boundary without changing noise values or
# any diffusion/DPS settings. This is a diagnosis, never a timed baseline.
original_reverse = model.reverse_conditional_diffusion


def barrier_reverse(*args, **kwargs):
    kwargs["initial_noise"] = jax.lax.optimization_barrier(kwargs["initial_noise"])
    return original_reverse(*args, **kwargs)


model.reverse_conditional_diffusion = barrier_reverse
barrier = jax.jit(lambda *a: inside_rng(*a))(
    jnp.asarray(measurement), jnp.asarray(mask), jnp.asarray(previous), state.seed
)
model.reverse_conditional_diffusion = original_reverse
compare("rng_barrier_vs_recover", barrier, expected)
compare("rng_barrier_vs_inside", barrier, inside)
for label, selected_model, z in (
    ("loaded_eager", model, eager_noise),
    ("loaded_jit", model, jit_noise),
    ("official_jit", official_model, jit_noise),
):
    result = jax.jit(matched_frame(selected_model, cfg["budget"]), static_argnames=("steps",))(
        jnp.asarray(measurement), jnp.asarray(mask), jnp.asarray(previous), jnp.asarray(z), steps=50
    )
    compare(label, result[0], expected)
    compare(label + "_vs_rng_barrier", result[0], barrier)
    results[label]["selected_equal"] = bool(np.array_equal(result[2], arrays["next_action"]))
atomic_npz(
    output / "warm.npz",
    measurement=measurement,
    mask=mask,
    previous=previous,
    initial_noise=eager_noise,
    jit_noise=np.asarray(jit_noise),
    expected=expected,
    selected=arrays["next_action"],
    seed=np.asarray(state.seed),
)
atomic_json(output / "diagnosis.json", results)
