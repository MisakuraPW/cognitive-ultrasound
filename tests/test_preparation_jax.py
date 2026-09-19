"""Numerical equality to upstream warm solver, real branch execution, state isolation."""

import types
from dataclasses import replace

import numpy as np
import pytest

from cognitive_ultrasound.official import activate


@pytest.fixture(scope="module", autouse=True)
def backend():
    activate("jax")
    pytest.importorskip("jax")


def test_warm_solver_full_guidance_matches_upstream_and_sparse_executes_less():
    import jax
    import jax.numpy as jnp
    from zea.models.diffusion import DiffusionModel

    from cognitive_ultrasound.preparation.sampling import warm_solver

    calls = []
    model = types.SimpleNamespace(max_t=1.0)
    model.diffusion_schedule = lambda t: (0.1 + t * 0.4, 0.9 - t * 0.4)
    model.denoise = lambda x, n, s, training=False: (x * 0.2, (x - n * x * 0.2) / s)
    model.start_track_progress = lambda *a: None
    model.store_progress = lambda *a: None
    for name in (
        "prepare_schedule",
        "prepare_diffusion",
        "reverse_diffusion_step",
        "reverse_conditional_diffusion",
    ):
        setattr(model, name, types.MethodType(getattr(DiffusionModel, name), model))

    def guidance(value, measurements, noise_rates, signal_rates, **kwargs):
        jax.debug.callback(lambda _: calls.append(1), jnp.sum(value), ordered=True)

        def loss(x):
            pn, pi = model.denoise(x, noise_rates, signal_rates)
            return jnp.mean((pi - measurements) ** 2) * 0.01, (pn, pi)

        (error, aux), grad = jax.value_and_grad(loss, has_aux=True)(value)
        return grad, (error, aux)

    model.guidance_fn = guidance
    noise = jnp.array(np.random.default_rng(4).normal(size=(1, 4, 4, 1)), dtype=jnp.float32)
    samples = jnp.zeros_like(noise)
    key = jax.random.PRNGKey(4)
    original = model.reverse_conditional_diffusion(
        samples, noise, 10, samples, 6, seed=key, track_progress_type=None
    )
    actual = warm_solver(model, samples, noise, 10, samples, 6, key, {}, {})
    jax.block_until_ready(actual)
    np.testing.assert_allclose(actual, original, rtol=1e-5, atol=1e-6)
    calls.clear()

    def one(x):
        return warm_solver(model, samples, x, 10, samples, 6, key, {"guidance_steps": 2}, {})

    jax.block_until_ready(jax.jit(one)(noise))
    jax.effects_barrier()
    assert len(calls) == 2
    calls.clear()
    jax.block_until_ready(jax.jit(jax.vmap(one))(jnp.stack([noise, noise])))
    jax.effects_barrier()
    assert len(calls) == 4  # two particles x two actual backward calls, not all four steps


def test_snapshot_restores_key_history_and_does_not_alias_branches():
    import jax
    import jax.numpy as jnp

    from cognitive_ultrasound.preparation.casl import Adapter

    adapter = Adapter.__new__(Adapter)
    adapter.jax = jax
    action = jnp.zeros(112).at[::8].set(1)
    mask = jnp.broadcast_to(action, (112, 112))[..., None]
    adapter.agent = types.SimpleNamespace(
        input_shape=(112, 112, 3), initial_action_selection=lambda **kw: (action, mask, None)
    )
    adapter.reset(42)
    adapter.state.measurement_buffer.shift(jnp.ones((112, 112, 1)))
    adapter.state = replace(
        adapter.state,
        posterior_samples=jnp.ones((2, 112, 112, 3)),
        belief_distribution=jnp.ones((2, 112, 112, 1)),
    )
    saved = adapter.arrays()
    parent = adapter.clone()
    adapter.state.measurement_buffer.shift(jnp.zeros((112, 112, 1)))
    assert not np.array_equal(
        adapter.state.measurement_buffer.buffer, parent.measurement_buffer.buffer
    )
    adapter.restore(saved)
    for key, value in saved.items():
        np.testing.assert_array_equal(adapter.arrays()[key], value)
    adapter.force_lines([1, 4, 7])
    assert np.asarray(adapter.state.mask[0, :, -1]).sum() == 3
    np.testing.assert_array_equal(adapter.state.mask[..., :-1], parent.mask[..., :-1])
