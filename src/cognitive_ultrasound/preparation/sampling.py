"""Opt-in sampling variants; the pinned upstream repository is never modified."""

import types

import numpy as np

VARIANTS = {
    "reference": dict(initial_step=450, precision="float32"),
    "wrapper": dict(initial_step=450, precision="float32", wrapper=True),
    "profile": dict(initial_step=450, precision="float32", jit_mode="posterior_sample"),
    "official25": dict(initial_step=475, precision="float32"),
    "official25_fp16": dict(initial_step=475, precision="mixed_float16"),
    "sparse10": dict(initial_step=450, precision="float32", guidance_steps=10),
    "sparse5": dict(initial_step=450, precision="float32", guidance_steps=5),
    "coarse10": dict(initial_step=450, precision="float32", inference_steps=10),
}


def guidance_indices(steps, count):
    if not 1 <= count <= steps:
        raise ValueError("Guidance count must be in [1, inference steps]")
    if count == 1:
        return np.array([steps - 1], dtype=np.int32)
    return np.linspace(0, steps - 1, count).round().astype(np.int32)


def attach_sampler(model, variant):
    """Keep upstream cold start; patch only this instance's warm conditional solver."""
    original = model.reverse_conditional_diffusion

    def reverse(
        self,
        measurements,
        initial_noise,
        diffusion_steps,
        initial_samples=None,
        initial_step=0,
        stochastic_sampling=False,
        seed=None,
        verbose=False,
        track_progress_type=None,
        disable_jit=False,
        **kwargs,
    ):
        if initial_samples is None or initial_step == 0:
            return original(
                measurements,
                initial_noise,
                diffusion_steps,
                initial_samples,
                initial_step,
                stochastic_sampling,
                seed,
                verbose,
                track_progress_type,
                disable_jit,
                **kwargs,
            )
        if stochastic_sampling or verbose or track_progress_type:
            raise ValueError("Preparation variants support deterministic, untracked DDIM only")
        return warm_solver(
            self,
            measurements,
            initial_noise,
            diffusion_steps,
            initial_samples,
            initial_step,
            seed,
            variant,
            kwargs,
        )

    model.reverse_conditional_diffusion = types.MethodType(reverse, model)


def warm_solver(
    model,
    measurements,
    initial_noise,
    diffusion_steps,
    initial_samples,
    initial_step,
    seed,
    variant,
    kwargs,
):
    import jax
    import jax.numpy as jnp
    from zea.func import split_seed

    native_steps = diffusion_steps - initial_step
    steps = variant.get("inference_steps", native_steps)
    guides = variant.get("guidance_steps", steps)
    selected = np.zeros(steps, dtype=bool)
    selected[guidance_indices(steps, guides)] = True
    flags = jnp.asarray(selected)
    base = jnp.ones((len(initial_noise), *([1] * (initial_noise.ndim - 1)))) * model.max_t
    # Preserve the upstream warm-start noise convention (initial_step - 1).
    x = model.prepare_schedule(
        base, initial_noise, initial_samples, initial_step, model.max_t / diffusion_steps
    )
    delta = model.max_t * native_steps / diffusion_steps / steps

    def step(i, state):
        noisy, _, key = state
        t = base - initial_step * (model.max_t / diffusion_steps) - i * delta
        noise, signal = model.diffusion_schedule(t)
        next_noise, next_signal = model.diffusion_schedule(t - delta)

        def guided(value):
            gradient, (_, (pn, pi)) = model.guidance_fn(
                value, measurements=measurements, noise_rates=noise, signal_rates=signal, **kwargs
            )
            return gradient, pn, pi

        def unguided(value):
            pn, pi = model.denoise(value, noise, signal, training=False)
            return jnp.zeros_like(value), pn, pi

        # Scalar condition is shared across vmap particles, avoiding vmap-select execution
        # of both branches. No guidance/backward pass runs in the unguided branch.
        gradient, pn, pi = jax.lax.cond(flags[i], guided, unguided, noisy)
        key, child = split_seed(key, 2)
        nxt = model.reverse_diffusion_step(
            initial_noise.shape,
            pi,
            pn,
            signal,
            next_signal,
            next_noise,
            seed=child,
            stochastic_sampling=False,
        )
        return nxt - gradient, pi - gradient, key

    return jax.lax.fori_loop(0, steps, step, (x, jnp.zeros_like(x), seed))[1]
