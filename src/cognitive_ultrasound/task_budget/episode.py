"""Hard, causal two-stage rollouts plus explicitly local-frame GS gradient replay."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..preparation.common import metrics
from .policy import draw, st_weights
from .protocol import Observation, greedy_order, mask_bank, state_features


@jax.custom_jvp
def projected_observation(image, target, mask):
    return jnp.where(mask != 0, target, image)


@projected_observation.defjvp
def projected_jvp(primals, tangents):
    image, target, mask = primals
    di, dt, dm = tangents
    return projected_observation(image, target, mask), (1 - mask) * di + mask * dt + (
        target - image
    ) * dm


def rollout(
    cfg,
    perception,
    task,
    frames,
    params,
    method,
    seed,
    training=False,
    fixed=(10, 4),
    progress=None,
):
    rng = np.random.default_rng(seed)
    history = np.zeros((112, 112, 3), np.float32)
    masks = np.zeros_like(history)
    previous = None
    past_images, contexts, rows, images = [], [], [], []
    last_budget = 0
    fixed_array = np.asarray(fixed)
    if fixed_array.shape == (2,):
        fixed_array = np.broadcast_to(fixed_array, (len(frames), 2))
    if fixed_array.shape != (len(frames), 2) or fixed_array.dtype.kind not in "iu":
        raise ValueError("Fixed budget must be an integer pair or one pair per frame")
    for index, frame in enumerate(frames):
        start = time.perf_counter()
        if progress:
            progress(index)
        cold = previous is None
        old_image = np.zeros((112, 112), np.float32) if cold else images[-1]
        particles = np.zeros((2, 112, 112), np.float32) if cold else previous[..., -1]
        saliency_start = time.perf_counter()
        if cold:
            scores0, values0 = np.zeros(112, np.float32), np.zeros(2, np.float32)
        else:
            scores0, values0 = task.score(particles, past_images)
        score0_s = time.perf_counter() - saliency_start
        decision_start = time.perf_counter()
        state0 = state_features(
            particles, old_image, scores0, values0, history_length=index, last_budget=last_budget
        )
        first = cfg["budgets"]["first"]
        legal0 = np.ones(len(first), bool)
        fixed_now = cfg["budgets"]["fixed"] if cold else fixed_array[index].tolist()
        if method == "E0" or cold:
            k1 = fixed_now[0]
            action0, noise0 = first.index(k1), np.zeros(len(first), np.float32)
        else:
            action0, noise0 = draw(params["first"], state0, legal0, rng, training)
            k1 = first[action0]
        controller0_s = time.perf_counter() - decision_start
        order0 = greedy_order(scores0, max(first))
        bank0 = mask_bank(order0, first)
        oracle = Observation(frame)
        observation, mask1 = oracle.acquire(order0[:k1])
        # One temporal shift per frame. Stage2 only overwrites the final channel.
        current_history = np.concatenate([history[..., 1:], observation[..., None]], axis=-1)
        current_masks = np.concatenate([masks[..., 1:], mask1[..., None]], axis=-1)
        # Diffusion keys do not depend on how many policy RNG draws/second calls occurred.
        key0, key1 = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), index), 2)
        t = time.perf_counter()
        middle_samples = perception.infer(current_history, current_masks, previous, key0, cold)
        jax.block_until_ready(middle_samples)
        middle_samples = np.asarray(middle_samples, np.float32)
        stage1_s = time.perf_counter() - t
        t = time.perf_counter()
        scores1, values1 = task.score(middle_samples[..., -1], past_images)
        score1_s = time.perf_counter() - t
        decision_start = time.perf_counter()
        state1 = state_features(
            middle_samples[..., -1],
            old_image,
            scores1,
            values1,
            observation,
            mask1,
            index,
            last_budget,
        )
        second = cfg["budgets"]["second"]
        legal1 = np.array([k1 + k <= cfg["budgets"]["maximum"] for k in second])
        if method == "E0" or cold:
            k2 = fixed_now[1]
            action1, noise1 = second.index(k2), np.zeros(len(second), np.float32)
        else:
            action1, noise1 = draw(params["second"], state1, legal1, rng, training)
            k2 = second[action1]
        controller1_s = time.perf_counter() - decision_start
        order1 = greedy_order(scores1, max(second), order0[:k1])
        bank1 = mask_bank(order1, second)
        observation, final_mask = oracle.acquire(order1[:k2])
        current_history[..., -1] = observation
        current_masks[..., -1] = final_mask
        stage2_s = 0.0
        final_samples = middle_samples
        if k2:
            t = time.perf_counter()
            final_samples = perception.infer(current_history, current_masks, middle_samples, key1)
            jax.block_until_ready(final_samples)
            final_samples = np.asarray(final_samples, np.float32)
            stage2_s = time.perf_counter() - t
        prediction = np.where(final_mask.astype(bool), observation, final_samples[0, ..., -1])
        if not np.isfinite(prediction).all() or final_mask[0].sum() != k1 + k2:
            raise RuntimeError("Nonfinite output or acquisition cardinality mismatch")
        contexts.append(
            dict(
                cold=cold,
                history=history.copy(),
                masks=masks.copy(),
                previous=None if cold else previous.copy(),
                key0=key0,
                key1=key1,
                target=frame,
                state0=state0,
                state1=state1,
                legal0=legal0,
                legal1=legal1,
                action0=action0,
                action1=action1,
                noise0=noise0,
                noise1=noise1,
                bank0=bank0,
                bank1=bank1,
                k1=k1,
                k2=k2,
                prediction=prediction,
            )
        )
        rows.append(
            dict(
                frame=index,
                k1=k1,
                k2=k2,
                lines1=order0[:k1].tolist(),
                lines2=order1[:k2].tolist(),
                perception_calls=1 + int(k2 > 0),
                reverse_steps_stage1=500 if cold else cfg["perception"]["warm_steps"],
                reverse_steps_stage2=cfg["perception"]["warm_steps"] if k2 else 0,
                stage1_s=stage1_s,
                stage2_s=stage2_s,
                score_s=score0_s + score1_s,
                controller_s=controller0_s + controller1_s,
                state_before=state0.tolist(),
                state_intermediate=state1.tolist(),
                causal_task_particles_before=values0.tolist(),
                causal_task_particles_intermediate=values1.tolist(),
                frame_wall_s=time.perf_counter() - start,
                **metrics(frame[..., None], prediction[..., None], final_mask[..., None]),
            )
        )
        images.append(prediction)
        history, masks, previous = current_history, current_masks, final_samples
        past_images.append(prediction)
        span = (cfg["task"]["frames"] - 1) * cfg["task"]["period"] + 1
        past_images = past_images[-span:]
        last_budget = k1 + k2
    return np.stack(images), rows, contexts


def gs_local_objective(params, context, adjoint, perception, cfg, temperature, cost_weight, length):
    """Hard-mask forward; ST mask derivatives, detached discrete ranks and temporal history.

    K2=0 reuses the middle posterior. Its backward approximation uses only the
    direct observation projection, without a fictitious extra DPS call.
    """
    c = context
    w0 = st_weights(
        params["first"], c["state0"], c["legal0"], c["noise0"], c["action0"], temperature
    )
    w1 = st_weights(
        params["second"], c["state1"], c["legal1"], c["noise1"], c["action1"], temperature
    )
    line0, line1 = w0 @ c["bank0"], w1 @ c["bank1"]
    mask0 = jnp.broadcast_to(line0, (112, 112))
    mask2 = jnp.broadcast_to(line1, (112, 112))
    target = jnp.asarray(c["target"])
    h = jnp.concatenate([jnp.asarray(c["history"])[..., 1:], (mask0 * target)[..., None]], axis=-1)
    m = jnp.concatenate([jnp.asarray(c["masks"])[..., 1:], mask0[..., None]], axis=-1)
    infer = getattr(perception, "infer_for_replay", perception.infer)
    samples = infer(h, m, c["previous"], c["key0"])
    middle = projected_observation(samples[0, ..., -1], target, mask0)
    if c["k2"]:
        full_mask = mask0 + mask2
        h = h.at[..., -1].set(full_mask * target)
        m = m.at[..., -1].set(full_mask)
        samples = infer(h, m, samples, c["key1"])
        output = projected_observation(samples[0, ..., -1], target, full_mask)
    else:
        output = projected_observation(middle, target, mask2)
    task_term = jnp.sum(output * adjoint)
    cost = (
        w0 @ jnp.asarray(cfg["budgets"]["first"]) + w1 @ jnp.asarray(cfg["budgets"]["second"])
    ) / (112 * length)
    return task_term + cost_weight * cost, output


def gs_gradient(params, contexts, image_gradients, perception, cfg, temperature, cost_weight):
    total = jax.tree_util.tree_map(jnp.zeros_like, params)
    task_total = jax.tree_util.tree_map(jnp.zeros_like, params)
    max_replay = 0.0
    gradient_primal_delta = 0.0
    started = time.perf_counter()
    length = len(contexts)
    for c, g in zip(contexts, image_gradients):
        if c["cold"]:
            continue  # shared fixed initialization is not a policy action

        def objective(p, weight):
            return gs_local_objective(p, c, g, perception, cfg, temperature, weight, length)

        # Validate the actual physical hard replay outside AD. The GS Jacobian is
        # an explicitly approximate estimator; AD's auxiliary primal can be
        # fused differently on GPU and is recorded separately, never used as video output.
        _, out = objective(params, 0.0)
        if cfg["runtime"].get("gs_execution", "eager") == "jit":
            (_, ad_out), task_gradient = compiled_gs_gradient(
                params, c, g, perception, cfg, temperature, length
            )
        else:
            (_, ad_out), task_gradient = jax.value_and_grad(lambda p: objective(p, 0.0), has_aux=True)(
                params
            )
        if not np.isfinite(np.asarray(ad_out)).all():
            raise FloatingPointError("Nonfinite AD linearization primal")
        gradient_primal_delta = max(
            gradient_primal_delta, float(np.max(np.abs(np.asarray(ad_out) - np.asarray(out))))
        )
        if gradient_primal_delta > 2e-4:
            raise AssertionError(
                "GS automatic-differentiation primal differs from executed replay: "
                f"{gradient_primal_delta}; reject this policy update, do not treat it as harmless drift"
            )

        # Compute the inexpensive cost derivative separately, avoiding a second DPS VJP.
        def cost_fn(p):
            w0 = st_weights(
                p["first"], c["state0"], c["legal0"], c["noise0"], c["action0"], temperature
            )
            w1 = st_weights(
                p["second"], c["state1"], c["legal1"], c["noise1"], c["action1"], temperature
            )
            return (
                cost_weight
                * (
                    w0 @ jnp.asarray(cfg["budgets"]["first"])
                    + w1 @ jnp.asarray(cfg["budgets"]["second"])
                )
                / (112 * length)
            )

        gradient = jax.tree_util.tree_map(
            lambda a, b: a + b, task_gradient, jax.grad(cost_fn)(params)
        )
        if not np.isfinite(np.asarray(out)).all():
            raise FloatingPointError("Nonfinite GS replay")
        max_replay = max(max_replay, float(np.max(np.abs(np.asarray(out) - c["prediction"]))))
        total = jax.tree_util.tree_map(lambda a, b: a + b, total, gradient)
        task_total = jax.tree_util.tree_map(lambda a, b: a + b, task_total, task_gradient)
    if max_replay > 2e-4:
        raise AssertionError(f"GS hard replay differs from executed trajectory: {max_replay}")
    norms = {
        h: float(jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(v))))
        for h, v in task_total.items()
    }
    if not all(np.isfinite(x) for x in norms.values()):
        raise FloatingPointError("Nonfinite GS task gradient")
    return total, dict(
        task_gradient_norms=norms,
        hard_replay_max_abs=max_replay,
        gradient_linearization_primal_max_abs=gradient_primal_delta,
        gradient_estimator_approximate=True,
        backward_wall_s=time.perf_counter() - started,
        gradient_scope="local frame; temporal history, state features, ranking detached; "
        "zero-K2 direct-projection ST surrogate",
    )


def compiled_gs_gradient(params, context, adjoint, perception, cfg, temperature, length):
    """Two cached signatures; observations/history/noise stay dynamic, never stale constants."""
    kernels = getattr(perception, "_gs_kernels", None)
    if kernels is None:
        kernels = perception._gs_kernels = {}
    second = bool(context["k2"])
    if second not in kernels:
        def operation(p, c, g, t, n):
            c = dict(c, k2=int(second))
            return jax.value_and_grad(
                lambda q: gs_local_objective(q, c, g, perception, cfg, t, 0.0, n),
                has_aux=True,
            )(p)

        kernels[second] = jax.jit(operation)
    fields = ("state0", "state1", "legal0", "legal1", "noise0", "noise1", "action0",
              "action1", "bank0", "bank1", "target", "history", "masks", "previous", "key0", "key1")
    values = {k: jnp.asarray(context[k]) for k in fields}
    return kernels[second](params, values, jnp.asarray(adjoint), jnp.asarray(temperature), jnp.asarray(length))
