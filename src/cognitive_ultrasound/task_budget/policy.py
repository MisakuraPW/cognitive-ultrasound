"""Two observable-state MLP heads, straight-through GS and REINFORCE, shared Adam."""

import jax
import jax.numpy as jnp
import numpy as np

from ..preparation.common import atomic_npz
from .protocol import FEATURES


def initialize(seed, cfg):
    key = jax.random.PRNGKey(seed)
    params = {}
    for head, levels in (("first", cfg["budgets"]["first"]), ("second", cfg["budgets"]["second"])):
        key, a, b = jax.random.split(key, 3)
        hidden = cfg["training"]["hidden"]
        params[head] = dict(
            w1=jax.random.normal(a, (len(FEATURES), hidden)) * 0.1,
            b1=jnp.zeros(hidden),
            w2=jax.random.normal(b, (hidden, len(levels))) * 0.01,
            b2=jnp.zeros(len(levels)),
        )
    return params


def logits(params, state):
    return jnp.tanh(jnp.asarray(state) @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]


def probabilities(params, state, legal):
    return jax.nn.softmax(jnp.where(jnp.asarray(legal), logits(params, state), -1e9))


def draw(params, state, legal, rng, training):
    if not np.any(legal):
        raise ValueError("Empty legal budget set")
    values = np.asarray(logits(params, state))
    noise = (
        -np.log(-np.log(rng.uniform(1e-6, 1 - 1e-6, len(values))))
        if training
        else np.zeros_like(values)
    )
    action = int(np.argmax(np.where(legal, values + noise, -np.inf)))
    return action, noise.astype(np.float32)


def st_weights(params, state, legal, noise, action, temperature):
    soft = jax.nn.softmax(
        (jnp.where(jnp.asarray(legal), logits(params, state), -1e9) + noise) / temperature
    )
    hard = jax.nn.one_hot(action, len(legal))
    # Do not implement ST with cancelling floating additions: XLA may reassociate
    # them under AD. The physical forward must remain exactly one-hot.
    return hard_forward_soft_jvp(hard, soft)


@jax.custom_jvp
def hard_forward_soft_jvp(hard, soft):
    return hard


@hard_forward_soft_jvp.defjvp
def hard_soft_jvp(primals, tangents):
    hard, _ = primals
    _, soft_tangent = tangents
    return hard, soft_tangent


def rl_loss(params, contexts, advantage, score_scale=1.0):
    terms = []
    for c in contexts:
        if c["cold"]:
            continue
        for head, index in (("first", 0), ("second", 1)):
            p = probabilities(params[head], c[f"state{index}"], c[f"legal{index}"])
            terms.append(jnp.log(jnp.maximum(p[c[f"action{index}"]], 1e-20)))
    return -jax.lax.stop_gradient(jnp.asarray(advantage)) * jnp.sum(jnp.stack(terms))/score_scale


def adam_state(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return dict(m=z, v=z, step=0)


def adam(params, gradients, state, cfg):
    arrays = jax.tree_util.tree_leaves(gradients)
    if not all(np.isfinite(np.asarray(g)).all() for g in arrays):
        raise FloatingPointError("Nonfinite budget-policy gradient")
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in arrays))
    if not np.isfinite(float(norm)):
        raise FloatingPointError("Nonfinite aggregate gradient norm; reject optimizer update")
    scale = jnp.minimum(1.0, cfg["training"]["gradient_clip"] / (norm + 1e-12))
    gradients = jax.tree_util.tree_map(lambda g: g * scale, gradients)
    step = state["step"] + 1
    m = jax.tree_util.tree_map(lambda x, g: 0.9 * x + 0.1 * g, state["m"], gradients)
    v = jax.tree_util.tree_map(lambda x, g: 0.999 * x + 0.001 * g * g, state["v"], gradients)
    updated = jax.tree_util.tree_map(
        lambda p, a, b: (
            p
            - cfg["training"]["learning_rate"]
            * (a / (1 - 0.9**step))
            / (jnp.sqrt(b / (1 - 0.999**step)) + 1e-8)
        ),
        params,
        m,
        v,
    )
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(updated)):
        raise FloatingPointError("Nonfinite updated policy")
    return updated, dict(m=m, v=v, step=step), float(norm)


def save(file, params, optimizer, baseline):
    arrays = {"step": np.array(optimizer["step"]), "baseline": np.array(baseline)}
    for prefix, tree in (("params", params), ("m", optimizer["m"]), ("v", optimizer["v"])):
        for head, values in tree.items():
            for name, value in values.items():
                arrays[f"{prefix}.{head}.{name}"] = np.asarray(value)
    atomic_npz(file, **arrays)


def restore(file):
    with np.load(file, allow_pickle=False) as arrays:
        trees = {}
        for prefix in ("params", "m", "v"):
            trees[prefix] = {
                head: {
                    name: jnp.asarray(arrays[f"{prefix}.{head}.{name}"])
                    for name in ("w1", "b1", "w2", "b2")
                }
                for head in ("first", "second")
            }
        return (
            trees["params"],
            dict(m=trees["m"], v=trees["v"], step=int(arrays["step"])),
            float(arrays["baseline"]),
        )
