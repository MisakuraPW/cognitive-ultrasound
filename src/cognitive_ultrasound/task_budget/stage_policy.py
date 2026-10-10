"""One causal GRU budget head per experiment; frozen CASL is never differentiated."""

import copy

import jax
import jax.numpy as jnp
import numpy as np

from .stage_protocol import ACTIVE_STAGE, FIXED, quota_mask
from .value_storage import tree_digest

INPUTS = 18


def initialize(seed, cfg, arm):
    h = cfg["stage_budget"]["hidden"]
    levels = cfg["budgets"]["first" if ACTIVE_STAGE[arm] == 0 else "second"]
    a, b, c = jax.random.split(jax.random.PRNGKey(seed), 3)
    bias = np.full(len(levels), -0.5, np.float32)
    bias[levels.index(7)] = 0.5
    return dict(
        wx=jax.random.normal(a, (INPUTS, 3 * h)) * 0.1,
        wh=jax.random.normal(b, (h, 3 * h)) * 0.1,
        b=jnp.zeros(3 * h),
        wo=jax.random.normal(c, (h, len(levels))) * 0.01,
        bo=jnp.asarray(bias),
    )


def logits(params, sequence, valid):
    h = params["wh"].shape[0]

    def step(hidden, item):
        x, keep = item
        xx = x @ params["wx"] + params["b"]
        hh = hidden @ params["wh"]
        z = jax.nn.sigmoid(xx[:h] + hh[:h])
        r = jax.nn.sigmoid(xx[h : 2 * h] + hh[h : 2 * h])
        candidate = jnp.tanh(xx[2 * h :] + (r * hidden) @ params["wh"][:, 2 * h :])
        new = z * hidden + (1 - z) * candidate
        return jnp.where(keep, new, hidden), None

    final, _ = jax.lax.scan(step, jnp.zeros(h), (jnp.asarray(sequence), jnp.asarray(valid)))
    return final @ params["wo"] + params["bo"]


def probabilities(params, sequence, valid, legal):
    return jax.nn.softmax(jnp.where(jnp.asarray(legal), logits(params, sequence, valid), -1e9))


probabilities_jit = jax.jit(probabilities)


def reinforce(params, sequence, valid, legal, actions, advantage, score_scale):
    p = jax.vmap(probabilities, in_axes=(None, 0, 0, 0))(params, sequence, valid, legal)
    chosen = jnp.take_along_axis(p, jnp.asarray(actions)[:, None], axis=1)[:, 0]
    # A CONSTANT scale preserves equal-video objective weights for variable lengths.
    return (
        -jax.lax.stop_gradient(advantage) * jnp.log(jnp.maximum(chosen, 1e-20)).sum() / score_scale
    )


reinforce_gradient = jax.jit(jax.grad(reinforce))


class Controller:
    def __init__(self, params, cfg, arm, frames, greedy=False):
        self.params, self.cfg, self.arm = params, cfg, arm
        self.stage = ACTIVE_STAGE[arm]
        self.frames, self.greedy = frames, greedy
        self.window = cfg["stage_budget"]["history_frames"]
        self.levels = cfg["budgets"]["first" if self.stage == 0 else "second"]
        self.quota = arm != "Q2"
        self.remaining = 7 * (frames - 1) if self.quota else None
        self.history, self.trace = [], []
        self.identity = dict(
            arm=arm,
            parameters=tree_digest(params),
            frames=frames,
            mode="argmax_auxiliary" if greedy else "sampled_main",
            protocol="causal_gru_exact_quota_v1",
            window=self.window,
        )

    def select(
        self, stage, index, state, legal, rng, *, particles, prior, past, observed, mask, k1
    ):
        levels = self.cfg["budgets"]["first" if stage == 0 else "second"]
        if stage != self.stage:
            return levels.index(FIXED[stage])
        left = self.frames - index
        base_legal = np.array(legal, bool)
        constrained = (
            base_legal & quota_mask(self.levels, self.remaining, left) if self.quota else base_legal
        )
        if not constrained.any():
            raise ValueError("No physically valid stage action")
        if stage == 0:
            motion = float(np.abs(past[-1] - past[-2]).mean()) if len(past) > 1 else 0.0
            extras = [motion, 0.0, 0.0, 0.0]
        else:
            measured = np.asarray(mask, bool)
            innovation = np.abs(np.mean(prior, axis=0) - observed)[measured]
            extras = [
                float(innovation.mean()),
                float(np.quantile(innovation, 0.9)),
                float(np.abs(np.mean(particles, 0) - np.mean(prior, 0)).mean()),
                k1 / 112,
            ]
        extras.extend(
            [self.remaining / (7 * left) if self.quota else 0.0, left / (self.frames - 1)]
        )
        feature = np.r_[state, np.sign(extras) * np.log1p(np.abs(extras))].astype(np.float32)
        if feature.shape != (INPUTS,) or not np.isfinite(feature).all():
            raise FloatingPointError("Invalid causal policy features")
        self.history = [*self.history, feature][-self.window :]
        sequence = np.zeros((self.window, INPUTS), np.float32)
        valid = np.zeros(self.window, bool)
        sequence[-len(self.history) :] = self.history
        valid[-len(self.history) :] = True
        p = np.asarray(probabilities_jit(self.params, sequence, valid, constrained))
        if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
            raise FloatingPointError("Invalid policy distribution")
        action = int(np.argmax(p)) if self.greedy else int(rng.choice(len(p), p=p / p.sum()))
        self.trace.append(
            dict(
                sequence=sequence,
                valid=valid,
                legal=constrained,
                action=action,
                frame=index,
                max_probability=float(p.max()),
                entropy=float(-(p * np.log(p + 1e-20)).sum()),
                quota_constrained=bool(np.any(base_legal != constrained)),
                quota_forced=bool(constrained.sum() == 1 and base_legal.sum() > 1),
            )
        )
        if self.quota:
            self.remaining -= self.levels[action]
            if index == self.frames - 1 and self.remaining != 0:
                raise AssertionError("Whole-video acquisition quota was not exhausted")
        return action

    def snapshot(self):
        return copy.deepcopy(
            dict(
                identity=self.identity,
                remaining=self.remaining,
                history=self.history,
                trace=self.trace,
            )
        )

    def restore(self, value):
        if value["identity"] != self.identity:
            raise ValueError("Policy/state identity mismatch")
        self.remaining = value["remaining"]
        self.history, self.trace = copy.deepcopy(value["history"]), copy.deepcopy(value["trace"])

    def gradients(self, advantage):
        if not self.trace:
            raise ValueError("No warm-frame learning decisions")
        n = len(self.trace)
        padded = ((n + 63) // 64) * 64
        sequence = np.zeros((padded, self.window, INPUTS), np.float32)
        valid = np.zeros((padded, self.window), bool)
        legal = np.zeros((padded, len(self.levels)), bool)
        legal[:, 0] = True  # Padding has exactly probability1 and zero policy gradient.
        actions = np.zeros(padded, np.int32)
        sequence[:n] = np.stack([x["sequence"] for x in self.trace])
        valid[:n] = np.stack([x["valid"] for x in self.trace])
        legal[:n] = np.stack([x["legal"] for x in self.trace])
        actions[:n] = [x["action"] for x in self.trace]
        return reinforce_gradient(
            self.params,
            sequence,
            valid,
            legal,
            actions,
            jnp.asarray(advantage),
            float(self.cfg["stage_budget"]["rl_score_scale"]),
        )
