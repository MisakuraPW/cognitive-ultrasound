"""Causal acquisition primitives; no image/EF truth enters a policy state."""

import numpy as np

FEATURES = (
    "variance_mean",
    "variance_q90",
    "variance_max",
    "image_change",
    "observed_residual",
    "acquired_fraction",
    "task_mean",
    "task_std",
    "task_score_mean",
    "task_score_max",
    "history_fraction",
    "last_budget_fraction",
)


def state_features(
    particles,
    previous,
    scores,
    task_values,
    observed=None,
    mask=None,
    history_length=0,
    last_budget=0,
    clip_span=63,
    feature_scales=None,
):
    variance = np.var(particles, axis=0)
    image = np.asarray(particles)[0]
    measured = np.zeros_like(image, dtype=bool) if mask is None else np.asarray(mask, bool)
    residual = 0.0 if not measured.any() else np.abs(image - observed)[measured].mean()
    values = [
        variance.mean(),
        np.quantile(variance, 0.9),
        variance.max(),
        np.abs(image - previous).mean(),
        residual,
        measured[0].mean(),
        np.mean(task_values) / 100,
        np.std(task_values) / 100,
        np.mean(scores),
        np.max(scores),
        min(history_length / clip_span, 1),
        last_budget / 112,
    ]
    result = np.asarray(values, np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite observable state")
    # Fixed transform, not normalization fitted with future/confirmation cases.
    transformed = np.sign(result) * np.log1p(np.abs(result))
    if feature_scales is not None:
        scale = np.asarray(feature_scales, np.float32)
        if scale.shape != transformed.shape or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Feature scales must be12 positive finite training-only values")
        # Compress unusually large values, while lifting small task-sensitivity inputs to usable units.
        value = transformed.astype(np.float64) / scale.astype(np.float64)
        transformed = (np.sign(value) * np.log1p(np.abs(value))).astype(np.float32)
        if not np.isfinite(transformed).all():raise FloatingPointError("Nonfinite normalized state")
    return transformed


def task_scores(particles, gradients):
    """Official TBIG-code convention: variance * square(mean input gradient)."""
    particles, gradients = np.asarray(particles), np.asarray(gradients)
    if particles.shape != gradients.shape or particles.ndim != 3 or len(particles) < 2:
        raise ValueError("Expected corresponding [particles, depth, angle] arrays")
    score = (np.var(particles, axis=0) * np.mean(gradients, axis=0) ** 2).sum(axis=0)
    if not np.isfinite(score).all():
        raise FloatingPointError("Nonfinite task saliency")
    return score.astype(np.float32)


def greedy_order(scores, maximum, excluded=(), radius=2):
    """TBIG-style Gaussian redundancy suppression, explicit unique-line exclusion."""
    scores = np.array(scores, dtype=np.float64, copy=True)
    available = np.ones(len(scores), bool)
    available[np.asarray(excluded, dtype=int)] = False
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid saliency")
    if not 0 <= maximum <= available.sum() or radius < 1:
        raise ValueError("Invalid selector capacity")
    order = []
    # Zero-saliency ties use the same deterministic evenly spread ranking in every arm.
    fallback = np.linspace(0, len(scores) - 1, len(scores), dtype=int)
    # Farthest-point tie ordering keeps the first-frame mask spatially spread.
    anchors = []
    for _ in range(maximum):
        eligible = np.where(available, scores, -np.inf)
        if np.max(eligible) == 0:
            distance = (
                np.ones(len(scores)) * len(scores)
                if not anchors
                else np.min(np.abs(fallback[:, None] - np.array(anchors)[None]), axis=1)
            )
            index = int(np.argmax(np.where(available, distance, -1)))
        else:
            index = int(np.argmax(eligible))
        order.append(index)
        anchors.append(index)
        available[index] = False
        for offset in range(-radius, radius + 1):
            column = index + offset
            if 0 <= column < len(scores):
                scores[column] *= 1 - np.exp(-0.5 * (2 * offset / radius) ** 2)
    return np.asarray(order, np.int32)


def mask_bank(order, budgets, width=112):
    bank = np.zeros((len(budgets), width), np.float32)
    for i, k in enumerate(budgets):
        if not 0 <= k <= len(order):
            raise ValueError("Budget exceeds ordered unique lines")
        bank[i, order[:k]] = 1
    return bank


def causal_window(history, current, frames=32, period=2):
    """Only past reconstructions plus current posterior hypothesis; early left padding."""
    available = [*history, current]
    indices = np.maximum(0, len(available) - 1 - period * np.arange(frames - 1, -1, -1))
    return np.stack([available[int(i)] for i in indices])


def clip_indices(length, frames=32, period=2, stride=16):
    if length < 1 or min(frames, period, stride) < 1:
        raise ValueError("Invalid EF temporal protocol")
    span = (frames - 1) * period + 1
    if length < span:
        # Repeat the last acquired frame, never a future frame.
        return np.minimum(np.arange(frames) * period, length - 1)[None]
    starts = list(range(0, length - span + 1, stride))
    if starts[-1] != length - span:
        starts.append(length - span)
    return np.asarray(starts)[:, None] + np.arange(frames)[None] * period


class Observation:
    """Simulator boundary: policy/selector receive acquired values only."""

    def __init__(self, frame):
        self._frame = np.array(frame, np.float32, copy=True)
        self.mask = np.zeros_like(frame, np.float32)
        self.values = np.zeros_like(frame, np.float32)

    def acquire(self, lines):
        lines = np.asarray(lines, int)
        if (
            len(lines) != len(set(lines.tolist()))
            or (lines < 0).any()
            or (lines >= self.mask.shape[1]).any()
            or self.mask[0, lines].any()
        ):
            raise ValueError("Repeated or invalid acquisition")
        self.mask[:, lines] = 1
        self.values[:, lines] = self._frame[:, lines]
        return self.values.copy(), self.mask.copy()
