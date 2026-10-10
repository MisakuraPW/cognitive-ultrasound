"""Explicit CPU functional fixture ONLY. Outputs cannot establish scientific results."""

import numpy as np

from .protocol import clip_indices, task_scores


class SyntheticPerception:
    def infer(self, h, m, p, key, cold=False):
        rng = np.random.default_rng(int(np.asarray(key, dtype=np.uint64).sum()))
        old = np.zeros_like(h) if p is None else np.mean(p, axis=0)
        estimate = (
            0.45 * np.asarray(h)
            + 0.35 * old
            + rng.normal(0, 0.01, np.asarray(h).shape).astype(np.float32)
        )
        return np.stack([estimate - 0.04, estimate + 0.04]).astype(np.float32)


class SyntheticTask:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = 0
        self.seconds = 0
        self.closed = False

    def score(self, p, h):
        gradients = np.ones_like(p) / p[0].size
        score = task_scores(p, gradients)
        self.last_score_diagnostics = dict(
            variance_lines=np.var(p, axis=0).sum(0),
            sensitivity_lines=(np.mean(gradients, axis=0) ** 2).sum(0),
            combined_lines=score,
            gradient_cancellation_ratio=1.0,
            mean_squared_gradient_sum=float(np.mean(gradients**2, axis=0).sum()),
        )
        return score, np.array([50 + 20 * x.mean() for x in p])

    def request(self, clips, gradient=False, domain="polar"):
        self.calls += 1
        return dict(predictions=np.array([50 + 20 * x.mean() for x in clips]))

    def video(self, images, gradient=False, domain="polar", stride=None):
        ids = clip_indices(len(images), 32, 2, 16 if stride is None else stride)
        values = self.request(np.stack([images[x] for x in ids]))["predictions"]
        return float(values.mean()), np.zeros_like(images), values.tolist()

    def close(self):
        self.closed = True
