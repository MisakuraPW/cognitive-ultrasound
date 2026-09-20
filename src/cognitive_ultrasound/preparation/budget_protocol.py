"""Causal, exactly funded budget baselines and train-only risk fitting."""

from collections import defaultdict
from functools import lru_cache

import numpy as np

from .analysis import fit_ridge, predict
from .common import FEATURES


@lru_cache(maxsize=2048)
def reachable(frames, lines, budgets=(7, 14, 28)):
    if frames == 0:
        return lines == 0
    if lines < frames * min(budgets) or lines > frames * max(budgets):
        return False
    return any(reachable(frames - 1, lines - k, budgets) for k in budgets)


def feasible(frames_left, remaining, budgets=(7, 14, 28)):
    """Only allow actions leaving an exactly realizable remaining allocation."""
    result = [k for k in budgets if reachable(frames_left - 1, remaining - k, tuple(budgets))]
    if not result:
        raise ValueError("Infeasible remaining budget")
    return result


def funded_action(desired, frames_left, remaining, budgets=(7, 14, 28)):
    return min(feasible(frames_left, remaining, budgets), key=lambda k: (abs(k - desired), k))


def observable(row):
    x = np.array([row[k] for k in FEATURES], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite observable input")
    return x


def design(x, budget, kind):
    x = np.atleast_2d(x)
    b = np.broadcast_to(np.asarray(budget, dtype=float), (len(x),))[:, None] / 112
    z = x[:, :1] if kind == "uncertainty" else x
    return np.concatenate([z, b, b * b, z * b], axis=1)


def estimate(fitted, x, budget):
    return np.clip(predict(fitted["model"], design(x, budget, fitted["kind"])), 0, 2)


def patient_error(samples, estimates):
    groups = defaultdict(list)
    for row, value in zip(samples, estimates):
        groups[row["case"]].append(abs(row["y"] - value))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def fit_models(train, development, ridge=1.0):
    if not train or not development:
        raise ValueError("Nonempty train/development samples required")
    if {r["case"] for r in train} & {r["case"] for r in development}:
        raise ValueError("Patient leakage")
    x = np.array([r["x"] for r in train])
    budgets = np.array([r["budget"] for r in train])
    if set(budgets) != {7, 14, 28} or set(np.round(x[:, -1] * 112).astype(int)) != {7, 14, 28}:
        raise ValueError("Both previous-budget and candidate-budget coverage required")
    fitted = {}
    for kind in ("uncertainty", "all"):
        model = fit_ridge(design(x, budgets, kind), [r["y"] for r in train], ridge)
        entry = dict(kind=kind, model=model)
        train_risk = estimate(entry, x, 14)
        entry["thresholds"] = np.quantile(train_risk, [1 / 3, 2 / 3]).tolist()
        dx = np.array([r["x"] for r in development])
        predictions = estimate(entry, dx, [r["budget"] for r in development])
        entry["development_patient_mae"] = patient_error(development, predictions)
        fitted[kind] = entry
    return dict(
        models=fitted,
        uncertainty_thresholds=np.quantile(x[:, 0], [1 / 3, 2 / 3]).tolist(),
        train_min=x.min(0).tolist(),
        train_max=x.max(0).tolist(),
        features=FEATURES,
        candidate_budgets=[7, 14, 28],
        target="Next-frame all-pixel MAE given candidate budget and previous observables",
        note="Train-only scaling/coefficients/thresholds. No confirmation tuning.",
    )


def proposal(policy, previous, bundle, rng, available):
    """No target/quality labels, remaining future images, or future schedules are accepted."""
    if previous is None or policy == "fixed":
        return 14, []
    if policy == "random":
        return int(rng.choice(available)), []
    x = observable(previous)
    bad = [
        k
        for k, v, lo, hi in zip(FEATURES, x, bundle["train_min"], bundle["train_max"])
        if not lo <= v <= hi
    ]
    if bad:
        return 14, bad
    if policy == "current_uncertainty":
        value, cut = x[0], bundle["uncertainty_thresholds"]
    else:
        fitted = bundle["models"][
            {"simple_forecast": "uncertainty", "future_forecast": "all"}[policy]
        ]
        value, cut = float(estimate(fitted, x, 14)[0]), fitted["thresholds"]
    return [7, 14, 28][int(np.searchsorted(cut, value))], []
