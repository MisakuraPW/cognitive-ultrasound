"""Frozen asymmetric research design, exact quotas and a shared pilot curriculum."""

from functools import lru_cache

import numpy as np

ARMS = ("T1", "Q2", "T2")
ACTIVE_STAGE = {"T1": 0, "Q2": 1, "T2": 1}
FIXED = (7, 7)
COLD = (10, 4)


@lru_cache(maxsize=128)
def reachable(levels, count):
    """Bit s is set iff exactly count future decisions can consume s lines."""
    if count < 0 or not levels or min(levels) < 0:
        raise ValueError("Invalid acquisition quota")
    bits, result = 1, [1]
    for _ in range(count):
        value = 0
        for k in levels:
            value |= bits << k
        bits = value
        result.append(bits)
    return tuple(result)


def quota_mask(levels, remaining, decisions_left):
    if decisions_left < 1 or remaining < 0:
        raise ValueError("No quota decision remains")
    future = reachable(tuple(levels), decisions_left - 1)[-1]
    result = np.array([remaining >= k and bool(future & (1 << (remaining - k))) for k in levels])
    if not result.any():
        raise ValueError("Exact quota became infeasible")
    return result


def curriculum(names, cfg):
    """Pilot is the first16 updates, not a separate model to discard/retrain."""
    rng = np.random.default_rng(cfg["stage_budget"]["training_seed"])
    order = rng.permutation(names).tolist()
    p = cfg["stage_budget"]["pilot_train"]
    epochs = cfg["stage_budget"]["pilot_passes"]
    stream = []
    visits = {n: 0 for n in names}

    def append(items, section):
        for name in items:
            stream.append(dict(case=name, visit=visits[name], section=section))
            visits[name] += 1

    for e in range(epochs):
        append(order[:p] if e == 0 else rng.permutation(order[:p]).tolist(), "pilot")
    append(order[p:], "full_pass_1")
    ends = {1: len(stream)}
    for e in range(2, cfg["stage_budget"]["full_passes"] + 1):
        append(rng.permutation(names).tolist(), f"full_pass_{e}")
        ends[e] = len(stream)
    return stream, ends


def training_objective(arm, metrics, reference, dual, cfg):
    """Q2 minimizes acquisition under declared mean/p90 reconstruction constraints."""
    d = cfg["stage_budget"]
    if arm in ("T1", "T2"):
        # A full-input error reference is action-independent, never a policy input.
        return metrics["ef_error"] - reference["full_ef_error"], None
    ratios = np.array(
        [
            (metrics["mse"] - reference["mse"]) / max(reference["mse"], d["mse_floor"]),
            (metrics["p90"] - reference["p90"]) / max(reference["p90"], d["mse_floor"]),
        ]
    )
    violation = ratios - np.array([d["mse_margin"], d["p90_margin"]])
    objective = metrics["warm_mean_lines"] / 14 + float(np.asarray(dual) @ violation)
    if not np.isfinite(objective) or not np.isfinite(violation).all():
        raise FloatingPointError("Nonfinite reconstruction objective")
    return float(objective), violation


def dual_update(dual, violation, cfg):
    d = cfg["stage_budget"]
    return np.clip(np.asarray(dual) + d["dual_lr"] * violation, 0, d["dual_max"])


def checkpoint_choice(arm, rows, cfg):
    """Development-only selection; CI is descriptive, never an exploration gate."""
    if not rows or any(r["cohort"] != "development" for r in rows):
        raise ValueError("Checkpoint choice may only consume development evidence")
    grouped = {}
    for r in rows:
        grouped.setdefault(r["update"], []).append(r)
    candidates = []
    for update, values in grouped.items():
        x = dict(
            update=update,
            ef=float(np.mean([v["ef_error"] for v in values])),
            lines=float(np.mean([v["mean_lines"] for v in values])),
            mse_ratio=float(np.mean([v["mse_ratio"] for v in values])),
            p90_ratio=float(np.mean([v["p90_ratio"] for v in values])),
        )
        d = cfg["stage_budget"]
        x["feasible_point_estimate"] = (
            x["mse_ratio"] <= d["mse_margin"] and x["p90_ratio"] <= d["p90_margin"]
        )
        candidates.append(x)
    if arm == "Q2":
        feasible = [x for x in candidates if x["feasible_point_estimate"]]
        chosen = (
            min(feasible, key=lambda x: (x["lines"], x["mse_ratio"], x["update"]))
            if feasible
            else min(
                candidates,
                key=lambda x: (
                    max(x["mse_ratio"] - cfg["stage_budget"]["mse_margin"], 0)
                    + max(x["p90_ratio"] - cfg["stage_budget"]["p90_margin"], 0),
                    x["lines"],
                ),
            )
        )
    else:
        chosen = min(candidates, key=lambda x: (x["ef"], x["update"]))
    return dict(
        chosen=chosen,
        candidates=candidates,
        selection_split="development",
        confidence_intervals_are_not_a_gate=True,
    )
