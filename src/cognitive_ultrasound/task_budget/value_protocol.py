"""Pure, bounded protocols for budget value diagnostics (no model imports)."""

from collections import Counter
from itertools import combinations

import numpy as np

LEVELS = {"L": (7, 0), "M": (7, 7), "H": (7, 14)}
READOUTS = {"legacy_v4": 16, "all_starts_v1": 1}


def blocks(length):
    q = (length - 1) // 4
    if q < 1:
        raise ValueError("Four nonempty warm-frame blocks required")
    return [(1 + i * q, 1 + (i + 1) * q) for i in range(4)], (1 + 4 * q, length)


def anchors(length):
    return [(a + b - 1) // 2 for a, b in blocks(length)[0]]


def uniform(length, pair):
    result = np.tile(np.array(pair, np.int32), (length, 1))
    result[0] = (10, 4)
    return result


def mixed(length, high):
    if len(set(high)) != len(high) or any(x not in range(4) for x in high):
        raise ValueError("Distinct block indexes0..3 required")
    result = uniform(length, LEVELS["M"])
    for i, (a, b) in enumerate(blocks(length)[0]):
        result[a:b] = LEVELS["H"] if i in high else LEVELS["L"]
    return result


def candidates(length):
    return [("same", list(c), mixed(length, c)) for c in combinations(range(4), 2)] + [
        ("saving", [i], mixed(length, [i])) for i in range(4)
    ]


def periodic(length, count):
    if count not in (1, 2):
        raise ValueError("Only locked one/two-high-block costs supported")
    q = (length - 1) // 4
    result = uniform(length, LEVELS["M"])
    n, high = 4 * q, q * count
    for i in range(n):
        result[i + 1] = LEVELS["H"] if ((i + 1) * high // n > i * high // n) else LEVELS["L"]
    assert Counter(map(tuple, result[1:])) == Counter(map(tuple, mixed(length, range(count))[1:]))
    return result


def reverse(schedule):
    result = np.array(schedule, np.int32, copy=True)
    tail = blocks(len(result))[1][0]
    result[1:tail] = result[1:tail][::-1]
    return result


def hybrid(low, medium, high, high_blocks):
    if low.shape != medium.shape or low.shape != high.shape:
        raise ValueError("Cached videos must align exactly")
    result = medium.copy()
    for i, (a, b) in enumerate(blocks(len(low))[0]):
        result[a:b] = high[a:b] if i in high_blocks else low[a:b]
    return result


def coverage(indices, length):
    indexes = np.asarray(indices)
    if indexes.ndim != 2 or indexes.size == 0 or indexes.min() < 0 or indexes.max() >= length:
        raise ValueError("Invalid EF indexes")
    counts = np.bincount(indexes.ravel(), minlength=length)
    return dict(
        total_frames=length,
        direct_frames=int((counts > 0).sum()),
        direct_fraction=float((counts > 0).mean()),
        appearances=counts.tolist(),
        complete=bool((counts > 0).all()),
    )


def quality(target, images):
    target, images = np.asarray(target), np.asarray(images)
    if target.shape != images.shape or len(images) < 1 or not np.isfinite(images).all():
        raise ValueError("Invalid/nonfinite aligned reconstruction")
    error = images.astype(np.float64) - target
    frame_mse = np.mean(error**2, axis=(1, 2))
    threshold = float(np.quantile(frame_mse, 0.9))
    streak = longest = 0
    for value in frame_mse:
        streak = streak + 1 if value > threshold else 0
        longest = max(longest, streak)
    return dict(
        mse=float(frame_mse.mean()),
        mae=float(np.abs(error).mean()),
        frame_mse=frame_mse.tolist(),
        frame_mse_p90=threshold,
        longest_above_own_p90=longest,
        temporal_mse=float(np.mean(np.diff(error, axis=0) ** 2)) if len(images) > 1 else None,
        clipped_fraction=float(((images < -1) | (images > 1)).mean()),
        output_min=float(images.min()),
        output_max=float(images.max()),
    )


def common_unobserved(target, images, masks):
    if not (len(images) == len(masks)):
        raise ValueError("Aligned variants required")
    unknown = ~np.any(np.stack(masks).astype(bool), axis=0)
    return dict(
        pixels=int(unknown.sum()),
        fraction=float(unknown.mean()),
        mse=[
            float(np.mean((x[unknown] - target[unknown]) ** 2)) if unknown.any() else None
            for x in images
        ],
    )


def value_labels(target, base, branch, at, base_eval, branch_eval):
    a, b = quality(target, base), quality(target, branch)
    horizons = {}
    for width in [1, 8, 32, len(base) - at]:
        stop = min(len(base), at + width)
        horizons[str(width)] = float(
            np.mean(np.array(a["frame_mse"])[at:stop] - np.array(b["frame_mse"])[at:stop])
        )
    return dict(
        mse_gain=a["mse"] - b["mse"],
        horizon_mse_gain=horizons,
        temporal_gain=(a["temporal_mse"] - b["temporal_mse"])
        if a["temporal_mse"] is not None
        else None,
        ef_gain=base_eval["all_starts_v1"]["absolute_error"]
        - branch_eval["all_starts_v1"]["absolute_error"],
        preservation_gain=base_eval["all_starts_v1"]["preservation_error"]
        - branch_eval["all_starts_v1"]["preservation_error"],
    )


def bootstrap(values, one_sided=False, seed=20261010):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite independent video values required")
    indexes = np.random.default_rng(seed).integers(len(values), size=(5000, len(values)))
    sample = values[indexes].mean(1)
    return dict(
        mean=float(values.mean()),
        ci95=np.quantile(sample, [0.025, 0.975]).tolist(),
        upper95=float(np.quantile(sample, 0.95)),
        replicates=5000,
        unit="video",
        seed=seed,
    )


def rank_correlation(x, y):
    def rank(v):
        _, inverse, count = np.unique(v, return_inverse=True, return_counts=True)
        return (np.cumsum(count) - count + (count - 1) / 2)[inverse]

    x, y = rank(np.asarray(x)), rank(np.asarray(y))
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None


def predictability(rows):
    """LOVO ridge with fold-only scaling, fixed alpha; never a budget controller."""
    result = []
    for stage in ["first", "second"]:
        group = [x for x in rows if x["stage"] == stage]
        if len(set(x["case"] for x in group)) < 2:
            result.append(dict(stage=stage, status="insufficient_videos"))
            continue
        x = np.array([r["features"] for r in group], float)
        for label in ["mse_gain", "ef_gain"]:
            y = np.array([r["gain"][label] for r in group], float)
            prediction = np.zeros(len(y))
            constant = np.zeros(len(y))
            folds = []
            for case in sorted(set(r["case"] for r in group)):
                test = np.array([r["case"] == case for r in group])
                train = ~test
                mean = x[train].mean(0)
                std = np.maximum(x[train].std(0), 1e-8)
                a = np.c_[np.ones(train.sum()), (x[train] - mean) / std]
                b = np.c_[np.ones(test.sum()), (x[test] - mean) / std]
                penalty = np.eye(a.shape[1])
                penalty[0, 0] = 0
                coef = np.linalg.solve(a.T @ a + penalty, a.T @ y[train])
                prediction[test] = b @ coef
                constant[test] = y[train].mean()
                folds.append(
                    dict(
                        held_out=case,
                        training_cases=sorted(set(r["case"] for r, t in zip(group, train) if t)),
                    )
                )
            result.append(
                dict(
                    stage=stage,
                    label=label,
                    observations=len(y),
                    videos=len(folds),
                    rmse=float(np.sqrt(np.mean((prediction - y) ** 2))),
                    constant_rmse=float(np.sqrt(np.mean((constant - y) ** 2))),
                    rank_correlation=rank_correlation(prediction, y),
                    positive_accuracy=float(((prediction > 0) == (y > 0)).mean()),
                    predictions=prediction.tolist(),
                    truth=y.tolist(),
                    folds=folds,
                    alpha=1,
                    scope="observable-state screening; not online efficacy or convergence",
                )
            )
    return result
