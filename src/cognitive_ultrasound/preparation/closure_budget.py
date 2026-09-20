"""Finite GPU data collection and fixed-total-budget controls for preparation closure."""

import json
from pathlib import Path

import numpy as np

from ..experiments import case_seed
from .analysis import paired_comparison
from .budget_protocol import (
    estimate,
    feasible,
    fit_models,
    funded_action,
    observable,
    patient_error,
    proposal,
)
from .common import atomic_json, atomic_npz, emit, read_frames, read_json, rows_from


def calibrate_selection(cfg, root):
    checks = {}
    for k in cfg["closure"]["budgets"]:
        names = [f"anchor_{k}_{v}" for v in ("reference", "official25_fp16")]
        complete = all(
            (root / "jobs" / n / "result.json").exists()
            and read_json(root / "jobs" / n / "result.json").get("status") == "completed"
            for n in names
        )
        checks[str(k)] = (
            paired_comparison(*[rows_from(root / "jobs" / n) for n in names], cfg["gate"])
            if complete
            else dict(passed=False, reasons=["anchor incomplete"])
        )
    # A failed/missing reference is not an excuse to launch an unverified fallback.
    reference_ready = all(
        (root / "jobs" / f"anchor_{k}_reference" / "result.json").exists()
        and read_json(root / "jobs" / f"anchor_{k}_reference" / "result.json").get("status")
        == "completed"
        for k in cfg["closure"]["budgets"]
    )
    selected = "official25_fp16" if all(c["passed"] for c in checks.values()) else "reference"
    value = dict(
        variant=selected,
        ready=reference_ready,
        checks=checks,
        scope="New hardware and 7/14/28 uniform-line anchors; prior common-history evidence retained in source receipt",
    )
    atomic_json(root / "budget_selection.json", value)
    return value


def collect(task, cfg, manifest, output, root):
    from .casl import Adapter

    cohort = task["cohort"]
    adapter = Adapter(cfg, read_json(root / "budget_selection.json")["variant"], "uniform")
    records = []
    for name in manifest["cohorts"][cohort]:
        for seed in cfg["seeds"]:
            directory = output / str(seed) / Path(name).stem
            receipt = directory / "complete.json"
            if receipt.exists():
                records.append(read_json(receipt))
                continue
            directory.mkdir(parents=True, exist_ok=True)
            frames = read_frames(
                cfg, "train" if cohort == "train" else "val", name, cfg["frames"][cohort]
            )
            rng = np.random.default_rng(case_seed(seed, name))
            adapter.reset(case_seed(seed, name))
            previous = None
            samples = []
            trajectory = []
            # Behavior is specified without looking at errors; each clip covers every budget.
            schedule = np.resize(np.array([7, 14, 28]), len(frames) - 1)
            rng.shuffle(schedule)
            for i in range(len(frames)):
                chosen = 14 if i == 0 else int(schedule[i - 1])
                parent = adapter.clone()
                candidates = [14] if i == 0 else cfg["closure"]["budgets"]
                for k in candidates:
                    adapter.state = adapter.jax.tree_util.tree_map(lambda v: v, parent)
                    adapter.force_lines(np.linspace(0, 111, k, dtype=int))
                    row, arrays = adapter.step(frames[i])
                    if row["actual_lines"] != k:
                        raise ValueError("Wrong actual acquisition count")
                    if previous is not None:
                        samples.append(
                            dict(
                                case=name,
                                seed=seed,
                                frame=i,
                                budget=k,
                                x=observable(previous).tolist(),
                                y=row["mae"],
                                unobserved_mae=row["unobserved_mae"],
                            )
                        )
                    if k == chosen:
                        live_state = adapter.clone()
                        live_row = row
                        if i in (0, len(frames) // 2, len(frames) - 1):
                            atomic_npz(
                                directory / f"frame_{i:04d}.npz",
                                row=np.array(json.dumps(row)),
                                **arrays,
                            )
                adapter.state = live_state
                previous = live_row
                trajectory.append(dict(frame=i, **live_row))
                emit("BUDGET_DATA", cohort=cohort, case=name, seed=seed, frame=i, lines=chosen)
            record = dict(case=name, seed=seed, samples=samples, trajectory=trajectory)
            atomic_json(receipt, record)
            records.append(record)
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            cohort=cohort,
            cases=len(manifest["cohorts"][cohort]),
            records=records,
            protocol="Same causal history/key for counterfactual budgets; independent mixed-budget behavior; truth only in labels",
        ),
    )


def samples(root, cohort):
    result = read_json(root / "jobs" / ("budget_data_" + cohort) / "result.json")
    return [s for r in result["records"] for s in r["samples"]]


def fit(task, cfg, manifest, output, root):
    training, development = samples(root, "train"), samples(root, "development")
    bundle = fit_models(training, development, cfg["risk"]["ridge"])
    atomic_json(output / "model.json", bundle)
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            train_rows=len(training),
            development_rows=len(development),
            scores={k: v["development_patient_mae"] for k, v in bundle["models"].items()},
            caveat="Both fixed probe baselines proceed regardless of gain; no performance-driven expansion",
        ),
    )


def evaluate_probe(task, cfg, manifest, output, root):
    data = samples(root, "confirmation")
    bundle = read_json(root / "jobs/budget_fit/model.json")
    forbidden = set(manifest["cohorts"]["train"] + manifest["cohorts"]["development"])
    if forbidden & {s["case"] for s in data}:
        raise ValueError("Confirmation leakage")
    x = np.array([r["x"] for r in data])
    bs = [r["budget"] for r in data]
    scores = {k: patient_error(data, estimate(v, x, bs)) for k, v in bundle["models"].items()}
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            patient_mae=scores,
            mean_mae={k: float(np.mean(list(v.values()))) for k, v in scores.items()},
            frames=len(data),
            note="Held-out action-conditional forecast check, not a clinical guarantee",
        ),
    )


def closed_loop(task, cfg, manifest, output, root):
    from .casl import Adapter

    adapter = Adapter(cfg, read_json(root / "budget_selection.json")["variant"], "uniform")
    bundle = read_json(root / "jobs/budget_fit/model.json")
    records = []
    cohort = task["cohort"]
    n = cfg["frames"][cohort]
    policies = ["fixed", "random", "current_uncertainty", "simple_forecast", "future_forecast"]
    for name in manifest["cohorts"][cohort]:
        for seed in cfg["seeds"]:
            directory = output / str(seed) / Path(name).stem
            receipt = directory / "complete.json"
            if receipt.exists():
                records.append(read_json(receipt))
                continue
            frames = read_frames(cfg, "val", name, n)
            scores = {}
            rows = {}
            masks = {}
            errors = {}
            for policy in policies:
                adapter.reset(case_seed(seed, name))
                rng = np.random.default_rng(case_seed(seed + 703, name))
                remaining = 14 * n
                previous = None
                rows[policy] = []
                masks[policy] = []
                errors[policy] = []
                for i in range(n):
                    allowed = feasible(n - i, remaining, tuple(cfg["closure"]["budgets"]))
                    desired, bad = proposal(policy, previous, bundle, rng, allowed)
                    k = funded_action(desired, n - i, remaining, tuple(cfg["closure"]["budgets"]))
                    adapter.force_lines(np.linspace(0, 111, k, dtype=int))
                    row, arrays = adapter.step(frames[i])
                    if row["actual_lines"] != k:
                        raise ValueError("Incorrect line accounting")
                    remaining -= k
                    row.update(
                        frame=i,
                        policy=policy,
                        requested_lines=k,
                        desired_lines=desired,
                        budget_constrained=k != desired,
                        ood_features=bad,
                        remaining_lines=remaining,
                    )
                    rows[policy].append(row)
                    previous = row
                    errors[policy].append(np.abs(arrays["prediction"] - arrays["target"]))
                    masks[policy].append(arrays["mask"].astype(bool))
                    if i in (0, n // 2, n - 1):
                        atomic_npz(
                            directory / policy / f"frame_{i:04d}.npz",
                            row=np.array(json.dumps(row)),
                            **arrays,
                        )
                assert remaining == 0
            common = ~np.logical_or.reduce([np.stack(masks[p]) for p in policies])
            for policy in policies:
                rr = rows[policy]
                e = np.stack(errors[policy])
                for i, row in enumerate(rr):
                    row["common_unobserved_mae"] = float(e[i][common[i]].mean())
                    row["common_unobserved_pixels"] = int(common[i].sum())
                scores[policy] = dict(
                    all_pixel_mae=float(e.mean()),
                    common_unobserved_mae=float(
                        np.mean([e[i][common[i]].mean() for i in range(n)])
                    ),
                    unobserved_mae=float(np.mean([r["unobserved_mae"] for r in rr])),
                    low_quality_fraction=float(
                        np.mean([r["mae"] > cfg["closure"]["quality_mae"] for r in rr])
                    ),
                    total_lines=sum(r["actual_lines"] for r in rr),
                    fallback_frames=sum(bool(r["ood_features"]) for r in rr),
                    constrained_frames=sum(r["budget_constrained"] for r in rr),
                    warm_algorithm_s=float(np.median([r["algorithm_s"] for r in rr[2:]])),
                    cold_s=rr[0]["algorithm_s"],
                )
            record = dict(case=name, seed=seed, scores=scores, rows=rows)
            atomic_json(receipt, record)
            records.append(record)
            emit("FUNDED_LOOP", cohort=cohort, case=name, seed=seed, scores=scores)
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            records=records,
            total_budget_per_clip=14 * n,
            quality_threshold=cfg["closure"]["quality_mae"],
            caveat="Small held-out controls; fixed a-priori total, causal decisions, no hindsight-matched budget or retuning; timing includes first policy compile effects",
        ),
    )
