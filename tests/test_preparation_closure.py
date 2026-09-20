"""Meaningful CPU checks of causality, exact resource accounting and finite handoff."""

import copy
import csv
import types

import numpy as np
import pytest

from cognitive_ultrasound.preparation.budget_protocol import (
    feasible,
    fit_models,
    funded_action,
    proposal,
    reachable,
)
from cognitive_ultrasound.preparation.closure import render_report, source_receipt
from cognitive_ultrasound.preparation.common import FEATURES, atomic_json, read_json


def test_funded_actions_finish_exactly_for_arbitrary_causal_proposals():
    rng = np.random.default_rng(42)
    for frames in range(1, 65):
        for _ in range(10):
            remaining = 14 * frames
            for left in range(frames, 0, -1):
                allowed = feasible(left, remaining)
                assert all(reachable(left - 1, remaining - k) for k in allowed)
                k = funded_action(int(rng.choice([7, 14, 28])), left, remaining)
                remaining -= k
            assert remaining == 0
    with pytest.raises(ValueError, match="Infeasible"):
        feasible(1, 21)


def training_samples(case):
    result = []
    for i in range(12):
        x = [0.01 + i * 0.001, 0.02, 0.003, 0.01, 0.1, [7, 14, 28][i % 3] / 112]
        for budget in [7, 14, 28]:
            result.append(
                dict(case=case, frame=i, budget=budget, x=x, y=0.2 + 0.001 * i - 0.002 * budget)
            )
    return result


def test_fitting_uses_train_only_and_demands_budget_coverage():
    t, d = training_samples("train"), training_samples("dev")
    first = fit_models(t, d)
    for r in d:
        r["y"] += 100
    second = fit_models(t, d)
    for kind in first["models"]:
        assert first["models"][kind]["model"] == second["models"][kind]["model"]
        assert first["models"][kind]["thresholds"] == second["models"][kind]["thresholds"]
    with pytest.raises(ValueError, match="leakage"):
        fit_models(t, t)
    for r in t:
        r["x"][-1] = 0.125
    with pytest.raises(ValueError, match="coverage"):
        fit_models(t, d)


def test_decision_ignores_unobserved_labels_and_preserves_ood_guard():
    model = fit_models(training_samples("a"), training_samples("b"))
    row = dict(zip(FEATURES, training_samples("a")[12]["x"]))
    before = proposal("future_forecast", row, model, np.random.default_rng(1), [7, 14, 28])
    row.update(mae=1000, unobserved_mae=-1000, target="secret future truth")
    assert proposal("future_forecast", row, model, np.random.default_rng(1), [7, 14, 28]) == before
    row["budget_fraction"] = 1
    assert proposal("future_forecast", row, model, np.random.default_rng(1), [7, 14, 28]) == (
        14,
        ["budget_fraction"],
    )


class FakeAdapter:
    calls = 0

    def __init__(self, *args):
        self.jax = types.SimpleNamespace(
            tree_util=types.SimpleNamespace(tree_map=lambda f, s: copy.deepcopy(s))
        )
        self.reset(0)

    def reset(self, seed):
        self.state = dict(lines=np.linspace(0, 111, 14, dtype=int), history=0.0)

    def clone(self):
        return copy.deepcopy(self.state)

    def force_lines(self, lines):
        self.state["lines"] = np.asarray(lines)

    def step(self, target):
        from cognitive_ultrasound.preparation.common import metrics, observable_features
        from cognitive_ultrasound.preparation.followup import interpolate_observed

        type(self).calls += 1
        mask = np.zeros_like(target)
        mask[:, self.state["lines"]] = 1
        observation = np.where(mask, target, 0)
        prediction = interpolate_observed(observation, self.state["lines"])
        uncertainty = np.full((112, 112), 0.02, dtype="float32")
        row = dict(
            **metrics(target, prediction, mask),
            **observable_features(
                uncertainty, prediction, np.zeros_like(target), 0.01, len(self.state["lines"])
            ),
            algorithm_s=0.01,
        )
        self.state["history"] += 1
        return row, dict(target=target, prediction=prediction, mask=mask)


def test_cpu_pipeline_resume_accounting_and_report(tmp_path, monkeypatch):
    from cognitive_ultrasound.preparation import casl
    from cognitive_ultrasound.preparation import closure_budget as b

    monkeypatch.setattr(casl, "Adapter", FakeAdapter)

    def images(cfg, split, name, n):
        seed = sum(map(ord, name))
        rng = np.random.default_rng(seed)
        return rng.uniform(-1, 1, (n, 112, 112, 1)).astype("float32")

    monkeypatch.setattr(b, "read_frames", images)
    cfg = dict(
        seeds=[42],
        frames={k: 6 for k in ["train", "development", "confirmation"]},
        closure=dict(budgets=[7, 14, 28], quality_mae=0.15),
        risk=dict(ridge=1.0),
    )
    manifest = dict(cohorts=dict(train=["t1", "t2"], development=["d1"], confirmation=["c1"]))
    atomic_json(tmp_path / "budget_selection.json", dict(variant="official25_fp16", ready=True))
    ledger = {}
    for cohort in cfg["frames"]:
        output = tmp_path / "jobs" / ("budget_data_" + cohort)
        output.mkdir(parents=True)
        b.collect(dict(cohort=cohort), cfg, manifest, output, tmp_path)
        count = FakeAdapter.calls
        b.collect(dict(cohort=cohort), cfg, manifest, output, tmp_path)
        assert FakeAdapter.calls == count
        ledger["budget_data_" + cohort] = dict(status="completed", elapsed_s=1)
    output = tmp_path / "jobs/budget_fit"
    output.mkdir()
    b.fit({}, cfg, manifest, output, tmp_path)
    ledger["budget_fit"] = dict(status="completed", elapsed_s=1)
    output = tmp_path / "jobs/budget_probe_confirm"
    output.mkdir()
    b.evaluate_probe({}, cfg, manifest, output, tmp_path)
    ledger["budget_probe_confirm"] = dict(status="completed", elapsed_s=1)
    for cohort in ["development", "confirmation"]:
        output = tmp_path / "jobs" / ("funded_loop_" + cohort)
        output.mkdir()
        b.closed_loop(dict(cohort=cohort), cfg, manifest, output, tmp_path)
        value = read_json(output / "result.json")
        for record in value["records"]:
            for policy, scores in record["scores"].items():
                assert scores["total_lines"] == 84
                assert record["rows"][policy][0]["actual_lines"] == 14
                assert record["rows"][policy][-1]["remaining_lines"] == 0
        count = FakeAdapter.calls
        b.closed_loop(dict(cohort=cohort), cfg, manifest, output, tmp_path)
        assert FakeAdapter.calls == count
        ledger["funded_loop_" + cohort] = dict(status="completed", elapsed_s=1)
    atomic_json(
        tmp_path / "status.json", dict(status="completed", jobs=ledger, preparation_closed=True)
    )
    render_report(tmp_path)
    assert (tmp_path / "REPORT.md").exists()
    with (tmp_path / "excel_facts.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7 and all(not r["prediction_lock"] and not r["belief_update"] for r in rows)
    summary = read_json(tmp_path / "closure_summary.json")
    assert summary["preparation_closed"] and not summary["automatic_followup"]


def test_report_handles_no_completed_experiments(tmp_path):
    atomic_json(
        tmp_path / "status.json",
        dict(status="failed", jobs={"budget_fit": dict(status="blocked", reason="input missing")}),
    )
    render_report(tmp_path)
    assert "input missing" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


def test_source_receipt_excludes_all_old_validation_but_not_train(tmp_path):
    for i in range(2):
        p = tmp_path / str(i)
        p.mkdir()
        for name in ["identity", "selection"]:
            atomic_json(p / (name + ".json"), {})
        atomic_json(
            p / "manifest.json",
            dict(
                cohorts=dict(
                    train=["t"], debug=[f"d{i}"], development=[f"v{i}"], confirmation=[f"c{i}"]
                )
            ),
        )
    cfg = dict(
        closure=dict(
            previous_runs=[str(tmp_path / "0"), str(tmp_path / "1")], bf_source=str(tmp_path / "0")
        )
    )
    receipt, _ = source_receipt(cfg)
    assert set(receipt["excluded_validation_cases"]) == {"d0", "d1", "v0", "v1", "c0", "c1"}
    assert not receipt["bf_available"]


def test_closure_has_fixed_finite_jobs_not_performance_driven_expansion(tmp_path, monkeypatch):
    from cognitive_ultrasound.config import ROOT, load
    from cognitive_ultrasound.preparation import closure_budget
    from cognitive_ultrasound.preparation.closure import Closure, validate_closure

    cfg = load(ROOT / "configs/preparation_closure.yaml")
    validate_closure(cfg)
    c = Closure(cfg, tmp_path)
    atomic_json(tmp_path / "closure_sources.json", dict(bf_available=True))
    monkeypatch.setattr(
        closure_budget, "calibrate_selection", lambda *a: dict(ready=True, variant="reference")
    )
    called = []

    def job(name, kind, phase, **fields):
        called.append((name, kind, fields))
        c.state["jobs"][name] = dict(
            status="blocked" if fields.get("condition") else "completed", elapsed_s=0, phase=phase
        )

    c.job = job
    c.execute()
    assert len(called) == len({n for n, _, _ in called}) == 18
    assert {n for n, _, _ in called if n.startswith("funded_loop_")} == {
        "funded_loop_development",
        "funded_loop_confirmation",
    }
    assert all("condition" not in f for n, _, f in called if n.startswith("bf_history_eval_"))
    assert c.state["preparation_closed"] and c.state["automatic_followup"] is False
    assert c.state["status"] == "finished_with_gaps"  # TBIG resources are explicitly unavailable


def test_manifest_removes_prior_validation_before_sampling(tmp_path, monkeypatch):
    import h5py

    from cognitive_ultrasound.config import ROOT, load
    from cognitive_ultrasound.preparation import common

    splits = dict(train=["t"], val=["old0", "old1", "v0", "v1", "v2"], test=[])
    monkeypatch.setattr(common, "read_splits", lambda _: splits)
    for split in ("train", "val"):
        (tmp_path / split).mkdir()
        for name in splits[split]:
            with h5py.File(tmp_path / split / name, "w") as f:
                f.create_dataset("data/image", data=np.full((4, 112, 112), -20, np.float32))
    manifest = tmp_path / "split.yaml"
    manifest.write_text("pinned")
    cfg = load(ROOT / "configs/preparation_closure.yaml")
    cfg.update(
        data_root=str(tmp_path),
        split_manifest=str(manifest),
        excluded_validation_cases=["old0", "old1"],
        cohorts=dict(train=1, debug=1, development=1, confirmation=1),
        frames={k: 4 for k in ["train", "debug", "development", "confirmation"]},
    )
    chosen = common.make_manifest(cfg)
    assert set(sum((v for k, v in chosen["cohorts"].items() if k != "train"), [])) == {
        "v0",
        "v1",
        "v2",
    }
