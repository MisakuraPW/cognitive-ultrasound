import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "finish_b7", Path(__file__).parents[1] / "scripts/finish_compute_lab_b7.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path):
    selection = dict(selected=["jax_50_fp16"], torch_mode="graph", combined_components=[])
    for name in ["official", "jax_50_fp16"]:
        save(
            tmp_path / "jobs" / f"confirmation_{name}_b7/result.json",
            dict(
                status="completed",
                cohort="confirmation",
                budget=7,
                profile=dict(name=name, category="A" if name == "official" else "B"),
                rows=[dict(cold=False, warm_signature_first=False, core_s=0.1, closed_loop_s=0.12)],
                quality=dict(
                    passed=True, mean_loss=[0, 0, 0], upper95=[0, 0, 0], worst_loss=[0, 0, 0]
                ),
                process_wall_s=60,
                equivalence_passed=name == "official",
            ),
        )
    save(tmp_path / "identity.json", dict(source={}, environment="fingerprint"))
    save(tmp_path / "config.json", {})
    save(tmp_path / "status.json", dict(status="paused_at_boundary", completed=[], failed=[]))
    (tmp_path / "PAUSE_AFTER_JOB").write_text("confirmation_jax_50_fp16_b7")
    return tmp_path, selection


def test_missing_or_wrong_budget_not_accepted(prepared):
    root, selection = prepared
    p = root / "jobs/confirmation_jax_50_fp16_b7/result.json"
    a = json.loads(p.read_text())
    a["budget"] = 14
    save(p, a)
    with pytest.raises(RuntimeError, match="Wrong evidence"):
        module.check_ready(root, selection)
    p.unlink()
    with pytest.raises(RuntimeError, match="not finished"):
        module.check_ready(root, selection)


def test_stop_not_cleared(prepared):
    root, selection = prepared
    (root / "STOP").touch()
    with pytest.raises(RuntimeError, match="STOP"):
        module.wait_for_boundary(root, selection, root / "control.json")
    assert (root / "STOP").exists()


def test_summary_limits_scope_and_does_not_adopt_approximation(prepared):
    root, selection = prepared
    module.write_summary(root, selection)
    text = (root / "ACCELERATION_SUMMARY.md").read_text(encoding="utf-8")
    assert "14/28 条线确认取消" in text
    assert "不自动采用近似版本" in text
    assert "jax_50_fp16" in text


def test_continuation_runs_only_foundation_preserves_identity(prepared, monkeypatch):
    from cognitive_ultrasound.compute_lab import closure, environment, report

    root, selection = prepared
    identity = (root / "identity.json").read_bytes()
    calls = []

    class FakeSuite:
        def __init__(self, cfg, root):
            self.state = dict(status="running")

        def save(self, **fields):
            self.state.update(fields)

        def job(self, name, task, backend):
            assert task["kind"] == "profile"
            calls.append(name)

        def train(self):
            calls.append("training")

    monkeypatch.setattr(module, "Suite", FakeSuite)
    monkeypatch.setattr(module, "source_identity", lambda: {})
    monkeypatch.setattr(
        environment, "probe", lambda *_: dict(fingerprint="fingerprint", passed=True)
    )
    monkeypatch.setattr(closure, "finish", lambda *_: calls.append("closure"))
    monkeypatch.setattr(report, "report", lambda *_: None)
    monkeypatch.setattr(report, "archive", lambda *_: dict(sha256="verified"))
    module.finish(root, selection, root / "control.json")
    assert calls == ["profile_official", "profile_jax_50_fp16", "closure", "training"]
    assert (root / "identity.json").read_bytes() == identity
    assert not (root / "PAUSE_AFTER_JOB").exists()
    assert json.loads((root / "scope_amendment.json").read_text())["confirmation_budgets"] == [7]


def test_changed_source_rejected_before_clearing_boundary(prepared, monkeypatch):
    root, selection = prepared
    monkeypatch.setattr(module, "source_identity", lambda: {"changed.py": "different"})
    with pytest.raises(RuntimeError, match="Pinned source changed"):
        module.finish(root, selection, root / "control.json")
    assert (root / "PAUSE_AFTER_JOB").exists()


def test_only_exact_capacity_metadata_repair_is_allowed(prepared, monkeypatch):
    root, _ = prepared
    file = "src/cognitive_ultrasound/compute_lab/suite.py"
    before = b"capacity_results.append(dict(batch=batch, **value))"
    after = b"capacity_results.append(dict(value, batch=batch))"
    save(root / "identity.json", dict(source={file: hashlib.sha256(before).hexdigest()}))
    original = (root / "identity.json").read_bytes()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: before)
    monkeypatch.setattr(
        module, "source_identity", lambda: {file: hashlib.sha256(after).hexdigest()}
    )
    module.validate_source_with_capacity_repair(root)
    assert (root / "identity.json").read_bytes() == original
    assert (root / "source_repair_capacity_merge.json").exists()
    monkeypatch.setattr(
        module, "source_identity", lambda: {file: hashlib.sha256(after + b"change").hexdigest()}
    )
    with pytest.raises(RuntimeError, match="exact one-line"):
        module.validate_source_with_capacity_repair(root)


def test_real_capacity_worker_payload_does_not_break_training_controller(tmp_path, monkeypatch):
    from cognitive_ultrasound.compute_lab import training
    from cognitive_ultrasound.compute_lab.suite import Suite

    class ReachedTrainingShort(Exception):
        pass

    monkeypatch.setattr(training, "prepare_batches", lambda *a: None)
    suite = Suite(dict(training_batch_size=32), tmp_path)

    def fake_job(name, task, backend):
        if task.get("initialize"):
            return dict(status="completed")
        if task["kind"] == "capacity":
            if task["batch"] == 1:
                return dict(status="completed", batch=1, loss=0.7)
            return dict(status="failed")
        raise ReachedTrainingShort

    monkeypatch.setattr(suite, "job", fake_job)
    with pytest.raises(ReachedTrainingShort):
        suite.train()
    result = json.loads((tmp_path / "capacity.json").read_text())
    assert result["records"] == [
        dict(batch=1, status="completed", loss=0.7),
        dict(batch=8, status="failed"),
    ]
    assert result["scientific_batch_size"] == 32
