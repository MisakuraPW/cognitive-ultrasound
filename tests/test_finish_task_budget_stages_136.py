"""Finite closure does not rewrite protocol, discard checkpoints or run GPU jobs."""

import importlib.util

import pytest

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.preparation.common import atomic_json, digest, read_json
from cognitive_ultrasound.provenance import sha256
from cognitive_ultrasound.task_budget.value_storage import commit

spec = importlib.util.spec_from_file_location(
    "finish136", ROOT / "scripts/finish_task_budget_stages_136.py"
)
finish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finish)


@pytest.fixture
def ready(tmp_path):
    root = tmp_path / "run"
    atomic_json(root / "identity.json", dict(source={}, config="unchanged"))
    atomic_json(root / "config.json", dict(original_updates=264))
    atomic_json(root / "plan.json", dict(original_updates=264))
    atomic_json(root / "manifest.json", dict(cohorts=dict(development=["case.hdf5"])))
    atomic_json(
        root / "status.json", dict(status="stopped", phase=finish.BOUNDARY, pid=None, created=None)
    )
    batch = digest(read_json(root / "identity.json"))
    for arm in ["T1", "Q2", "T2"]:
        p = root / "training" / arm / "updates" / "00136"
        atomic_json(p / "record.json", dict(update=136, arm=arm))
        (p / "state.npz").write_bytes(b"preserved checkpoint fixture")
        commit(
            p,
            dict(batch=batch, arm=arm, update=136),
            dict(update=136, arm=arm),
            ["record.json", "state.npz"],
        )
    for phase, update in [("full_pass_1_evaluate", 72), (finish.BOUNDARY, 136)]:
        rows = [
            dict(
                arm=arm,
                case="case.hdf5",
                seed=42,
                update=update,
                ef_error=4.0,
                mean_lines=14.0,
                mse_ratio=0.01,
                p90_ratio=0.02,
            )
            for arm in ["T1", "Q2", "T2"]
        ]
        value = dict(status="completed", cases=rows)
        directory = root / "jobs" / phase
        atomic_json(directory / "result.json", value)
        commit(directory, dict(batch=batch, phase=phase), value, ["result.json"])
    return root


def test_ready_checks_all_models_and_development(ready):
    assert finish.boundary_ready(ready)
    record = ready / "training/Q2/updates/00136/record.json"
    record.write_text("corrupted")
    with pytest.raises(ValueError):
        finish.boundary_ready(ready)


def test_not_ready_before_boundary(tmp_path):
    atomic_json(tmp_path / "identity.json", {})
    assert finish.boundary_ready(tmp_path) is False


def test_export_preserves_protocol_and_checkpoint_and_verifies_archive(ready):
    before = {
        name: sha256(ready / name)
        for name in [
            "identity.json",
            "config.json",
            "plan.json",
            "training/T1/updates/00136/state.npz",
        ]
    }
    receipt = finish.export(ready, dict(coordinator_exited=True, forced=False))
    assert receipt["verified"] and sha256(receipt["archive"]) == receipt["sha256"]
    assert all(sha256(ready / name) == v for name, v in before.items())
    assert read_json(ready / "scope_closure_136.json")["confirmation_evaluated"] is False
    assert "state.npz" not in " ".join(receipt["source_hashes"])


def test_watch_stops_then_exports_and_is_idempotent(ready, monkeypatch):
    s = read_json(ready / "status.json")
    s["status"] = "running"
    atomic_json(ready / "status.json", s)
    real_ready = finish.boundary_ready
    attempts = []

    def boundary(root):
        attempts.append(1)
        return False if len(attempts) == 1 else real_ready(root)

    monkeypatch.setattr(finish, "boundary_ready", boundary)
    monkeypatch.setattr(finish, "process_alive", lambda *args: True)
    monkeypatch.setattr(finish.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        finish, "wait_stopped", lambda *args: dict(coordinator_exited=True, forced=False)
    )
    finish.watch(ready)
    assert (ready / "STOP").exists()
    assert read_json(ready / ".finish_136/status.json")["status"] == "completed"
    old = read_json(ready / "scope_136_bundle_receipt.json")
    finish.watch(ready)
    assert read_json(ready / "scope_136_bundle_receipt.json") == old


def test_upstream_failure_does_not_claim_completion(tmp_path):
    root = tmp_path / "run"
    atomic_json(root / "identity.json", {})
    atomic_json(
        root / "status.json",
        dict(status="failed", phase="full_pass_2_train", pid=None, created=None),
    )
    with pytest.raises(RuntimeError):
        finish.watch(root)
    assert (root / "STOP").exists()
    assert not (root / "scope_closure_136.json").exists()
    assert read_json(root / ".finish_136/status.json")["status"] == "failed"
