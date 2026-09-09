import json
import runpy
import subprocess
from pathlib import Path

import pytest

from cognitive_ultrasound.config import ROOT


@pytest.fixture
def runner():
    scope = runpy.run_path(str(ROOT / "scripts/autodl_overnight.py"))
    return scope["run"].__globals__


def test_full_plan_orders_training_after_paper_and_uses_resume_only_when_present(runner, tmp_path):
    paths = {
        "raw_root": str(tmp_path / "raw"),
        "polar_root": str(tmp_path / "polar"),
        "output_root": str(tmp_path / "outputs"),
    }
    plan = runner["steps"](tmp_path, paths, True, True)
    names = [name for name, _ in plan]
    assert names.index("conversion") < names.index("audit") < names.index("paper")
    assert names.index("paper") < names.index("training_pilot") < names.index("training_estimate")
    assert names.index("training_estimate") < names.index("training") < names.index("paper_trained")
    commands = dict(plan)
    assert "--resume" not in commands["training"]
    assert "--resume" in commands["paper_trained"]
    manifest = Path(paths["output_root"]) / "training_fp32/training_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    assert "--resume" in dict(runner["steps"](tmp_path, paths, False, True))["training"]
    resumed = dict(runner["steps"](tmp_path, paths, False, True, 8))["conversion"]
    assert "--resume" in resumed and resumed[resumed.index("--workers") + 1] == "8"
    assert "training" not in dict(runner["steps"](tmp_path, paths, False))
    uploaded = dict(runner["steps"](tmp_path, paths, False, True, 8, prepared_data=True))
    assert "conversion" not in uploaded and "audit" in uploaded and "training" in uploaded


@pytest.mark.parametrize("shutdown", [False, True])
def test_failed_stage_stops_later_work_and_records_error(runner, monkeypatch, tmp_path, shutdown):
    (tmp_path / "configs").mkdir()
    paths = {
        "polar_root": str(tmp_path / "polar"),
        "raw_root": str(tmp_path / "raw"),
        "output_root": str(tmp_path / "outputs"),
    }
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setitem(runner, "ROOT", tmp_path)
    monkeypatch.setitem(runner, "validate_config_paths", lambda *args: None)
    monkeypatch.setitem(runner, "storage_check", lambda *args: {})
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda p: str(p).replace("\\", "/") == "/usr/bin/shutdown" or real_is_file(p),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if any("check_gpu.py" in token for token in command):
            raise subprocess.CalledProcessError(2, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert runner["run"](paths, state, shutdown, True) == 1
    result = json.loads((state / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed" and result["stage"] == "gpu"
    assert result["completed_stages"] == []
    assert len(calls) == 2 + int(shutdown)  # freeze, GPU, optional explicitly requested shutdown
    if shutdown:
        assert calls[-1] == ["/usr/bin/shutdown"]


def test_success_exports_both_results_and_training_after_all_stages(runner, monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    paths = {
        "polar_root": str(tmp_path / "polar"),
        "raw_root": str(tmp_path / "raw"),
        "output_root": str(tmp_path / "outputs"),
    }
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setitem(runner, "ROOT", tmp_path)
    monkeypatch.setitem(runner, "validate_config_paths", lambda *args: None)
    monkeypatch.setitem(runner, "validate_paths", lambda *args: None)
    monkeypatch.setitem(runner, "storage_check", lambda *args: {})
    calls, exports = [], []

    def fake_run(command, **kwargs):
        assert kwargs["env"]["OMP_NUM_THREADS"] == "8"
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    def fake_export(source, archive, **kwargs):
        assert "paper_trained" in calls[-1][-2]
        exports.append((source, kwargs))
        return {"archive": str(archive), "sha256": "test-digest"}

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(runner, "export_results", fake_export)
    assert runner["run"](paths, state, False, True) == 0
    result = json.loads((state / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["completed_stages"][-2:] == ["paper_trained", "export"]
    assert result["training_archive_sha256"] == "test-digest"
    assert len(exports) == 2 and exports[1][1] == {"include_checkpoints": True}


def test_storage_floor_fails_before_conversion(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner["shutil"], "disk_usage", lambda folder: type("Disk", (), {"free": 50 * 2**30})()
    )
    with pytest.raises(RuntimeError, match="180 GiB"):
        runner["storage_check"]({"polar_root": str(tmp_path / "new")}, True)
