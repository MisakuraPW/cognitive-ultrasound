"""Umbrella preserves terminal results and sequences independent finite suites."""

import os
from pathlib import Path

import pytest

from cognitive_ultrasound import all_preparation as app
from cognitive_ultrasound.preparation.common import atomic_json, read_json


def make_stages(tmp_path):
    return [
        dict(name=n, module=m, config=str(tmp_path / (n + ".yaml")), output=str(tmp_path / n))
        for n, m in zip(["closure", "torch"], app.MODULES)
    ]


def finish(folder, status="completed", jobs=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "status.json", dict(status=status, jobs=jobs or {}))
    (folder / "REPORT.md").write_text("fixture report")
    Path(str(folder) + ".results.tar.gz").touch()
    Path(str(folder) + ".results.tar.gz.sha256").touch()


def test_completed_and_closed_gaps_skip_without_spawning(tmp_path, monkeypatch):
    stages = make_stages(tmp_path)
    finish(
        stages[0]["output"],
        "finished_with_gaps",
        {"tbig": {"status": "blocked", "reason": "assets absent"}},
    )
    finish(stages[1]["output"])

    def forbidden(*a, **k):
        raise AssertionError("Completed suites must not rerun")

    monkeypatch.setattr(app.subprocess, "Popen", forbidden)
    app.run(tmp_path / "control", stages)
    state = read_json(tmp_path / "control/status.json")
    assert state["status"] == "finished_with_gaps"
    assert all(s["skipped_existing"] for s in state["stages"].values())


def test_stop_is_not_misreported_as_terminal_and_resume_is_explicit(tmp_path):
    stages = make_stages(tmp_path)
    finish(stages[0]["output"], "finished_with_gaps", {"job": {"status": "stop_requested"}})
    assert not app.is_finished(stages[0]["output"])
    (Path(stages[0]["output"]) / "STOP").touch()
    with pytest.raises(RuntimeError, match="paused"):
        app.run(tmp_path / "control", stages)
    assert (Path(stages[0]["output"]) / "STOP").exists()


def test_failure_of_first_suite_still_attempts_second_sequentially(tmp_path, monkeypatch):
    stages = make_stages(tmp_path)
    calls = []

    class FakeProcess:
        def __init__(self, command, **kw):
            if calls:
                assert calls[-1].polled
            import io

            self.stdout = io.StringIO("test progress\n")
            self.polled = False
            self.pid = os.getpid()
            self.returncode = 1 if not calls else 0
            self.output = Path(command[command.index("--output") + 1])
            finish(self.output, "failed" if self.returncode else "completed")
            calls.append(self)

        def poll(self):
            self.polled = True
            return self.returncode

    monkeypatch.setattr(app.subprocess, "Popen", FakeProcess)
    app.run(tmp_path / "control", stages)
    assert len(calls) == 2
    state = read_json(tmp_path / "control/status.json")
    assert state["stages"]["closure"]["status"] == "failed"
    assert state["stages"]["torch"]["status"] == "completed"


def test_existing_identity_adds_resume_but_never_torch_retry_flag(tmp_path):
    stage = make_stages(tmp_path)[1]
    folder = Path(stage["output"])
    folder.mkdir()
    (folder / "identity.json").write_text("{}")
    cmd = app.child_command(stage, retry_failed=True)
    assert "--resume" in cmd and "--retry-failed" not in cmd


def test_no_duplicate_live_coordinator(tmp_path):
    import psutil

    with pytest.raises(RuntimeError, match="still running"):
        app.assert_idle(
            dict(stages={"closure": dict(pid=os.getpid(), created=psutil.Process().create_time())})
        )
