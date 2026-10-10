"""Pilot-first one-click execution, bounded workers and content-checked resume."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from ..preparation.common import atomic_json, digest, emit, read_json
from ..preparation.suite import run_lock
from .stage_data import prepare
from .suite import process_alive as process_alive
from .suite import terminate_owned_worker
from .value_storage import committed
from .value_suite import cleanup_worker

PHASES = (
    "probe",
    "pilot_baselines",
    "pilot_train",
    "pilot_evaluate",
    "pilot_gate",
    "full_baselines",
    "full_pass_1_train",
    "full_pass_1_evaluate",
    "full_pass_2_train",
    "full_pass_2_evaluate",
    "full_pass_4_train",
    "full_pass_4_evaluate",
    "select",
    "confirmation_baselines",
    "confirmation_policies",
)


def run(cfg, root, action="run", probe_only=False, launch=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        old = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if action == "resume":
            terminate_owned_worker(old)
            (root / "STOP").unlink(missing_ok=True)
        elif (root / "STOP").exists():
            raise ValueError("Stopped batch; use resume explicitly")
        initial = not (root / "manifest.json").exists()
        if not cfg["stage_budget"]["functional_fixture"]:
            required = cfg["stage_budget"]["fresh_free_gib"] if initial else 2
            if shutil.disk_usage(root).free < required * 2**30:
                raise OSError(f"Require {required}GiB free; preserve previous data")
        cfg, m = prepare(cfg, root)
        batch = digest(read_json(root / "identity.json"))
        atomic_json(
            root / "plan.json",
            dict(
                phases=PHASES,
                pilot_train=m["pilot_train"],
                pilot_development=m["pilot_development"],
                pass_ends=m["pass_ends"],
                updates_per_arm=len(m["curriculum"]),
                automatic_full_after_healthy_pilot=True,
                no_positive_effect_gate=True,
                no_automatic_search_or_extra_epochs=True,
            ),
        )
        state = dict(
            status="running",
            pid=os.getpid(),
            created=psutil.Process().create_time(),
            started=old.get("started", time.time()),
            completed_phases=[],
            failures=[],
            functional_fixture=cfg["stage_budget"]["functional_fixture"],
        )
        atomic_json(root / "status.json", state)
        try:
            for phase in PHASES[:1] if probe_only else PHASES:
                unit = dict(batch=batch, phase=phase)
                directory = root / "jobs" / phase
                prior = committed(directory, unit)
                if prior is not None:
                    state["completed_phases"].append(phase)
                    emit("REUSE", phase=phase)
                    continue
                if (root / "STOP").exists():
                    raise InterruptedError("User requested stop")
                state.update(phase=phase)
                atomic_json(root / "status.json", state)
                emit("STAGE", phase=phase)
                if phase == "pilot_gate":
                    from .stage_report import pilot_gate

                    value = pilot_gate(root)
                    if value["status"] != "passed":
                        raise RuntimeError(
                            f"Pilot implementation health failed: {value['reasons']}"
                        )
                    from .value_storage import commit

                    result = {**value, "status": "completed", "gate_status": value["status"]}
                    atomic_json(directory / "result.json", result)
                    commit(directory, unit, result, ["result.json"])
                    emit(
                        "PILOT_READY", report=str(root / "PILOT_REPORT.md"), automatic_continue=True
                    )
                elif launch:
                    launch(phase, cfg, m, root)
                else:
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory / "console.log").open("a", encoding="utf-8") as log:
                        proc = subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                "-m",
                                "cognitive_ultrasound.task_budget",
                                "stage-worker",
                                "--output",
                                str(root),
                                "--stage-phase",
                                phase,
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=os.name != "nt",
                            env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                        )
                        state.update(
                            worker_pid=proc.pid,
                            worker_created=psutil.Process(proc.pid).create_time(),
                        )
                        atomic_json(root / "status.json", state)
                        started, last = time.monotonic(), None
                        try:
                            while proc.poll() is None:
                                if (root / "STOP").exists():
                                    raise InterruptedError("User requested stop")
                                if (
                                    time.monotonic() - started
                                    > cfg["runtime"]["job_timeout_hours"] * 3600
                                ):
                                    raise TimeoutError("Bounded worker timeout")
                                p = root / "worker_progress.json"
                                if p.exists():
                                    progress = read_json(p)
                                    if progress != last and progress.get("phase") == phase:
                                        state["progress"] = progress
                                        atomic_json(root / "status.json", state)
                                        if (
                                            progress.get("frame", 0) % 16 == 0
                                            or "operation" in progress
                                        ):
                                            emit("PROGRESS", **progress)
                                        last = progress
                                time.sleep(0.5)
                            if proc.returncode:
                                raise RuntimeError(f"{phase} failed; see {directory}/console.log")
                        finally:
                            cleanup_worker(proc)
                            state.pop("worker_pid", None)
                            state.pop("worker_created", None)
                result = committed(directory, unit)
                if result is None or result["status"] != "completed":
                    raise RuntimeError(f"{phase} did not commit a complete result")
                state["completed_phases"].append(phase)
                emit("STAGE_END", phase=phase, status="completed", seconds=result.get("seconds"))
                atomic_json(root / "status.json", state)
                from .stage_report import report

                report(root)
            state["status"] = "probed" if probe_only else "completed"
            state.pop("progress", None)
        except InterruptedError:
            state["status"] = "stopped"
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            state["failures"].append(state.get("phase", "prepare"))
            raise
        finally:
            state["finished"] = time.time()
            atomic_json(root / "status.json", state)
            from .stage_report import report

            report(root)
        if state["status"] == "completed":
            from .stage_report import bundle

            emit("BUNDLE", **bundle(root))
        return state
