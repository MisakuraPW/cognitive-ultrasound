"""Reuse the project's locks/process cleanup; finite EF batch with update/case resume."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from ..config import CASL_COMMIT, ROOT, ZEA_COMMIT
from ..hardware import snapshot
from ..preparation.common import atomic_json, digest, emit, read_json
from ..preparation.suite import run_lock, stop_process
from ..provenance import sha256
from .data import lock_manifest


def prepare(cfg, root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists() and read_json(root / "config.json") != cfg:
        raise ValueError("Existing batch configuration changed; use a new directory")
    atomic_json(root / "config.json", cfg)
    with (root / "prepare.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                cfg["torch_python"],
                "-u",
                "-m",
                "cognitive_ultrasound.task_budget.ef_worker",
                "prepare",
                "--config",
                str(root / "config.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        started = time.perf_counter()
        while process.poll() is None:
            if (root / "STOP").exists() or time.perf_counter() - started > 1800:
                stop_process(process)
                raise InterruptedError("Asset preparation stopped or exceeded30 minutes")
            time.sleep(1)
        if process.wait():
            raise RuntimeError("EF asset preparation failed; see prepare.log")
    return lock_manifest(cfg, root)


def identity(cfg, manifest):
    files = [*Path(__file__).parent.glob("*.py")] + [
        ROOT / "src/cognitive_ultrasound" / p
        for p in (
            "official.py",
            "config.py",
            "data.py",
            "experiments.py",
            "evaluation/metrics.py",
            "preparation/common.py",
            "preparation/suite.py",
        )
    ]
    query = "import sys,json; from importlib.metadata import version; print(json.dumps(dict(python=sys.version,packages={n:version(n) for n in ['torch','torchvision','numpy','h5py']})))"
    task_env = json.loads(
        subprocess.check_output([cfg["torch_python"], "-c", query], text=True, timeout=60)
    )
    weights = Path(cfg["checkpoint"])
    return dict(
        config=digest(cfg),
        manifest=manifest["identity"],
        source={str(f.relative_to(ROOT)): sha256(f) for f in sorted(files)},
        checkpoint={f.name: sha256(f) for f in sorted(weights.iterdir()) if f.is_file()},
        ef_weights=sha256(cfg["ef_weights"]),
        ef_stats=sha256(cfg["ef_stats"]),
        hardware=snapshot(),
        python=sys.executable,
        task_environment=task_env,
        upstream=dict(casl=CASL_COMMIT, zea=ZEA_COMMIT),
        runtime_environment={
            k: os.environ.get(k)
            for k in (
                "LD_LIBRARY_PATH",
                "OMP_NUM_THREADS",
                "NVIDIA_TF32_OVERRIDE",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    )


def process_alive(pid, created):
    if not pid or not created:
        return False
    try:
        p = psutil.Process(pid)
        return abs(p.create_time() - created) < 0.05 and p.is_running()
    except psutil.NoSuchProcess:
        return False


def terminate_owned_worker(status):
    pid, created = status.get("worker_pid"), status.get("worker_created")
    if pid and created and process_alive(pid, created):
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        _, alive = psutil.wait_procs([*children, process], timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass


def jobs(cfg):
    if cfg.get("execution",{}).get("repair_milestones"):
        from .repair import repair_jobs
        return repair_jobs(cfg)
    specs = [dict(id="probe", kind="probe")]
    for seed in cfg["seeds"]:
        for pair in cfg["budgets"]["fixed_sweep"]:
            for cohort in ("development", "confirmation"):
                specs.append(
                    dict(
                        id=f"E0_{pair[0]}_{pair[1]}_s{seed}_{cohort}",
                        kind="evaluate",
                        method="E0",
                        seed=seed,
                        fixed=pair,
                        cohort=cohort,
                    )
                )
        for method in ("E1", "E2"):
            for index, weight in enumerate(cfg["training"]["lambdas"]):
                train_id = f"{method}_l{index}_s{seed}_train"
                specs.append(
                    dict(id=train_id, kind="train", method=method, seed=seed, **{"lambda": weight})
                )
                for cohort in ("development", "confirmation"):
                    specs.append(
                        dict(
                            id=f"{method}_l{index}_s{seed}_{cohort}",
                            kind="evaluate",
                            method=method,
                            seed=seed,
                            cohort=cohort,
                            parent=train_id,
                            **{"lambda": weight},
                        )
                    )
    # Keep confirmation blind until every predeclared train/development job is terminal.
    return [s for s in specs if s.get("cohort") != "confirmation"] + [
        s for s in specs if s.get("cohort") == "confirmation"
    ]


def run(cfg, root, action="run", phase="all"):
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        old = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if action == "resume":
            terminate_owned_worker(old)
            (root / "STOP").unlink(missing_ok=True)
        elif (root / "STOP").exists():
            raise RuntimeError("Stopped batch; explicitly use resume")
        emit("STAGE", job="assets_and_manifest")
        manifest = prepare(cfg, root)
        if cfg.get("feature_calibration") and "feature_scales" not in cfg:
            raise ValueError("Prepare frozen TRAIN-only feature scales before launching repaired training")
        current = identity(cfg, manifest)
        ident = root / "identity.json"
        if ident.exists() and read_json(ident) != current:
            raise ValueError(
                "Source/config/data/weights/hardware changed; use a new output directory"
            )
        atomic_json(ident, current)
        plan = jobs(cfg)
        # All weights/seeds/working points frozen before the first task result.
        atomic_json(root / "plan.json", plan)
        state = dict(
            status="running",
            pid=os.getpid(),
            created=psutil.Process().create_time(),
            phase=phase,
            failures=[],
            started=time.time(),
        )
        atomic_json(root / "status.json", state)
        try:
            from .staging import execution_schedule, pilot_report
            schedule = execution_schedule(plan, cfg, manifest) if phase == "all" else [(s, False) for s in plan]
            atomic_json(root / "execution_schedule.json", [dict(job=s["id"] if s else "PILOT_REPORT", pilot=p) for s,p in schedule])
            for spec, is_pilot in schedule:
                if spec is None:
                    first = pilot_report(root, cfg, manifest)
                    emit("PILOT_READY", status=first["status"], report=str(root/"PILOT_REPORT.md"))
                    if first["status"] != "completed":
                        raise RuntimeError("Pilot incomplete; preserve evidence and do not launch remaining batch")
                    state.update(pilot_completed=True, execution_phase="full")
                    atomic_json(root/"status.json",state)
                    continue
                if phase in ("probe", "calibrate") and spec["id"] != "probe":
                    break
                if (root / "STOP").exists():
                    state["status"] = "stopped"
                    break
                directory = root / "jobs" / spec["id"]
                state["execution_phase"] = "pilot" if is_pilot else "full"
                if spec.get("_pilot") and (directory/"result.json").exists() and read_json(directory/"result.json").get("status")=="completed":
                    emit("REUSE", job=spec["id"], reason="full result already includes pilot cases")
                    continue
                result = directory / ("pilot_result.json" if spec.get("_pilot") else "result.json")
                if result.exists():
                    value = read_json(result)
                    if value.get("status") == "completed":
                        emit("REUSE", job=spec["id"], scope="pilot" if is_pilot else "full")
                        continue
                    if action == "resume":
                        # Explicit bounded retry, preserving original terminal evidence.
                        attempts = directory / "attempts"
                        attempts.mkdir(exist_ok=True)
                        shutil.copy2(result, attempts / f"{time.time_ns()}.json")
                        result.unlink()
                    else:
                        state["failures"].append(spec["id"])
                        continue
                if (
                    spec.get("method") == "E1"
                    and read_json(root / "jobs/probe/result.json").get("gs_status") != "passed"
                ):
                    directory.mkdir(parents=True, exist_ok=True)
                    atomic_json(
                        result,
                        dict(
                            status="blocked", reason="GS task-gradient gate failed; no RL fallback"
                        ),
                    )
                    state["failures"].append(spec["id"])
                    continue
                if spec.get("parent"):
                    parent = root / "jobs" / spec["parent"]
                    if (
                        not (parent / "result.json").exists()
                        or read_json(parent / "result.json").get("status") != "completed"
                    ):
                        directory.mkdir(parents=True, exist_ok=True)
                        atomic_json(result, dict(status="blocked", reason="Training parent failed"))
                        state["failures"].append(spec["id"])
                        continue
                    spec = dict(spec, checkpoint=str(parent / "policy.npz"))
                if spec.get("resume_parent"):
                    previous=root/"jobs"/spec["resume_parent"]
                    if not (previous/"result.json").exists() or read_json(previous/"result.json").get("status")!="completed":
                        raise RuntimeError("Previous training milestone failed; do not restart from scratch")
                    prior=read_json(previous/"result.json")
                    if prior.get("method")!=spec["method"] or prior.get("seed")!=spec["seed"] or prior.get("cost_weight")!=spec["lambda"] or prior["updates"]>=spec["end_update"]:
                        raise ValueError("Incompatible training milestone checkpoint")
                    spec=dict(spec,resume_checkpoint=str(previous/"policy.npz"))
                if spec.get("case_limit"):
                    spec=dict(spec,_case_names=manifest["cohorts"][spec["cohort"]][:spec["case_limit"]])
                if shutil.disk_usage(root).free < cfg["runtime"]["min_free_gib"] * 2**30:
                    raise RuntimeError("Insufficient storage reserve")
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / "task.json", spec)
                emit("STAGE", job=spec["id"], execution_phase=state["execution_phase"], cases=spec.get("_case_names"))
                with (directory / "console.log").open("a", encoding="utf-8") as stream:
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "cognitive_ultrasound.task_budget",
                            "worker",
                            "--output",
                            str(root),
                            "--task",
                            str(directory / "task.json"),
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name != "nt",
                    )
                    created = psutil.Process(proc.pid).create_time()
                    state.update(job=spec["id"], worker_pid=proc.pid, worker_created=created)
                    atomic_json(root / "status.json", state)
                    start = time.perf_counter()
                    stopped = False
                    ram_peak = 0
                    while proc.poll() is None:
                        try:
                            owned = psutil.Process(proc.pid)
                            ram_peak = max(
                                ram_peak,
                                sum(
                                    p.memory_info().rss
                                    for p in [owned, *owned.children(recursive=True)]
                                    if p.is_running()
                                ),
                            )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        if (root / "STOP").exists() or time.perf_counter() - start > (
                            cfg["runtime"]["job_timeout_hours"] * 3600
                        ):
                            stopped = (root / "STOP").exists()
                            stop_process(proc)
                            break
                        time.sleep(1)
                    code = proc.wait()
                state.pop("worker_pid", None)
                state.pop("worker_created", None)
                if stopped:
                    state["status"] = "stopped"
                    atomic_json(root / "status.json", state)
                    break
                if code or not result.exists():
                    atomic_json(
                        result,
                        dict(
                            status="failed",
                            returncode=code,
                            reason="Worker exited/timeout; inspect console.log, preserve partial updates",
                        ),
                    )
                    state["failures"].append(spec["id"])
                    if spec["id"] == "probe":
                        raise RuntimeError("Perception/task probe failed; see probe console.log")
                if result.exists():
                    details = read_json(result)
                    details.update(
                        process_wall_s=time.perf_counter() - start,
                        process_ram_sample_peak_bytes=ram_peak,
                        memory_scope="worker plus EF/compiler children RSS sum, 1Hz sampled; may miss spikes",
                    )
                    atomic_json(result, details)
                emit(
                    "STAGE_END",
                    job=spec["id"],
                    status=read_json(result)["status"],
                    seconds=time.perf_counter() - start,
                )
                atomic_json(root / "status.json", state)
                if cfg.get("execution",{}).get("repair_milestones"):
                    if spec["id"]=="probe" and read_json(result).get("gs_status")!="passed":
                        raise RuntimeError("Repair GPU task-gradient gate failed; stop before policy training")
                    from .report import report
                    report(root)
                    if read_json(result).get("status")!="completed":
                        raise RuntimeError("Repaired worker failed; preserve checkpoints and stop this batch")
            else:
                state["status"] = "completed" if not state["failures"] else "finished_with_gaps"
            if state["status"] == "running":
                state["status"] = "calibrated"
            state["finished"] = time.time()
            atomic_json(root / "status.json", state)
            from .report import bundle, report

            report(root)
            if phase == "all" and state["status"] in ("completed", "finished_with_gaps"):
                bundle(root)
        except BaseException as error:
            state.update(
                status="failed", error=f"{type(error).__name__}: {error}", finished=time.time()
            )
            atomic_json(root / "status.json", state)
            raise
