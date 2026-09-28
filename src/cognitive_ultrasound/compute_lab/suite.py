"""Resumable fixed experiment DAG using the existing preparation lock/process helpers."""

import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from ..config import ROOT
from ..preparation.common import atomic_json, read_json, source_identity
from ..preparation.suite import run_lock, stop_process
from ..provenance import sha256
from .data import lock_manifest, throughput
from .environment import probe
from .protocol import profiles, quality, science_identity, select
from .scheduling import cost_decision, fastest_torch, record_decision


class Stopped(Exception):
    pass


class BoundaryPause(Exception):
    pass


def owned_alive(pid, created):
    try:
        return bool(
            pid
            and psutil.Process(pid).create_time() == created
            and psutil.Process(pid).is_running()
        )
    except psutil.Error:
        return False


def stop_orphan(root):
    status = root / "status.json"
    if not status.exists():
        return
    state = read_json(status)
    if owned_alive(state.get("pid"), state.get("created")):
        return  # live coordinator receives STOP
    pid = state.get("worker_pid")
    if owned_alive(pid, state.get("worker_created")):
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for p in [*children, parent]:
            try:
                p.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs([*children, parent], timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.Error:
                pass


class Suite:
    def __init__(self, cfg, root):
        self.cfg, self.root = cfg, root
        self.state = dict(
            status="running",
            pid=os.getpid(),
            created=psutil.Process().create_time(),
            stage="initializing",
            completed=[],
            failed=[],
        )

    def save(self, **fields):
        self.state.update(fields)
        self.state["updated_at_unix"] = time.time()
        atomic_json(self.root / "status.json", self.state)

    def job(self, name, task, backend="jax"):
        if (self.root / "STOP").exists():
            raise Stopped()
        boundary = self.root / "PAUSE_AFTER_JOB"
        if boundary.exists():
            target = boundary.read_text().strip()
            if (
                target in ("initializing", "jax_boundary", "torch_boundary")
                or (self.root / "jobs" / target / "result.json").exists()
            ):
                raise BoundaryPause()
        directory = self.root / "jobs" / name
        result = directory / "result.json"
        if result.exists():
            saved = read_json(result)
            self.state["completed" if saved["status"] == "completed" else "failed"].append(name)
            # Failed candidates are recorded, never automatically retried/search-expanded.
            return saved
        directory.mkdir(parents=True, exist_ok=True)
        task = dict(task, output=str(directory))
        atomic_json(directory / "task.json", task)
        self.save(stage=name)
        print(f"STAGE {name}", flush=True)
        command = [
            self.cfg["pythons"][backend],
            "-m",
            "cognitive_ultrasound.compute_lab",
            "worker",
            "--config",
            str(self.root / "config.json"),
            "--output",
            str(self.root),
            "--task",
            str(directory / "task.json"),
        ]
        env = dict(
            os.environ,
            NVIDIA_TF32_OVERRIDE="0",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            KERAS_BACKEND="tensorflow" if backend == "tensorflow" else "jax",
            OMP_NUM_THREADS=str(self.cfg["threads"]),
            TF_NUM_INTRAOP_THREADS=str(self.cfg["threads"]),
            TF_NUM_INTEROP_THREADS="1",
            PYTHONUNBUFFERED="1",
            PYTHONUTF8="1",
        )
        start, peak, descendants = time.perf_counter(), 0, {}
        heartbeat = start
        with (directory / "console.log").open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            self.save(worker_pid=child.pid, worker_created=psutil.Process(child.pid).create_time())
            try:
                while child.poll() is None:
                    if (self.root / "STOP").exists():
                        raise Stopped()
                    try:
                        proc = psutil.Process(child.pid)
                        children = proc.children(recursive=True)
                        peak = max(peak, sum(x.memory_info().rss for x in [proc, *children]))
                        for p in children:
                            descendants[p.pid] = p.create_time()
                    except psutil.Error:
                        pass
                    if time.perf_counter() - heartbeat >= 5:
                        self.save(
                            worker_elapsed_s=time.perf_counter() - start, worker_ram_peak_bytes=peak
                        )
                        heartbeat = time.perf_counter()
                    if shutil.disk_usage(self.root).free < 2 * 1024**3:
                        raise RuntimeError(
                            "Disk reserve below 2 GiB; worker stopped, no incomplete output accepted"
                        )
                    time.sleep(0.5)
            finally:
                if child.poll() is None:
                    stop_process(child)
                if os.name != "nt":
                    # A killed/OOM coordinator child can exit before we observe descendants.
                    # The dedicated session owns this process group, including orphan compilers.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                for pid, created in descendants.items():
                    try:
                        p = psutil.Process(pid)
                        if p.create_time() == created and p.is_running():
                            p.terminate()
                            try:
                                p.wait(5)
                            except psutil.TimeoutExpired:
                                p.kill()
                    except psutil.Error:
                        pass
        if not result.exists():
            atomic_json(
                result,
                dict(
                    status="failed",
                    returncode=child.returncode,
                    reason="worker failed; see console.log; no precision fallback",
                ),
            )
        value = read_json(result)
        value.update(process_wall_s=time.perf_counter() - start, process_ram_peak_bytes=peak)
        atomic_json(result, value)
        self.state["completed" if value["status"] == "completed" else "failed"].append(name)
        self.save(worker_pid=None, worker_created=None)
        print(f"STAGE_END {name}: {value['status']} ({value['process_wall_s']:.1f}s)", flush=True)
        return value

    def inference(self, p, cohort, budget, short=False):
        name = f"{'short' if short else cohort}_{p.name}_b{budget}"
        record = self.job(
            name,
            dict(kind="inference", profile=asdict(p), cohort=cohort, budget=budget, short=short),
            p.backend,
        )
        if record["status"] == "completed":
            refname = f"{'short' if short else cohort}_official_b{budget}"
            ref = read_json(self.root / "jobs" / refname / "result.json")
            record["quality"] = quality(ref["rows"], record["rows"], cohort == "confirmation")
            if p.backend == "torch" and p.category == "B" and cohort == "development":
                parent_file = (
                    self.root / "jobs" / f"development_torch_{p.mode}_b{budget}" / "result.json"
                )
                parent = read_json(parent_file)
                record["within_backend_quality"] = quality(parent["rows"], record["rows"])
                record["attribution"] = (
                    "Also compared with same-mode Torch FP32/50; official comparison includes framework-port differences"
                )
            # Complete trajectory plus replay/operator gate, not just mean image metrics.
            debug = self.root / "jobs" / f"short_{p.name}_b14" / "result.json"
            diag = read_json(debug) if debug.exists() else record
            replay_pass = bool(diag.get("replay")) and all(
                c["passed"] for r in diag.get("replay", []) for c in r["checks"].values()
            )
            operator = diag.get("operator_checks")
            op_pass = operator.get("passed", False) if operator else False
            record["equivalence_passed"] = record["equivalence_passed"] and replay_pass and op_pass
            record["internal_correctness"] = bool(
                diag.get("operator_checks", {}).get("internal_correctness", False)
            )
            record["verdict"] = (
                "bitwise_same"
                if record["equivalence_passed"] and record["bitwise"]
                else "within_tolerance"
                if record["equivalence_passed"]
                else "quality_close_only"
                if record["quality"]["passed"]
                else "failed"
            )
            atomic_json(self.root / "jobs" / name / "result.json", record)
        else:
            record.update(
                profile=p.record(),
                cohort=cohort,
                budget=budget,
                verdict="failed",
                equivalence_passed=False,
            )
            atomic_json(self.root / "jobs" / name / "result.json", record)
            if p.name == "official":
                raise RuntimeError(
                    f"Official reference failed in {name}; dependent comparisons stopped"
                )
        return record

    def calibration(self, candidates):
        self.job("export", dict(kind="export"))
        short_results = {}
        for p in candidates.values():
            short_results[p.name] = self.inference(p, "debug", 14, True)
            if p.name == "official" and short_results[p.name]["status"] != "completed":
                raise RuntimeError(
                    "Official short reference failed; inspect its console before rerun"
                )
        manifest = read_json(self.root / "manifest.json")
        estimates = []
        for name, r in short_results.items():
            if r["status"] != "completed":
                estimates.append(dict(name=name, status=r["status"]))
                continue
            rows = r["rows"]
            warm = np_mean([x["closed_loop_s"] for x in rows if x["frame"] >= 2])
            cold = np_mean([x["closed_loop_s"] for x in rows if x["cold"]])
            dev_frames = (
                sum(
                    min(128, manifest["files"][n]["frames"])
                    for n in manifest["cohorts"]["development"]
                )
                * 2
            )
            conf_frames = (
                sum(manifest["files"][n]["frames"] for n in manifest["cohorts"]["confirmation"]) * 9
            )
            estimates.append(
                dict(
                    name=name,
                    status="measured",
                    development_hours=(warm * (dev_frames - 16) + cold * 16) / 3600,
                    confirmation_hours=(warm * (conf_frames - 288) + cold * 288) / 3600,
                    caveat="projection; excludes new signatures, I/O contention and diagnostic overhead",
                )
            )
        atomic_json(
            self.root / "estimates.json",
            dict(
                candidates=estimates,
                total_gpu_cap=None,
                frozen_search=True,
                confirm_at_most=4,
                automatic_approximation_adoption=False,
            ),
        )
        return short_results

    def worthwhile(self, profile, short):
        reference = read_json(self.root / "jobs/short_official_b14/result.json")
        decision = cost_decision(short, reference)
        record_decision(self.root, profile.name, decision)
        # Never recompute a compatible completed job just because the new cost gate differs.
        complete = self.root / "jobs" / f"development_{profile.name}_b14/result.json"
        return (complete.exists() and read_json(complete)["status"] == "completed") or decision[
            "proceed"
        ]

    def train(self):
        from .training import compare, prepare_batches

        report = {}
        (self.root / "training").mkdir(exist_ok=True)
        for workload in ("casl", "codec", "prior", "filter"):
            batch_file = self.root / "training" / (workload + ".batches.json")
            if not batch_file.exists():
                prepare_batches(self.cfg, workload, batch_file)
            init = self.job(
                "train_" + workload + "_init",
                dict(kind="training", workload=workload, mode="eager", initialize=True),
                "tensorflow",
            )
            if init["status"] != "completed":
                report[workload] = dict(status="blocked", reason="initialization failed")
                continue
            if workload == "casl":
                capacity_results = []
                for batch in (1, 8, 16, 32, 64, 128):
                    value = self.job(
                        f"capacity_casl_{batch}", dict(kind="capacity", batch=batch), "tensorflow"
                    )
                    capacity_results.append(dict(batch=batch, **value))
                    if value["status"] != "completed":
                        break
                atomic_json(
                    self.root / "capacity.json",
                    dict(
                        records=capacity_results,
                        scientific_batch_size=self.cfg["training_batch_size"],
                        advisory_only=True,
                        adopted=False,
                        gradient_checkpointing="not enabled; memory/compute tradeoff, not presumed faster",
                    ),
                )
            values = {}
            training_shorts = {}
            for candidate_mode in ("eager", "graph"):
                training_shorts[candidate_mode] = self.job(
                    f"train_{workload}_{candidate_mode}_short",
                    dict(kind="training", workload=workload, mode=candidate_mode, until=3),
                    "tensorflow",
                )
            successful = {m: r for m, r in training_shorts.items() if r["status"] == "completed"}
            projected = {
                m: 2
                * (
                    r["records"][0]["seconds"]
                    + 199 * np_mean([x["seconds"] for x in r["records"][1:]])
                )
                for m, r in successful.items()
            }
            print(
                f"TRAINING_COST {workload}: 400 actual updates per admitted mode (200 continuous + 100 + 100 resume), projected seconds={projected}; initialization/checkpoints extra",
                flush=True,
            )
            for mode in ("eager", "graph"):
                prefix = f"train_{workload}_{mode}"
                warmup = self.job(
                    prefix + "_short",
                    dict(kind="training", workload=workload, mode=mode, until=3),
                    "tensorflow",
                )
                if warmup["status"] != "completed":
                    continue
                if (
                    mode == "graph"
                    and "eager" in projected
                    and projected[mode] > 4 * projected["eager"]
                ):
                    record_decision(
                        self.root,
                        prefix,
                        dict(
                            proceed=False,
                            kind="performance_not_worth_continuing",
                            projected_400_s=projected[mode],
                            reference_400_s=projected["eager"],
                            maximum_ratio=4,
                            reason="Graph route exceeds finite relative cost gate; eager recovery still executed",
                        ),
                    )
                    continue
                estimate = warmup["records"][0]["seconds"] + 199 * np_mean(
                    [r["seconds"] for r in warmup["records"][1:]]
                )
                atomic_json(
                    self.root / "training" / (workload + "_" + mode + "_estimate.json"),
                    dict(
                        measured_updates=3,
                        projected_200_s=estimate,
                        projected_total_400_s=2 * estimate,
                        production_state_advanced=False,
                    ),
                )
                full = self.job(
                    prefix + "_continuous",
                    dict(kind="training", workload=workload, mode=mode, until=200),
                    "tensorflow",
                )
                first = self.job(
                    prefix + "_100",
                    dict(kind="training", workload=workload, mode=mode, until=100),
                    "tensorflow",
                )
                if first["status"] == "completed":
                    self.job(
                        prefix + "_resumed",
                        dict(
                            kind="training",
                            workload=workload,
                            mode=mode,
                            until=200,
                            resume_from=str(
                                self.root / "jobs" / (prefix + "_100") / "checkpoint.npz"
                            ),
                        ),
                        "tensorflow",
                    )
                a, b = (
                    self.root / "jobs" / (prefix + "_continuous"),
                    self.root / "jobs" / (prefix + "_resumed"),
                )
                if all((x / "checkpoint.npz").exists() for x in (a, b)):
                    values[mode] = dict(recovery=compare(a, b), elapsed_s=full["process_wall_s"])
                    resumed_rows = read_json(b / "result.json")["records"]
                    expected_rows = full["records"][100:]
                    from .protocol import numeric

                    values[mode]["recovery_loss"] = numeric(
                        [x["loss"] for x in resumed_rows], [x["loss"] for x in expected_rows]
                    )
            if len(values) == 2:
                a, b = [
                    self.root / "jobs" / f"train_{workload}_{m}_continuous"
                    for m in ("eager", "graph")
                ]
                values["engineering_comparison"] = compare(a, b)
                from .protocol import numeric

                values["engineering_loss"] = numeric(
                    [r["loss"] for r in read_json(a / "result.json")["records"]],
                    [r["loss"] for r in read_json(b / "result.json")["records"]],
                )
                values["recommended"] = (
                    "graph"
                    if values["engineering_comparison"]["passed"]
                    and values["engineering_loss"]["passed"]
                    and values["graph"]["recovery"]["passed"]
                    and values["graph"]["recovery_loss"]["passed"]
                    and values["graph"]["elapsed_s"] < values["eager"]["elapsed_s"]
                    else "eager"
                )
            report[workload] = values
            atomic_json(self.root / "training_report.json", report)

    def run(self, action, phase="all"):
        candidates = profiles()
        jax = {k: p for k, p in candidates.items() if p.backend == "jax"}
        short = self.calibration(jax)
        if action == "calibrate":
            return
        dev = []
        for p in jax.values():
            if short[p.name]["status"] == "completed" and self.worthwhile(p, short[p.name]):
                value = self.inference(p, "development", 14)
                if value["status"] == "completed":
                    dev.append(value)
        self.save(stage="jax_boundary")
        if phase == "jax":
            raise BoundaryPause()
        # Repair validation always precedes extension. Eager is retained historical evidence,
        # never the implicit fallback for four expensive approximation DEV jobs.
        for mode in ("compile", "graph"):
            p = candidates["torch_" + mode]
            record = self.inference(p, "debug", 14, True)
            internal = record.get("operator_checks", {}).get("internal_correctness", False)
            if record["status"] == "completed" and internal and self.worthwhile(p, record):
                value = self.inference(p, "development", 14)
                if value["status"] == "completed":
                    dev.append(value)
            else:
                record_decision(
                    self.root,
                    p.name,
                    dict(
                        proceed=False,
                        kind=record.get("failure_kind", "numerical_correctness_failure")
                        if not internal
                        else "performance_not_worth_continuing",
                        reason="Mode must pass internal changed-input/gradient checks and finite cost gate before DEV",
                    ),
                )
        fastest = fastest_torch(dev)
        record_decision(
            self.root,
            "torch_extension",
            dict(
                proceed=fastest is not None,
                mode=fastest,
                kind="cost_eligible" if fastest else "performance_or_correctness_no_eligible_mode",
                reason="Only repaired compile/graph faster than official DEV; no eager fallback",
            ),
        )
        self.save(stage="torch_boundary")
        if phase == "torch":
            raise BoundaryPause()
        combination = [
            x["profile"]["name"]
            for x in dev
            if x["profile"]["name"].startswith("jax_")
            and x["profile"]["category"] == "A"
            and x["equivalence_passed"]
        ]
        additional = profiles(fastest, combination)
        for name in sorted(set(additional) - set(candidates)):
            p = additional[name]
            short_value = self.inference(p, "debug", 14, True)
            if short_value["status"] == "completed" and self.worthwhile(p, short_value):
                value = self.inference(p, "development", 14)
                if value["status"] == "completed":
                    dev.append(value)
        selected = select(dev)
        selection = dict(
            selected=selected,
            torch_mode=fastest,
            combined_components=combination,
            basis="development only; confirmation not inspected",
            adopted="official",
        )
        lock = self.root / "selection.json"
        if lock.exists() and read_json(lock) != selection:
            raise ValueError("Locked selection changed")
        atomic_json(lock, selection)
        atomic_json(self.root / "versions.json", {k: p.record() for k, p in additional.items()})
        if selected:
            for budget in (7, 14, 28):
                self.inference(additional["official"], "confirmation", budget)
                for name in selected:
                    self.inference(additional[name], "confirmation", budget)
        for name in ["official", *selected]:
            self.job(
                "profile_" + name,
                dict(kind="profile", profile=asdict(additional[name])),
                additional[name].backend,
            )
        from .closure import finish

        finish(self)
        self.train()


def np_mean(values):
    return sum(values) / len(values) if values else 0.0


def ensure_identity(file, identity):
    if file.exists() and read_json(file) != identity:
        raise ValueError("Hardware/environment/source/science changed: use a NEW output directory")
    atomic_json(file, identity)


def run(cfg, root, action="run", phase="all", inherit_from=None):
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        if (root / "STOP").exists():
            raise RuntimeError("STOP requested; use resume to explicitly clear it")
        status_file = root / "status.json"
        if status_file.exists():
            old = read_json(status_file)
            if owned_alive(old.get("worker_pid"), old.get("worker_created")):
                raise RuntimeError("Previous worker still alive; use stop before resume")
        cfg = dict(cfg, checkpoint_sha256=sha256(Path(cfg["checkpoint"]) / "model.weights.h5"))
        cfg["checkpoint_files"] = {
            f.name: sha256(f) for f in Path(cfg["checkpoint"]).glob("*") if f.is_file()
        }
        if cfg["training_batch_size"] != 32:
            raise ValueError(
                "Protocol fixes scientific training batch=32; capacity cannot replace it"
            )
        existing = root / "config.json"
        if existing.exists() and read_json(existing) != cfg:
            raise ValueError("Config changed: use a new output directory")
        atomic_json(existing, cfg)
        environment = probe(cfg, root / "probe")
        if not environment["frameworks"]["jax"].get("passed"):
            raise RuntimeError("Official JAX GPU probe failed; see probe/jax.log")
        if inherit_from:
            from .migration import inherit

            inherit(inherit_from, root, cfg, environment["fingerprint"])
        from .report import closure_audit

        closure_audit(cfg, root)
        manifest = lock_manifest(cfg, root)
        identity = dict(
            science=science_identity(cfg, manifest),
            source=source_identity(),
            environment=environment["fingerprint"],
        )
        identity_file = root / "identity.json"
        ensure_identity(identity_file, identity)
        atomic_json(root / "versions.json", {k: p.record() for k, p in profiles().items()})
        if not (root / "throughput.json").exists():
            throughput(cfg, manifest, root)
        # State/noise cache upper estimate, not a GPU-time cap. Fail before filling disk.
        needed = sum(x["frames"] for x in manifest["files"].values()) * 9 * 800000
        if shutil.disk_usage(root).free < needed + 10 * 1024**3:
            raise RuntimeError(
                f"Insufficient disk for reference cache (estimate {needed / 1024**3:.1f} GiB + 10 GiB reserve)"
            )
        suite = Suite(cfg, root)
        try:
            suite.run(action, phase)
            suite.save(
                status="calibrated"
                if action == "calibrate"
                else "finished_with_gaps"
                if suite.state["failed"]
                else "completed"
            )
        except Stopped:
            suite.save(status="stopped")
        except BoundaryPause:
            suite.save(status="paused_at_boundary", worker_pid=None, worker_created=None)
        except BaseException as exc:
            suite.save(status="failed", error=str(exc))
            raise
        finally:
            from .report import report

            report(cfg, root)
        if suite.state["status"] in ("completed", "finished_with_gaps"):
            from .report import archive

            try:
                atomic_json(
                    root.with_name(root.name + ".bundle.json"),
                    archive(root, root.with_name(root.name + ".results.tar.gz")),
                )
            except Exception as exc:
                suite.save(status="archive_failed", error=str(exc), experiments_finished=True)
                raise
