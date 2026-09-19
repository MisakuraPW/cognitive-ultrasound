"""Resumable, time-capped preparation coordinator; no full training/evaluation."""

import contextlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..config import CASL_COMMIT, ROOT, ZEA_COMMIT
from ..provenance import sha256
from .common import (
    atomic_json,
    digest,
    emit,
    make_manifest,
    normalized_config,
    read_json,
    source_identity,
    verify_manifest,
)


@contextlib.contextmanager
def run_lock(root):
    """OS-owned advisory lock automatically released on crash (no stale PID deletion)."""
    file = root / "run.lock"
    with file.open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("This output directory already has a running coordinator") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        import psutil

        children = psutil.Process(process.pid).children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        _, alive = psutil.wait_procs(children, timeout=3)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def directory_bytes(root):
    size = 0
    for file in root.rglob("*"):
        try:
            if file.is_file() and not file.is_symlink():
                size += file.stat().st_size
        except FileNotFoundError:
            # A worker may atomically replace its temporary snapshot during this scan.
            continue
    return size


def bundle_results(root):
    destination = root.parent / (root.name + ".results.tar.gz")
    temporary = destination.with_suffix(".tmp")
    size = directory_bytes(root)
    if shutil.disk_usage(root).free < size + 512 * 1024**2:
        atomic_json(
            root / "archive_status.json",
            dict(status="skipped", reason="Not enough free disk for safe results copy"),
        )
        return None
    with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
        for file in sorted(root.rglob("*")):
            if (
                file.is_file()
                and file.name != "run.lock"
                and not file.is_symlink()
                and file.suffix != ".tmp"
            ):
                tar.add(
                    file, arcname=str(Path(root.name) / file.relative_to(root)), recursive=False
                )
    # Read back tar member checksums, then hash transport file. No source files deleted.
    with tarfile.open(temporary, "r:gz") as tar:
        for item in tar:
            if item.isfile():
                stream = tar.extractfile(item)
                while stream.read(1024 * 1024):
                    pass
    temporary.replace(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        sha256(destination) + "  " + destination.name + "\n", encoding="ascii"
    )
    return destination


class Coordinator:
    def __init__(self, cfg, root, resume=False, retry_failed=False):
        self.cfg, self.root = normalized_config(cfg), Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retry_failed = retry_failed
        self.stop_requested = False
        self.state = {"status": "running", "jobs": {}, "pid": os.getpid()}
        self.resume = resume

    def initialize(self):
        cfg, root = self.cfg, self.root
        weights = Path(cfg["checkpoint"])
        identity = dict(
            config=cfg,
            source=source_identity(),
            weights={
                f.name: sha256(f) for f in [weights / "config.json", weights / "model.weights.h5"]
            },
            casl_commit=CASL_COMMIT,
            zea_commit=ZEA_COMMIT,
        )
        fingerprint = digest(identity)
        if (root / "identity.json").exists():
            if not self.resume:
                raise FileExistsError("Existing run: use --resume; files are preserved")
            if read_json(root / "identity.json")["fingerprint"] != fingerprint:
                raise ValueError("Code/config/checkpoint changed; use a new output directory")
            self.state = read_json(root / "status.json")
            import psutil

            for job in self.state["jobs"].values():
                if job.get("status") == "running" and job.get("worker_pid"):
                    try:
                        process = psutil.Process(job["worker_pid"])
                        if abs(process.create_time() - job["worker_created"]) < 0.01:
                            raise RuntimeError(
                                "A previous worker is still alive; do not launch a duplicate. Inspect PID "
                                + str(process.pid)
                            )
                    except psutil.NoSuchProcess:
                        pass
            self.state.update(status="running", pid=os.getpid())
            verify_manifest(cfg, read_json(root / "manifest.json"))
        else:
            if any(f.name != "run.lock" for f in root.iterdir()):
                raise FileExistsError("Output must be empty for a fresh run")
            manifest = make_manifest(cfg)
            atomic_json(root / "manifest.json", manifest)
            atomic_json(root / "config.json", cfg)
            atomic_json(root / "identity.json", dict(fingerprint=fingerprint, **identity))
            versions = {}
            for package in (
                "jax",
                "jaxlib",
                "keras",
                "tensorflow",
                "numpy",
                "h5py",
                "scipy",
                "scikit-image",
            ):
                try:
                    versions[package] = version(package)
                except PackageNotFoundError:
                    versions[package] = None
            atomic_json(
                root / "environment.json",
                dict(
                    python=sys.version,
                    executable=sys.executable,
                    platform=platform.platform(),
                    packages=versions,
                    gpu=subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout,
                ),
            )
        self.save()

    def save(self):
        atomic_json(self.root / "status.json", self.state)

    def elapsed(self, phase=None):
        return sum(
            v.get("elapsed_s", 0)
            for v in self.state["jobs"].values()
            if phase is None or v["phase"] == phase
        )

    def complete(self, name):
        f = self.root / "jobs" / name / "result.json"
        return f.exists() and read_json(f).get("status") == "completed"

    def eligible(self, name):
        return self.complete(name) and read_json(self.root / "jobs" / name / "result.json").get(
            "eligible", False
        )

    def job(self, name, kind, phase, dependencies=(), condition=None, **fields):
        cfg, root = self.cfg, self.root
        previous = self.state["jobs"].get(name, {})
        if previous.get("status") == "completed":
            return
        if previous.get("status") == "failed" and not self.retry_failed:
            return
        spent = previous.get("elapsed_s", 0)
        record = dict(phase=phase, status="pending", elapsed_s=spent)
        self.state["jobs"][name] = record
        task = dict(id=name, kind=kind, **fields)
        reason = None
        if self.stop_requested or (root / "STOP").exists():
            reason = "Stopped by request; remove STOP and resume later"
        elif any(not self.complete(d) for d in dependencies):
            reason = "Missing completed prerequisite: " + ", ".join(dependencies)
        elif condition:
            reason = condition
        if reason:
            record.update(status="blocked", reason=reason)
            self.save()
            return
        remaining = min(
            cfg["max_hours"] * 3600 - self.elapsed(),
            cfg["phase_hours"][phase] * 3600 - self.elapsed(phase),
            fields.get("max_seconds", cfg["job_hours"] * 3600) - spent,
        )
        if remaining <= 0:
            record.update(status="capped", reason="Cumulative run/phase/job time budget exhausted")
            self.save()
            return
        output = root / "jobs" / name
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "task.json", task)
        env = dict(
            os.environ,
            PYTHONPATH=str(ROOT / "src"),
            PYTHONUNBUFFERED="1",
            PYTHONUTF8="1",
            OMP_NUM_THREADS="4",
            TF_NUM_INTRAOP_THREADS="4",
            TF_NUM_INTEROP_THREADS="2",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            MPLBACKEND="Agg",
            HF_HUB_OFFLINE="1",
        )
        python = cfg["tbig"]["python"] if kind == "tbig" else sys.executable
        command = [
            python,
            "-m",
            "cognitive_ultrasound.preparation",
            "worker",
            "--output",
            str(root),
            "--task",
            str(output / "task.json"),
        ]
        started = time.monotonic()
        record.update(status="running", command=command)
        self.save()
        emit("STAGE", job=name, phase=phase, remaining_seconds=round(remaining))
        reason = None
        with (output / "console.log").open("a", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                )
            except OSError as error:
                record.update(
                    status="failed",
                    reason=f"Cannot start worker: {error}",
                    elapsed_s=spent + time.monotonic() - started,
                )
                log.write(record["reason"] + "\n")
                self.save()
                emit("STAGE_END", job=name, **record)
                return
            import psutil

            record["worker_pid"] = process.pid
            try:
                record["worker_created"] = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                record["worker_created"] = 0
            self.save()
            try:
                while process.poll() is None:
                    now = time.monotonic() - started
                    record["elapsed_s"] = spent + now
                    if self.stop_requested or (root / "STOP").exists():
                        reason = "stop_requested"
                    elif now >= remaining:
                        reason = "timed_out"
                    elif shutil.disk_usage(root).free < cfg["min_free_gib"] * 1024**3:
                        reason = "disk_free_limit"
                    elif directory_bytes(root) > cfg["max_output_gib"] * 1024**3:
                        reason = "output_size_limit"
                    self.save()
                    if reason:
                        stop_process(process)
                        break
                    time.sleep(2)
            finally:
                stop_process(process)
                record["elapsed_s"] = spent + time.monotonic() - started
            result = read_json(output / "result.json") if (output / "result.json").exists() else {}
            if reason:
                record.update(status=reason, reason=reason)
            elif process.returncode != 0:
                record.update(
                    status="failed",
                    reason=f"Worker exit {process.returncode}; see jobs/{name}/console.log",
                )
            elif result.get("status") == "completed":
                record.update(status="completed")
            else:
                record.update(
                    status=result.get("status", "failed"),
                    reason=result.get("reason", "Worker produced no completion record"),
                )
        self.save()
        emit("STAGE_END", job=name, **record)

    def execute(self):
        from .analysis import report, select_variant

        cfg, root = self.cfg, self.root
        for variant in ["reference", "wrapper", "profile"]:
            self.job("debug_" + variant, "trajectory", "A", cohort="debug", variant=variant)
        self.job("parity", "parity", "A", dependencies=("debug_reference", "debug_wrapper"))
        for variant in cfg["candidates"]:
            self.job(
                "fixed_" + variant,
                "fixed_history",
                "A",
                dependencies=("debug_reference",),
                variant=variant,
            )
            self.job(
                "debug_" + variant,
                "trajectory",
                "A",
                dependencies=("fixed_" + variant,),
                cohort="debug",
                variant=variant,
            )
        provisional = select_variant(root, cfg)["variant"]
        self.job(
            "confirm_reference",
            "trajectory",
            "A",
            dependencies=("debug_reference",),
            cohort="confirmation",
            variant="reference",
        )
        if provisional != "reference":
            self.job(
                "confirm_" + provisional,
                "trajectory",
                "A",
                cohort="confirmation",
                variant=provisional,
            )
        chosen = select_variant(root, cfg, confirmation=True)["variant"]
        for stage in ["codec", "prior", "filter"]:
            parent = {"prior": "codec", "filter": "prior"}.get(stage)
            self.job(
                "bf_" + stage,
                "bf_train",
                "B",
                stage=stage,
                condition="Previous BF qualification did not pass"
                if parent and not self.eligible("bf_" + parent + "_qualify")
                else None,
            )
            self.job(
                "bf_" + stage + "_qualify",
                "bf_qualify",
                "B",
                dependencies=("bf_" + stage,),
                stage=stage,
            )
        self.job(
            "bf_filter_confirm",
            "bf_qualify",
            "B",
            dependencies=("bf_filter",),
            stage="filter",
            cohort="confirmation",
            condition=None
            if self.eligible("bf_filter_qualify")
            else "Filter development gate did not pass",
        )
        spec = cfg["tbig"]
        ready = all(
            spec.get(k)
            for k in ["repo", "config", "python", "expected_commit", "sequences", "asset_files"]
        )
        self.job(
            "tbig",
            "tbig",
            "C",
            condition=None
            if ready
            else "TBIG independent repository/environment/weights/compatible sequences not supplied; no download attempted",
            max_seconds=spec["max_seconds"],
        )
        self.job("branches", "branches", "C", dependencies=("confirm_" + chosen,))
        # Independent cohorts; every risk model fit uses TRAIN cases only.
        for cohort in ("train", "development", "confirmation"):
            if cohort == "confirmation" and self.complete("confirm_" + chosen):
                self.job(
                    "risk_confirmation",
                    "reuse",
                    "C",
                    source="confirm_" + chosen,
                    dependencies=("confirm_" + chosen,),
                )
                continue
            self.job(
                "risk_" + cohort,
                "trajectory",
                "C",
                dependencies=("debug_" + chosen,),
                cohort=cohort,
                variant=chosen,
            )
        self.job(
            "risk_probe",
            "risk_probe",
            "C",
            dependencies=tuple("risk_" + c for c in ("train", "development", "confirmation")),
        )
        self.job(
            "closed_loop",
            "closed_loop",
            "C",
            dependencies=("branches", "risk_probe"),
            condition=None
            if self.eligible("risk_probe")
            else "Independent future-risk prediction gate did not pass",
        )
        self.state["status"] = (
            "finished_with_gaps"
            if any(v["status"] != "completed" for v in self.state["jobs"].values())
            else "completed"
        )
        self.save()
        report(root)


def run(cfg, output, resume=False, retry_failed=False):
    coordinator = Coordinator(cfg, output, resume, retry_failed)
    with run_lock(coordinator.root):
        coordinator.initialize()
        handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(
                sig, lambda *_: setattr(coordinator, "stop_requested", True)
            )
        try:
            coordinator.execute()
        except BaseException as error:
            coordinator.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            coordinator.save()
            from .analysis import report

            report(coordinator.root)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            artifact = bundle_results(coordinator.root)
            emit(
                "RESULTS",
                output=coordinator.root,
                archive=artifact,
                status=coordinator.state["status"],
            )
