"""Queue a finite CPU analysis ONLY after the136 stop/export boundary."""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


@contextlib.contextmanager
def output_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = path.with_suffix(".tmp")
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    p.replace(path)


def alive(pid, created):
    if not pid or not created:
        return False
    try:
        p = psutil.Process(pid)
        return (
            p.is_running()
            and p.status() != psutil.STATUS_ZOMBIE
            and abs(p.create_time() - created) < 0.05
        )
    except psutil.NoSuchProcess:
        return False


def ready(source):
    s = read(source / "status.json")
    manager = source / ".finish_136/status.json"
    if manager.exists() and read(manager)["status"] == "failed":
        raise RuntimeError("136 closure watcher failed; analysis must not start")
    if alive(s.get("pid"), s.get("created")) or alive(s.get("worker_pid"), s.get("worker_created")):
        return False
    p = source / "scope_136_bundle_receipt.json"
    if not p.exists():
        if manager.exists() and alive(read(manager).get("pid"), read(manager).get("created")):
            return False
        raise RuntimeError("Training ended without validated136 closure")
    scope = read(source / "scope_closure_136.json")
    receipt = read(p)
    if scope["status"] != "completed" or scope["target_updates"] != 136 or not receipt["verified"]:
        raise ValueError("Wrong analysis boundary")
    with open(receipt["archive"], "rb") as f:
        h = (
            hashlib.file_digest(f, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(f.read()).hexdigest()
        )
    if h != receipt["sha256"]:
        raise ValueError("136 archive checksum mismatch")
    return manager.exists() and read(manager)["status"] == "completed"


def queue(source, out):
    out = Path(out).resolve()
    with output_lock(out / ".queue.lock"):
        return queue_impl(source, out)


def queue_impl(source, out):
    source = Path(source).resolve()
    out = Path(out).resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError("Offline analysis must use a separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    status_file = out / "queue_status.json"
    if status_file.exists():
        old = read(status_file)
        if alive(old.get("pid"), old.get("created")) or alive(
            old.get("analysis_pid"), old.get("analysis_created")
        ):
            raise RuntimeError("Another analysis queue already owns this directory")
    state = dict(
        status="queued_after136",
        pid=os.getpid(),
        created=psutil.Process().create_time(),
        source=str(source),
        output=str(out),
        cpu_only=True,
        started_epoch=time.time(),
    )
    write(status_file, state)
    try:
        while not ready(source):
            if time.time() - state["started_epoch"] > 48 * 3600:
                raise TimeoutError("Analysis queue timed out before136 closure")
            time.sleep(2)
        state.update(status="running_cpu_analysis", analysis_started_epoch=time.time())
        write(status_file, state)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONUTF8": "1",
        }
        with (out / "analysis.console.log").open("a", encoding="utf-8") as log:
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).with_name("analyze_task_budget_stages_136.py")),
                    "--source",
                    str(source),
                    "--output",
                    str(out),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            state.update(analysis_pid=p.pid, analysis_created=psutil.Process(p.pid).create_time())
            write(status_file, state)
            try:
                p.wait(timeout=2 * 3600)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                raise
            if p.returncode:
                raise RuntimeError(
                    "Offline analysis failed; see analysis.console.log; no new GPU work is launched"
                )
        result = read(out / "complete.json")
        receipt = read(out / "bundle_receipt.json")
        state.update(status=result["status"], finished_epoch=time.time(), receipt=receipt)
        write(status_file, state)
        print(json.dumps(dict(event="OFFLINE_ANALYSIS_COMPLETED", **receipt)), flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(status_file, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    queue(args.source, args.output)
