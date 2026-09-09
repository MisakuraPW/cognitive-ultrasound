"""Run full CASL preprocessing on the local CPU, independently of AutoDL GPU stages."""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cognitive_ultrasound.config import ROOT  # noqa: E402
from cognitive_ultrasound.conversion import atomic_json, conversion_lock  # noqa: E402
from cognitive_ultrasound.data import audit_dataset, convert, read_splits  # noqa: E402


def available_memory_gib():
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        value = MemoryStatus()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError("Cannot read available physical memory")
        return value.available / 2**30
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("Cannot read available memory")


def preflight(raw, output, workers):
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Output overlaps raw data")
    if not (raw / "Videos").is_dir():
        raise FileNotFoundError(raw / "Videos")
    required = {
        name for names in read_splits(ROOT / "configs/splits/split.yaml").values() for name in names
    }
    present = {f"{p.stem}.hdf5" for p in (raw / "Videos").glob("*.avi") if p.stat().st_size > 0}
    if required - present:
        raise ValueError(f"Missing {len(required - present)} required videos")
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free / 2**30
    required_disk = 180 if not output.exists() or not any(output.iterdir()) else 10
    available = available_memory_gib()
    # Two real short AVIs peaked at ~1.1 GiB for the process tree on this host.
    # Reserve extra room for longer videos and coordinator/runtime overhead.
    required_memory = 1.5 if workers == 1 else 1 + workers
    print(
        f"Input videos: {len(present)}; free disk: {free:.1f} GiB; available RAM: {available:.1f} GiB; workers: {workers}",
        flush=True,
    )
    if free < required_disk:
        raise RuntimeError(f"Need at least {required_disk} GiB free disk space")
    if available < required_memory:
        raise RuntimeError(
            f"Need at least {required_memory} GiB available RAM for {workers} workers; close unused applications first"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 33), default=2)
    parser.add_argument("--start", action="store_true", help="Detach and write persistent logs")
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    preflight(raw, output, args.workers)
    state = ROOT / "results/local_conversion"
    state.mkdir(parents=True, exist_ok=True)
    if args.start:
        command = [
            sys.executable,
            "-X",
            "utf8",
            str(Path(__file__).resolve()),
            "--raw",
            str(raw),
            "--output",
            str(output),
            "--workers",
            str(args.workers),
        ]
        options = (
            {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        with open(state / "run.log", "a", encoding="utf-8") as stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                **options,
            )
        print(f"Started PID {child.pid}; log: {state / 'run.log'}; status: {state / 'status.json'}")
        return
    with conversion_lock(state):
        record = {
            "status": "running",
            "pid": os.getpid(),
            "raw": str(raw),
            "output": str(output),
            "workers": args.workers,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "conversion",
        }
        atomic_json(state / "status.json", record)
        try:
            convert(raw, output, ROOT / "configs/splits/split.yaml", args.workers, resume=True)
            record["stage"] = "audit"
            atomic_json(state / "status.json", record)
            audit_dataset(
                output, ROOT / "configs/splits/split.yaml", state / "dataset_statistics.md"
            )
            record["status"] = "completed"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(state / "status.json", record)
            print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
