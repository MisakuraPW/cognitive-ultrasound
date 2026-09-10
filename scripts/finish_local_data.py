"""Wait, repair, audit, archive, verify, then remove only the authorized derived folder."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cognitive_ultrasound.conversion import atomic_json, conversion_lock  # noqa: E402

SOURCE = Path(r"G:\SRTP\dataset\CASL-EchoNet-polar")
RAW = Path(r"G:\SRTP\dataset\EchoNet-Dynamic")
EXPORT = Path(r"G:\SRTP\dataset")
STATE = ROOT / "results/local_conversion"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_cleanup(report):
    source, raw, archive = SOURCE.resolve(), RAW.resolve(), Path(report["archive"]).resolve()
    if source != Path(r"G:\SRTP\dataset\CASL-EchoNet-polar") or source == raw:
        raise ValueError("Unexpected resolved cleanup target")
    if SOURCE.is_symlink() or source in archive.parents or raw in archive.parents:
        raise ValueError("Unsafe archive or cleanup path")
    if report["status"] != "verified" or Path(report["source"]).resolve() != source:
        raise ValueError("Missing verification for exact derived source")
    if (archive.stat().st_size, archive.stat().st_mtime_ns) != (
        report["archive_bytes"],
        report["archive_mtime_ns"],
    ):
        raise ValueError("Verified archive changed")
    if not (raw / "Videos").is_dir():
        raise ValueError("Original raw videos must still exist")
    if read(STATE / "status.json")["status"] != "completed":
        raise ValueError("Dataset audit is not completed")
    if read(source / "conversion_failures.json"):
        raise ValueError("Dataset has failures")


def cleanup(report):
    check_cleanup(report)
    # One native PowerShell command verifies exact absolute path and all reparse
    # points, then deletes only the explicitly authorized derived folder.
    command = r"""
$ErrorActionPreference = 'Stop'
$target = Get-Item -LiteralPath 'G:\SRTP\dataset\CASL-EchoNet-polar' -Force
if ($target.FullName -ne 'G:\SRTP\dataset\CASL-EchoNet-polar' -or -not $target.PSIsContainer) { throw 'Wrong cleanup target' }
if ($target.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point target' }
if (Get-ChildItem -LiteralPath $target.FullName -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | Select-Object -First 1) { throw 'Reparse point inside target' }
if (-not (Test-Path -LiteralPath 'G:\SRTP\dataset\EchoNet-Dynamic\Videos' -PathType Container)) { throw 'Raw data missing' }
Remove-Item -LiteralPath $target.FullName -Recurse -Force
if (Test-Path -LiteralPath 'G:\SRTP\dataset\CASL-EchoNet-polar') { throw 'Cleanup incomplete' }
"""
    shell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    subprocess.run([str(shell), "-NoProfile", "-NonInteractive", "-Command", command], check=True)
    report.update(source_deleted=True, deleted_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(STATE / "package_status.json", report)
    atomic_json(Path(report["archive"]).with_suffix(".verification.json"), report)


def run():
    state = {
        "status": "running",
        "pid": os.getpid(),
        "stage": "waiting_for_conversion",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Keep Windows awake for this process lifetime, without changing power settings.
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001) if os.name == "nt" else None
    lockdir = STATE / "finish_lock"
    lockdir.mkdir(exist_ok=True)
    with conversion_lock(lockdir):
        atomic_json(STATE / "finish_status.json", state)
        try:
            original = read(STATE / "status.json")
            if (
                Path(original["raw"]).resolve() != RAW
                or Path(original["output"]).resolve() != SOURCE
            ):
                raise ValueError("Conversion path mismatch")
            try:
                proc = psutil.Process(original["pid"])
                if str(ROOT / "scripts/local_conversion.py") not in proc.cmdline():
                    raise ValueError("Waiting PID is not this local conversion")
                while proc.is_running():
                    time.sleep(15)
            except psutil.NoSuchProcess:
                pass
            latest = read(STATE / "status.json")
            if latest["pid"] != original["pid"]:
                raise ValueError("Another conversion was started during the wait")
            if latest["status"] == "failed":
                state["stage"] = "repair_and_audit"
                atomic_json(STATE / "finish_status.json", state)
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "utf8",
                        str(ROOT / "scripts/repair_conversion_failures.py"),
                    ],
                    cwd=ROOT,
                    check=True,
                )
            elif latest["status"] != "completed":
                raise RuntimeError(
                    "Conversion exited without final status; refusing unsafe continuation"
                )
            state["stage"] = "archive_and_verify"
            atomic_json(STATE / "finish_status.json", state)
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(ROOT / "scripts/build_data_tar.py"),
                    "--source",
                    str(SOURCE),
                    "--archive",
                    str(EXPORT / "CASL-EchoNet-polar.tar"),
                ],
                cwd=ROOT,
                check=True,
            )
            report = read(STATE / "package_status.json")
            state["stage"] = "remove_verified_derived_folder"
            atomic_json(STATE / "finish_status.json", state)
            cleanup(report)
            state.update(
                status="completed", stage="finished", archive=report["archive"], source_deleted=True
            )
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(STATE / "finish_status.json", state)
            print(json.dumps(state, indent=2), flush=True)
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if args.start:
        with (STATE / "finish.log").open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-X", "utf8", str(Path(__file__).resolve())],
                cwd=ROOT,
                env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        print(f"Started finish coordinator PID {proc.pid}")
    else:
        run()
