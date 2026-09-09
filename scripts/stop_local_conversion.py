"""Stop only the verified local conversion tree; atomic published files are retained."""

import json
import os
import subprocess
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results/local_conversion/status.json"


def verify_target(command, cwd, raw, output):
    expected = str(ROOT / "scripts/local_conversion.py")
    if expected not in command or Path(cwd).resolve() != ROOT:
        raise RuntimeError("Refusing to stop an unrelated process")
    for option, value in (("--raw", raw), ("--output", output)):
        if (
            option not in command
            or Path(command[command.index(option) + 1]).resolve() != Path(value).resolve()
        ):
            raise RuntimeError("Refusing to stop a conversion with different paths")


def main():
    if os.name != "nt":
        raise RuntimeError("Use on the local Windows computer only")
    record = json.loads(STATUS.read_text(encoding="utf-8"))
    pid = int(record["pid"])
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print("Recorded local runner is already stopped.")
        return
    verify_target(process.cmdline(), process.cwd(), record["raw"], record["output"])
    members = [process] + process.children(recursive=True)
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=True)
    _, alive = psutil.wait_procs(members, timeout=30)
    if alive:
        raise RuntimeError("Some conversion processes remain; do not restart yet")
    record.update(status="stopped", stop_reason="User requested resource/performance tuning")
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(STATUS)
    print("Stopped the conversion tree. Published HDF5 files are unchanged.")


if __name__ == "__main__":
    main()
