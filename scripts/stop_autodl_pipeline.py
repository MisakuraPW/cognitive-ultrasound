"""Stop only this checkout's detached pipeline process group before switching converters."""

import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = Path("/root/autodl-tmp/outputs_casl/overnight/status.json")


def verify_target(pid, cmdline, cwd, pgid):
    if pid <= 1 or pgid != pid or Path(cwd).resolve() != ROOT.resolve():
        raise RuntimeError(
            "PID/cwd/process group does not match this detached pipeline; refusing to stop"
        )
    target = str(ROOT / "scripts/autodl_overnight.py")
    if target not in cmdline:
        raise RuntimeError("PID is not this checkout's unattended runner; refusing to stop")


def group_alive(pgid):
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            # comm may contain spaces and parentheses; fields after final ')' begin with state.
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
                return True
        except (OSError, IndexError, ValueError):
            continue
    return False


def main():
    if sys.platform != "linux":
        raise RuntimeError("Run this script only on the AutoDL Linux instance")
    if not STATUS.exists():
        print("No pipeline status found; nothing to stop.")
        return
    record = json.loads(STATUS.read_text())
    pid = int(record["pid"])
    proc = Path("/proc") / str(pid)
    if proc.exists():
        cmdline = (proc / "cmdline").read_bytes().decode().split("\0")
        verify_target(pid, cmdline, (proc / "cwd").resolve(), os.getpgid(pid))
        print(f"Stopping verified pipeline group {pid}, including its converter.", flush=True)
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif group_alive(pid):
        raise RuntimeError("Runner is absent but child processes remain; inspect before restarting")
    deadline = time.monotonic() + 30
    while group_alive(pid):
        if time.monotonic() >= deadline:
            raise RuntimeError("Pipeline has not fully stopped; do not start another converter yet")
        time.sleep(0.5)
    record.update(status="stopped", stop_reason="User requested parallel conversion/resume")
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(STATUS)
    print("Pipeline stopped. Complete HDF5 files are retained; safe to start the new runner.")


if __name__ == "__main__":
    main()
