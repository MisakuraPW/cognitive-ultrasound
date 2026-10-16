import argparse
import json
from pathlib import Path

import psutil

from .data import read


def status(root):
    root = Path(root)
    state = (
        read(root / "status.json") if (root / "status.json").exists() else {"status": "not_started"}
    )
    try:
        p = psutil.Process(state["pid"])
        state["alive"] = (
            p.is_running()
            and p.status() != psutil.STATUS_ZOMBIE
            and abs(p.create_time() - state["created"]) < 0.05
        )
    except (KeyError, psutil.NoSuchProcess):
        state["alive"] = False
    if (root / "progress.json").exists():
        state["progress"] = read(root / "progress.json")
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["run", "resume", "probe", "status", "stop", "report"])
    p.add_argument("--config", default="configs/fast_belief.yaml")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Functional local validation only, never a scientific GPU fallback",
    )
    a = p.parse_args()
    root = Path(a.output)
    if a.action == "status":
        print(json.dumps(status(root), ensure_ascii=False))
        return
    if a.action == "stop":
        if status(root).get("alive"):
            root.joinpath("STOP").touch()
            print("Stop requested. Wait for status=stopped before shutting down.")
        else:
            print("No live coordinator; existing evidence preserved.")
        return
    if a.action == "report":
        if status(root).get("alive"):
            raise RuntimeError("Do not bundle a running batch")
        from .report import report

        print(json.dumps(report(root), ensure_ascii=False))
        return
    from .runner import run

    try:
        run(a.config, root, a.cpu, a.action == "probe", resume=a.action == "resume")
    except InterruptedError:
        print("Stopped with resumable checkpoint.", flush=True)


if __name__ == "__main__":
    main()
