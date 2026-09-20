"""python -m cognitive_ultrasound.preparation run --config ... --output ..."""

import argparse
from pathlib import Path

from ..config import ROOT, load, path
from .common import emit, normalized_config, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "worker", "report", "plan"])
    parser.add_argument("--config", default=str(ROOT / "configs/preparation_auto.yaml"))
    parser.add_argument("--output", default="/root/autodl-tmp/outputs_casl/preparation_auto_v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--task")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.command == "worker":
        from .worker import execute

        execute(root, read_json(args.task))
    elif args.command == "report":
        from .analysis import report
        from .suite import bundle_results, run_lock

        with run_lock(root):
            report(root)
            emit("RESULTS", archive=bundle_results(root))
    elif args.command == "plan":
        cfg = normalized_config(load(path(args.config)))
        emit(
            "PLAN",
            config=cfg,
            note="No data reads or GPU execution. Per-task and cumulative caps apply; dependent jobs may be skipped.",
        )
    else:
        from .suite import run

        run(load(path(args.config)), root, args.resume, args.retry_failed)


if __name__ == "__main__":
    main()
