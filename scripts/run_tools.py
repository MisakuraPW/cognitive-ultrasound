"""Estimate from measured pilots, or export completed/stopped result folders."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cognitive_ultrasound.operations import (  # noqa: E402
    estimate_evaluation,
    estimate_training,
    export_results,
)
from cognitive_ultrasound.provenance import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate-evaluation", "estimate-training"):
        sub = commands.add_parser(name)
        sub.add_argument("--pilot", required=True)
        sub.add_argument("--config", required=True)
        sub.add_argument("--output", required=True)
    sub = commands.add_parser("export")
    sub.add_argument("--source", required=True)
    sub.add_argument("--archive", required=True)
    sub.add_argument("--include-checkpoints", action="store_true")
    sub.add_argument("--include-trajectories", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        result = export_results(
            args.source, args.archive, args.include_checkpoints, args.include_trajectories
        )
    else:
        estimate = estimate_training if args.command == "estimate-training" else estimate_evaluation
        result = estimate(args.pilot, args.config)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
