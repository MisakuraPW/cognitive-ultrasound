"""EF E0/E1/E2 experiment command line; imports ML libraries only in workers."""

import argparse
from pathlib import Path

from ..preparation.common import read_json
from .data import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "probe",
            "calibrate",
            "run",
            "resume",
            "status",
            "stop",
            "worker",
            "report",
            "bundle",
        ],
    )
    parser.add_argument("--config", default="configs/task_budget_ef.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.action == "worker":
        from .experiment import worker

        spec = read_json(args.task)
        worker(
            spec,
            read_json(root / "config.json"),
            read_json(root / "manifest.json"),
            root,
            root / "jobs" / spec["id"],
        )
    elif args.action == "status":
        import json

        from .suite import process_alive

        value = (
            read_json(root / "status.json")
            if (root / "status.json").exists()
            else {"status": "not_started"}
        )
        value["coordinator_alive"] = process_alive(value.get("pid", 0), value.get("created", 0))
        if value.get("worker_pid"):
            value["worker_alive"] = process_alive(value["worker_pid"], value["worker_created"])
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.action == "stop":
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP").write_text("User requested stop\n")
        from .suite import process_alive, terminate_owned_worker

        state = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if not process_alive(state.get("pid", 0), state.get("created", 0)):
            terminate_owned_worker(state)
    elif args.action in ("report", "bundle"):
        from .report import bundle, report

        report(root)
        if args.action == "bundle":
            print(bundle(root))
    else:
        cfg = configuration(args.config)
        from .suite import prepare, run

        if args.action == "prepare":
            prepare(cfg, root)
        else:
            run(
                cfg,
                root,
                "resume" if args.action == "resume" else "run",
                args.action if args.action in ("probe", "calibrate") else "all",
            )


if __name__ == "__main__":
    main()
