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
            "value-prepare",
            "value-probe",
            "value-calibrate",
            "value-run",
            "value-resume",
            "value-status",
            "value-stop",
            "value-worker",
            "value-report",
            "value-bundle",
            "stage-prepare",
            "stage-probe",
            "stage-calibrate",
            "stage-run",
            "stage-resume",
            "stage-status",
            "stage-stop",
            "stage-worker",
            "stage-report",
            "stage-bundle",
        ],
    )
    parser.add_argument("--config", default="configs/task_budget_ef.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task")
    parser.add_argument("--worker-output")
    parser.add_argument("--phase", choices=["P0", "P1", "P2", "P3", "P4", "P5"])
    parser.add_argument("--stage-phase")
    parser.add_argument(
        "--expand16", action="store_true", help="Explicit independent16-case diagnostic extension"
    )
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.action.startswith("stage-"):
        from . import stage_suite
        from .stage_data import config as stage_config

        action = args.action.removeprefix("stage-")
        if action == "worker":
            if args.stage_phase not in stage_suite.PHASES or args.stage_phase == "pilot_gate":
                parser.error("stage-worker requires a declared worker phase")
            from .stage_workers import worker as stage_worker

            stage_worker(
                args.stage_phase,
                read_json(root / "config.json"),
                read_json(root / "manifest.json"),
                root,
            )
        elif action == "status":
            import json

            v = (
                read_json(root / "status.json")
                if (root / "status.json").exists()
                else {"status": "not_started"}
            )
            v["coordinator_alive"] = stage_suite.process_alive(v.get("pid"), v.get("created"))
            if v.get("worker_pid"):
                v["worker_alive"] = stage_suite.process_alive(
                    v["worker_pid"], v.get("worker_created")
                )
            print(json.dumps(v, ensure_ascii=False, indent=2))
        elif action == "stop":
            root.mkdir(parents=True, exist_ok=True)
            (root / "STOP").write_text("User requested stop\n")
            state = read_json(root / "status.json") if (root / "status.json").exists() else {}
            if not stage_suite.process_alive(state.get("pid"), state.get("created")):
                stage_suite.terminate_owned_worker(state)
        elif action in ["report", "bundle"]:
            from . import stage_report

            stage_report.report(root)
            if action == "bundle":
                print(stage_report.bundle(root))
        else:
            file = (
                "configs/task_budget_stages.yaml"
                if args.config == "configs/task_budget_ef.yaml"
                else args.config
            )
            cfg = stage_config(file)
            if action == "prepare":
                stage_suite.prepare(cfg, root)
            else:
                stage_suite.run(
                    cfg,
                    root,
                    "resume" if action == "resume" else "run",
                    probe_only=action in ["probe", "calibrate"],
                )
    elif args.action.startswith("value-"):
        from . import value_suite

        action = args.action.removeprefix("value-")
        if action == "worker":
            if not args.phase:
                parser.error("value-worker requires --phase")
            from .value_workers import worker as value_worker

            value_worker(
                args.phase, read_json(root / "config.json"), read_json(root / "manifest.json"), root
            )
        elif action == "status":
            import json

            v = (
                read_json(root / "status.json")
                if (root / "status.json").exists()
                else {"status": "not_started"}
            )
            v["coordinator_alive"] = value_suite.process_alive(v.get("pid"), v.get("created"))
            if v.get("worker_pid"):
                v["worker_alive"] = value_suite.process_alive(v["worker_pid"], v["worker_created"])
            print(json.dumps(v, ensure_ascii=False, indent=2))
        elif action == "stop":
            root.mkdir(parents=True, exist_ok=True)
            (root / "STOP").write_text("User requested stop\n")
            state = read_json(root / "status.json") if (root / "status.json").exists() else {}
            if not value_suite.process_alive(state.get("pid"), state.get("created")):
                value_suite.terminate_owned_worker(state)
        elif action in ["report", "bundle"]:
            from . import value_report

            value_report.report(root)
            if action == "bundle":
                print(value_report.bundle(root))
        else:
            file = (
                "configs/task_budget_value.yaml"
                if args.config == "configs/task_budget_ef.yaml"
                else args.config
            )
            cfg = value_suite.config(file)
            if action == "prepare":
                value_suite.prepare(cfg, root, args.expand16)
            else:
                value_suite.run(
                    cfg,
                    root,
                    "resume" if action == "resume" else "run",
                    args.expand16,
                    probe_only=action in ["probe", "calibrate"],
                )
    elif args.action == "worker":
        from .experiment import worker

        spec = read_json(args.task)
        worker_output = (
            Path(args.worker_output).resolve() if args.worker_output else root / "jobs" / spec["id"]
        )
        if not worker_output.is_relative_to(root):
            raise ValueError("Worker output must stay inside its batch directory")
        worker(
            spec,
            read_json(root / "config.json"),
            read_json(root / "manifest.json"),
            root,
            worker_output,
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
