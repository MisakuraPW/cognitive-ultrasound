import argparse
import json
import traceback
from pathlib import Path

from ..config import load, path
from ..preparation.common import atomic_json, read_json


def main():
    p = argparse.ArgumentParser(description="Fixed acceleration audit; no complete CASL retraining")
    p.add_argument(
        "command",
        choices=[
            "probe",
            "calibrate",
            "run",
            "resume",
            "report",
            "status",
            "stop",
            "worker",
            "framework",
            "bundle",
            "closure",
        ],
    )
    p.add_argument("--config", default="configs/compute_lab.yaml")
    p.add_argument("--output", default="/root/autodl-tmp/outputs_casl/compute_lab_v1")
    p.add_argument("--task")
    p.add_argument("--framework", choices=["jax", "tensorflow", "torch"])
    args = p.parse_args()
    root = Path(args.output).resolve()
    if args.command == "framework":
        from .environment import probe_framework

        try:
            result = probe_framework(args.framework)
        except Exception as exc:
            traceback.print_exc()
            result = dict(passed=False, error=str(exc))
        atomic_json(root, result)
        return
    if args.command == "status":
        from .suite import owned_alive

        state = read_json(root / "status.json")
        state["coordinator_alive"] = owned_alive(state.get("pid"), state.get("created"))
        state["worker_alive"] = owned_alive(state.get("worker_pid"), state.get("worker_created"))
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.command == "stop":
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP").touch()
        from .suite import stop_orphan

        stop_orphan(root)
        print("STOP requested; coordinator will stop and reap its worker tree")
        return
    cfg = read_json(args.config) if str(args.config).endswith(".json") else load(path(args.config))
    for key in ("checkpoint", "split_manifest", "casl_training_config", "bf_config"):
        cfg[key] = str(path(cfg[key]).resolve())
    if args.command == "worker":
        task = read_json(args.task)
        output = Path(task["output"])
        try:
            if task["kind"] == "inference":
                from .inference import run

                run(task, cfg, root, output)
            elif task["kind"] == "export":
                from .inference import export

                export(cfg, root / "export")
                atomic_json(output / "result.json", dict(status="completed"))
            elif task["kind"] == "training":
                from .training import worker

                worker(task, cfg, root, output)
            elif task["kind"] in ("profile", "capacity"):
                from . import diagnostics

                getattr(diagnostics, task["kind"])(task, cfg, root, output)
            elif task["kind"] == "closure_repair":
                from .closure import repair

                repair(task, output)
        except BaseException as exc:
            atomic_json(
                output / "result.json",
                dict(status="failed", error=type(exc).__name__ + ": " + str(exc)),
            )
            raise
    elif args.command == "probe":
        from .environment import probe

        print(json.dumps(probe(cfg, root / "probe"), indent=2))
    elif args.command in ("closure", "report", "bundle"):
        from ..preparation.suite import run_lock
        from .report import archive, closure_audit, report

        root.mkdir(parents=True, exist_ok=True)
        with run_lock(root):
            if args.command == "closure":
                closure_audit(cfg, root)
            report(cfg, root)
            if args.command == "bundle":
                result = archive(root, root.with_name(root.name + ".results.tar.gz"))
                atomic_json(root.with_name(root.name + ".bundle.json"), result)
    else:
        from .suite import run

        if args.command == "resume":
            from ..preparation.suite import run_lock

            with run_lock(root):
                (root / "STOP").unlink(missing_ok=True)
        run(cfg, root, "calibrate" if args.command == "calibrate" else "run")


if __name__ == "__main__":
    main()
