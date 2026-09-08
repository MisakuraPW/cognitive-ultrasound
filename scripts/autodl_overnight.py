"""Run the EchoNet pipeline unattended on Linux; opt into full training with --with-training.

--start detaches from SSH. --shutdown-on-exit explicitly opts into powering off the
AutoDL instance after success or failure. Existing partial conversions are never overwritten.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cognitive_ultrasound.cloud import validate_paths  # noqa: E402
from cognitive_ultrasound.config import ROOT, load  # noqa: E402
from cognitive_ultrasound.operations import export_results  # noqa: E402


def steps(project, paths, convert_needed, with_training=False):
    py = sys.executable
    cli = [py, "-m", "cognitive_ultrasound"]
    prep = Path(paths["output_root"]) / "preparation"
    plan = [
        ("gpu", [py, str(project / "scripts/check_gpu.py"), "--output", str(prep / "gpu.json")]),
        (
            "inputs",
            [
                py,
                str(project / "scripts/autodl_preflight.py"),
                "--config",
                str(project / "configs/autodl/paths.yaml"),
                "--output",
                str(prep / "preflight.json"),
            ],
        ),
        ("environment", [*cli, "doctor", "--output", str(prep / "environment.md")]),
        ("assets", [*cli, "fetch-assets", "--with-evaluation"]),
    ]
    if convert_needed:
        plan.append(
            (
                "conversion",
                [*cli, "prepare-data", "--raw", paths["raw_root"], "--output", paths["polar_root"]],
            )
        )
    plan.append(
        (
            "audit",
            [
                *cli,
                "audit-data",
                "--data-root",
                paths["polar_root"],
                "--output",
                str(prep / "dataset_statistics.md"),
            ],
        )
    )
    for name, config in (("demo", "demo.yaml"), ("pilot", "pilot.yaml")):
        plan.append(
            (
                name,
                [
                    *cli,
                    "evaluate",
                    "--config",
                    str(project / "configs/autodl" / config),
                    "--resume",
                ],
            )
        )
    plan.append(
        (
            "estimate",
            [
                py,
                str(project / "scripts/run_tools.py"),
                "estimate-evaluation",
                "--pilot",
                str(Path(paths["output_root"]) / "pilot"),
                "--config",
                str(project / "configs/autodl/paper.yaml"),
                "--output",
                str(prep / "evaluation-eta.json"),
            ],
        )
    )
    plan.append(
        (
            "paper",
            [*cli, "evaluate", "--config", str(project / "configs/autodl/paper.yaml"), "--resume"],
        )
    )
    if with_training:
        for name, config, folder in (
            ("training_pilot", "training_pilot.yaml", "training_pilot"),
            ("training", "training.yaml", "training_fp32"),
        ):
            command = [*cli, "train", "--config", str(project / "configs/autodl" / config)]
            if (Path(paths["output_root"]) / folder / "training_manifest.json").exists():
                command.append("--resume")
            plan.append((name, command))
            if name == "training_pilot":
                plan.append(
                    (
                        "training_estimate",
                        [
                            py,
                            str(project / "scripts/run_tools.py"),
                            "estimate-training",
                            "--pilot",
                            str(Path(paths["output_root"]) / folder),
                            "--config",
                            str(project / "configs/autodl/training.yaml"),
                            "--output",
                            str(prep / "training-eta.json"),
                        ],
                    )
                )
        plan.append(
            (
                "paper_trained",
                [
                    *cli,
                    "evaluate",
                    "--config",
                    str(project / "configs/autodl/paper.yaml"),
                    "--checkpoint",
                    str(Path(paths["output_root"]) / "training_fp32/hub"),
                    "--output",
                    str(Path(paths["output_root"]) / "paper_trained"),
                    "--resume",
                ],
            )
        )
    return plan


def storage_check(paths, convert_needed):
    folder = Path(paths["polar_root"]) if convert_needed else Path(paths["output_root"])
    while not folder.exists():
        folder = folder.parent
    # Deliberately conservative floor for the unattended full converter, not a size prediction.
    required = (180 if convert_needed else 10) * 2**30
    free = shutil.disk_usage(folder).free
    if free < required:
        raise RuntimeError(
            f"Insufficient free space: {free / 2**30:.1f} GiB at {folder}; "
            f"unattended run requires {required / 2**30:.0f} GiB. Expand the data disk first."
        )
    return {"checked_path": str(folder), "free_gib": free / 2**30, "required_gib": required / 2**30}


def validate_config_paths(paths, with_training=False):
    validate_paths(paths)
    if ROOT.resolve() != Path(paths["code_root"]).resolve():
        raise ValueError(
            "code_root differs from this checkout; update the AutoDL configurations first"
        )
    for name in ("demo.yaml", "pilot.yaml", "paper.yaml"):
        cfg = load(ROOT / "configs/autodl" / name)
        if Path(cfg["data_root"]).resolve() != Path(paths["polar_root"]).resolve():
            raise ValueError(f"{name}: data_root differs from paths.yaml")
        if Path(cfg["output"]).resolve().parent != Path(paths["output_root"]).resolve():
            raise ValueError(f"{name}: output is not under the configured output_root")
    if with_training:
        for name in ("training.yaml", "training_pilot.yaml"):
            cfg = load(ROOT / "configs/autodl" / name)
            for split in ("train", "val"):
                if (
                    Path(cfg[f"{split}_folder"]).resolve()
                    != (Path(paths["polar_root"]) / split).resolve()
                ):
                    raise ValueError(f"{name}: {split}_folder differs from paths.yaml")
            if Path(cfg["output"]).resolve().parent != Path(paths["output_root"]).resolve():
                raise ValueError(f"{name}: output is not under the configured output_root")


def save_status(file, record):
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(file)


def run(paths, state_dir, shutdown, with_training=False):
    status_file = state_dir / "status.json"
    record = {
        "status": "running",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "initial_checks",
        "completed_stages": [],
        "shutdown_on_exit": shutdown,
        "with_training": with_training,
    }
    code = 1
    try:
        save_status(status_file, record)
        if shutdown and not Path("/usr/bin/shutdown").is_file():
            raise RuntimeError(
                "AutoDL shutdown command unavailable; cannot honor --shutdown-on-exit"
            )
        validate_config_paths(paths, with_training)
        polar = Path(paths["polar_root"])
        convert_needed = not polar.exists() or not any(polar.iterdir())
        record["storage"] = storage_check(paths, convert_needed)
        env = dict(
            os.environ,
            OMP_NUM_THREADS="8",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            TF_FORCE_GPU_ALLOW_GROWTH="true",
            PYTHONUNBUFFERED="1",
        )
        print("OMP_NUM_THREADS=8; existing CUDA libraries retained.", flush=True)
        print(
            f"Nonempty polar data is audited, never overwritten. Full training enabled: {with_training}",
            flush=True,
        )
        prep = Path(paths["output_root"]) / "preparation"
        prep.mkdir(parents=True, exist_ok=True)
        with open(prep / "requirements-resolved.txt", "w", encoding="utf-8") as stream:
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True, env=env
            )
        shutil.copytree(ROOT / "configs", prep / "configs", dirs_exist_ok=True)
        for name, command in steps(ROOT, paths, convert_needed, with_training):
            if name in ("demo", "paper", "training_pilot", "training", "paper_trained"):
                storage_check(paths, False)
            record["stage"] = name
            save_status(status_file, record)
            print(f"\n[{datetime.now(timezone.utc).isoformat()}] STAGE: {name}", flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            record["completed_stages"].append(name)
            save_status(status_file, record)
        record["stage"] = "export"
        save_status(status_file, record)
        # Keep the export outside outputs_casl to avoid including itself; shared data stays read-only.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = (
            Path(paths["output_root"]).parent / "casl_exports" / f"casl-results-{stamp}.tar.gz"
        )
        validate_paths(paths, archive)
        exported = export_results(paths["output_root"], archive)
        record.update(archive=exported["archive"], archive_sha256=exported["sha256"])
        if with_training:
            record["stage"] = "export_training"
            save_status(status_file, record)
            training_archive = archive.parent / f"casl-training-{stamp}.tar.gz"
            exported_training = export_results(
                Path(paths["output_root"]) / "training_fp32",
                training_archive,
                include_checkpoints=True,
            )
            record.update(
                training_archive=exported_training["archive"],
                training_archive_sha256=exported_training["sha256"],
            )
        record["completed_stages"].append("export")
        record["status"] = "completed"
        code = 0
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            save_status(status_file, record)
        except OSError:
            # A full disk must not prevent an explicitly requested power-off.
            traceback.print_exc()
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
        if shutdown:
            print("Explicit --shutdown-on-exit enabled: requesting AutoDL power-off.", flush=True)
            try:
                result = subprocess.run(["/usr/bin/shutdown"], timeout=30, check=False)
                if result.returncode:
                    print(
                        f"WARNING: shutdown returned {result.returncode}; check the console.",
                        flush=True,
                    )
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"WARNING: shutdown request failed: {error}; check the console.", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", action="store_true", help="Start detached from SSH, with a persistent log"
    )
    parser.add_argument(
        "--shutdown-on-exit",
        action="store_true",
        help="Power off this instance after success OR failure",
    )
    parser.add_argument(
        "--with-training",
        action="store_true",
        help="After official evaluation, run the training pilot, full configured training, and trained-model evaluation",
    )
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run only on the AutoDL Linux instance")
    import fcntl

    paths = load(ROOT / "configs/autodl/paths.yaml")
    validate_paths(paths)
    state_dir = Path(paths["output_root"]) / "overnight"
    state_dir.mkdir(parents=True, exist_ok=True)
    # Pass the locked open-file description to the detached child, leaving no launch race.
    lock = (
        os.fdopen(args.lock_fd, "a")
        if args.lock_fd is not None
        else open(state_dir / "runner.lock", "a")
    )
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A pipeline already holds the lock; do not start a second GPU workload")
        if args.start:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--lock-fd",
                str(lock.fileno()),
            ]
            if args.shutdown_on_exit:
                command.append("--shutdown-on-exit")
            if args.with_training:
                command.append("--with-training")
            with open(state_dir / "run.log", "a", buffering=1, encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(lock.fileno(),),
                )
            print(f"Started PID {process.pid}. Log: {state_dir / 'run.log'}")
            print(f"Status: {state_dir / 'status.json'}")
            print(f"Power off on success/failure: {args.shutdown_on_exit}")
            return 0
        return run(paths, state_dir, args.shutdown_on_exit, args.with_training)


if __name__ == "__main__":
    raise SystemExit(main())
