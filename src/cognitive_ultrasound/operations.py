"""Measured runtime projections and portable result exports; no cloud connection."""

import json
import statistics
import tarfile
from pathlib import Path

import yaml

from .config import load, path
from .provenance import sha256, write_json


def estimate_evaluation(pilot, config):
    pilot = Path(pilot)
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["identity"]["config"]
    target = load(config)
    # A speed change, missing endpoint, or different model invalidates the projection.
    for key in (
        "backend",
        "precision",
        "checkpoint",
        "particles",
        "temporal_window",
        "num_steps",
        "initial_step",
        "omega",
        "metrics",
        "segmentation",
        "profile",
        "save_trajectory",
        "visualize_every",
        "frames",
    ):
        if source[key] != target[key]:
            raise ValueError(f"Pilot/target differ in {key}; measure the matching configuration")
    if source.get("synthetic_input"):
        raise ValueError("Synthetic pilot cannot estimate a real-data run")
    splits = yaml.safe_load(path(target["split_manifest"]).read_text(encoding="utf-8"))
    count = len(splits[target["split"]])
    if target.get("limit_cases") is not None:
        count = min(count, target["limit_cases"])
    groups = []
    for method in target["methods"]:
        for budget in target["budgets"]:
            records = []
            for name in manifest["identity"]["cases"]:
                file = pilot / method / f"lines_{budget:03d}" / Path(name).stem / "complete.json"
                if file.exists():
                    record = json.loads(file.read_text(encoding="utf-8"))
                    if record.get("frames", 0) > 0 and "case_wall_s" in record:
                        records.append(record)
            if not records:
                raise ValueError(f"No measured completed pilot cases for {method}/{budget}")
            seconds_per_frame = sum(r["case_wall_s"] for r in records) / sum(
                r["frames"] for r in records
            )
            warmup = statistics.mean(r["warmup_compile_s"] for r in records)
            groups.append(
                {
                    "method": method,
                    "budget": budget,
                    "pilot_cases": len(records),
                    "wall_s_per_frame": seconds_per_frame,
                    "projected_hours": count
                    * (target["frames"] * seconds_per_frame + warmup)
                    / 3600,
                }
            )
    return {
        "kind": "MEASURED_PILOT_PROJECTION_NOT_A_GUARANTEE",
        "target_cases": count,
        "target_frames_per_case": target["frames"],
        "groups": groups,
        "projected_hours": sum(g["projected_hours"] for g in groups),
        "limitations": [
            "Requires the same hardware/environment; verify this manually",
            "Includes per-frame metrics/export and extrapolated warmup; excludes initial full-data audit/hash/model loading",
            "Pilot cases may not represent test throughput; short test videos use fewer than requested frames",
            "Small-pilot compilation overhead is extrapolated conservatively; not a confidence interval",
        ],
    }


def estimate_training(pilot, config):
    pilot = Path(pilot)
    target = load(config)
    manifest = json.loads((pilot / "training_manifest.json").read_text(encoding="utf-8"))
    source = manifest["identity"]["config"]
    if manifest["identity"]["synthetic_smoke"]:
        raise ValueError("Synthetic training cannot estimate real training throughput")
    for key in (
        "batch_size",
        "temporal_window",
        "precision",
        "image_size",
        "network_kwargs",
        "train_folder",
        "val_folder",
        "validation_steps",
    ):
        if source[key] != target[key]:
            raise ValueError(f"Training pilot/target differ in {key}")
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((pilot / "timing").glob("epoch_*.json"))
    ]
    # Discard the first epoch if possible, then the first 5 batches of each measured epoch.
    measured = records[1:] if len(records) > 1 else records
    batches = [t for r in measured for t in r["batch_wall_s"][5:]]
    if not batches:
        raise ValueError("Need at least 6 real training batches with timing records")
    step = statistics.mean(batches)
    overhead = statistics.mean(max(0, r["epoch_wall_s"] - sum(r["batch_wall_s"])) for r in measured)
    updates = target["epochs"] * target["steps_per_epoch"]
    return {
        "kind": "MEASURED_TRAINING_PROJECTION_NOT_A_GUARANTEE",
        "updates": updates,
        "batch_size": target["batch_size"],
        "measured_step_s": step,
        "estimated_epoch_overhead_s": overhead,
        "projected_hours": (updates * step + target["epochs"] * overhead) / 3600,
        "limitations": [
            "Same GPU/environment required; pilot sampling may not represent all data",
            "Excludes initial input validation/model setup and interruptions",
            "500 epochs x 10000 steps is a configured budget, not a verified paper training budget",
        ],
    }


def export_results(source, archive, include_checkpoints=False, include_trajectories=False):
    source, archive = Path(source).resolve(), Path(archive).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if archive == source or source in archive.parents:
        raise ValueError("Place the archive outside the source directory")
    if any(
        p.exists()
        for p in (
            archive,
            archive.with_suffix(archive.suffix + ".sha256"),
            archive.with_suffix(archive.suffix + ".json"),
        )
    ):
        raise FileExistsError(
            "Choose a new archive name; exports never overwrite an earlier archive"
        )
    files, excluded = [], []
    for file in sorted(source.rglob("*")):
        relative = file.relative_to(source)
        if file.is_symlink():
            excluded.append(str(relative))
            continue
        if not file.is_file():
            continue
        # Do not follow links or junctions into external data.
        if source not in file.resolve().parents:
            raise ValueError(f"Source file resolves outside results: {file}")
        checkpoint = bool({"hub", "resume"}.intersection(relative.parts)) or file.name.endswith(
            ".weights.h5"
        )
        trajectory = file.name == "state.npz"
        if (checkpoint and not include_checkpoints) or (trajectory and not include_trajectories):
            excluded.append(str(relative))
        else:
            files.append(file)
    if not files:
        raise ValueError("No result files selected")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz") as stream:
        for file in files:
            stream.add(
                file, arcname=str(Path(source.name) / file.relative_to(source)), recursive=False
            )
    digest = sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    result = {
        "archive": str(archive),
        "sha256": digest,
        "files": len(files),
        "excluded": excluded,
        "include_checkpoints": include_checkpoints,
        "include_trajectories": include_trajectories,
    }
    write_json(archive.with_suffix(archive.suffix + ".json"), result)
    return result
