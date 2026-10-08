"""Intersect diffusion/task training splits; lock cohorts before any model result."""

import csv
from pathlib import Path

import h5py
import numpy as np

from ..config import load, path
from ..data import read_splits
from ..preparation.common import atomic_json, digest, read_json
from ..provenance import sha256


def configuration(file):
    cfg = load(file)
    for key in (
        "data_root",
        "file_list",
        "raw_videos",
        "split_manifest",
        "checkpoint",
        "ef_weights",
        "ef_stats",
    ):
        cfg[key] = str(path(cfg[key]).resolve())
    b = cfg["budgets"]
    if len(b["fixed"]) != 2:
        raise ValueError("Two fixed stage budgets required")
    for key in ("first", "second"):
        if b[key] != sorted(set(b[key])) or any(type(k) is not int for k in b[key]):
            raise ValueError("Budget levels must be distinct ascending integers")
    if min(b["first"]) < 1 or min(b["second"]) != 0 or b["maximum"] > 112:
        raise ValueError("Invalid budget bounds")
    if max(b["first"]) > b["maximum"] or not set(b["fixed"][0:1]) <= set(b["first"]):
        raise ValueError("Invalid first-stage fixed action")
    if b["fixed"][1] not in b["second"] or sum(b["fixed"]) > b["maximum"]:
        raise ValueError("Invalid fixed budget")
    if cfg["perception"]["precision"] != "float32" or cfg["perception"]["particles"] != 2:
        raise ValueError("This batch explicitly supports frozen FP32 / two particles only")
    if not 1 <= cfg["perception"]["warm_steps"] <= 500:
        raise ValueError("Invalid warm-step count")
    if cfg["training"]["clip_frames"] < (cfg["task"]["frames"] - 1) * cfg["task"]["period"] + 1:
        raise ValueError("Training episode must cover the EF temporal window")
    if cfg["training"]["updates"] < 1 or min(cfg["cohorts"].values()) < 1:
        raise ValueError("Empty experiment")
    if cfg["task"]["frames"] != 32 or cfg["task"]["period"] != 2 or cfg["task"]["stride"] < 1:
        raise ValueError("Pinned EF checkpoint requires32 frames, period2")
    if cfg["task"]["score"] != "official_tbig_mean_gradient_squared":
        raise ValueError("This batch implements the declared official TBIG code convention only")
    if cfg["training"]["gs_gradient"] != "local_frame_vjp":
        raise ValueError("Unsupported GS estimator")
    if not 0 < cfg["training"]["temperature_end"] <= cfg["training"]["temperature_start"]:
        raise ValueError("Invalid GS temperatures")
    if (
        not 0 < cfg["training"]["learning_rate"]
        or not 0 <= cfg["training"]["rl_baseline_decay"] < 1
    ):
        raise ValueError("Invalid training parameters")
    if cfg["runtime"]["job_timeout_hours"] <= 0 or cfg["runtime"]["threads"] < 1:
        raise ValueError("Invalid runtime controls")
    pairs = cfg["budgets"]["fixed_sweep"]
    if any(
        len(p) != 2 or p[0] not in b["first"] or p[1] not in b["second"] or sum(p) > b["maximum"]
        for p in pairs
    ):
        raise ValueError("Invalid fixed comparison working point")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]) or not cfg["seeds"]:
        raise ValueError("Invalid seeds")
    if len(set(cfg["training"]["lambdas"])) != len(cfg["training"]["lambdas"]) or any(
        x < 0 for x in cfg["training"]["lambdas"]
    ):
        raise ValueError("Invalid finite cost-weight list")
    return cfg


def lock_manifest(cfg, root):
    destination = root / "manifest.json"
    splits = read_splits(Path(cfg["split_manifest"]))
    with open(cfg["file_list"], encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_name = {Path(r["FileName"]).stem + ".hdf5": r for r in rows}
    if len(by_name) != len(rows):
        raise ValueError("Duplicate video label identifiers")
    source_hashes = {k: sha256(cfg[k]) for k in ("file_list", "split_manifest")}
    if destination.exists():
        manifest = read_json(destination)
        if manifest["sources"] != source_hashes:
            raise ValueError("Label/split identity changed")
        for name, item in manifest["files"].items():
            if sha256(Path(cfg["data_root"]) / item["relative"]) != item["sha256"]:
                raise ValueError(f"Locked input changed: {name}")
        return manifest
    groups, files = {}, {}
    for group, task_split in (("train", "TRAIN"), ("development", "VAL"), ("confirmation", "TEST")):
        permitted = (
            set(splits["train"]) if group == "train" else (set(splits["val"]) | set(splits["test"]))
        )
        pool = sorted(
            n for n, r in by_name.items() if n in permitted and r["Split"].upper() == task_split
        )
        pool = np.random.default_rng(cfg["cohort_seed"]).permutation(pool).tolist()
        if len(pool) < cfg["cohorts"][group]:
            raise ValueError(f"Insufficient intersection cohort {group}: {len(pool)}")
        groups[group] = pool[: cfg["cohorts"][group]]
        for name in groups[group]:
            split = next(s for s in ("train", "val", "test") if name in splits[s])
            file = Path(cfg["data_root"]) / split / name
            with h5py.File(file) as h5:
                shape = h5["data/image"].shape
            if len(shape) != 3 or shape[1:] != (112, 112) or shape[0] < 3:
                raise ValueError(f"Invalid preselected input {file}")
            ef = float(by_name[name]["EF"])
            if not np.isfinite(ef) or not 0 <= ef <= 100:
                raise ValueError("Invalid EF percentage")
            files[name] = dict(
                relative=f"{split}/{name}",
                frames=shape[0],
                ef=ef,
                sha256=sha256(file),
                task_split=task_split,
            )
    manifest = dict(
        cohorts=groups,
        files=files,
        sources=source_hashes,
        locked_before_results=True,
        scope="New EF experiment; confirmation intersects both pretrained holdouts; "
        "pretrained EF split assumption recorded, not independently certified",
    )
    manifest["identity"] = digest(manifest)
    atomic_json(destination, manifest)
    return manifest


def read_episode(cfg, manifest, name, start=0, count=None):
    item = manifest["files"][name]
    count = item["frames"] - start if count is None else count
    with h5py.File(Path(cfg["data_root"]) / item["relative"]) as h5:
        images = h5["data/image"][start : start + count].astype(np.float32)
    if (
        len(images) != count
        or not np.isfinite(images).all()
        or images.min() < -60.001
        or (images.max() > 0.001)
    ):
        raise ValueError("Invalid full episode")
    return images / 30 + 1
