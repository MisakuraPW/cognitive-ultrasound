"""Lock64 TRAIN,16 VAL,32 fresh TEST before pilot; no outcome-based sampling."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from ..config import path
from ..data import read_splits
from ..preparation.common import atomic_json, digest, read_json
from ..provenance import sha256
from .data import configuration
from .stage_protocol import ARMS, curriculum


def config(file):
    c = configuration(file)
    d = c["stage_budget"]
    d["source_batch"] = str(path(d["source_batch"]).resolve())
    d["historical_roots"] = [str(path(x).resolve()) for x in d["historical_roots"]]
    if d["arms"] != list(ARMS) or d["full_passes"] != 4 or d["pilot_passes"] != 2:
        raise ValueError("This version freezes T1/Q2/T2, two pilot passes and four full passes")
    if c["budgets"]["fixed"] != [10, 4] or c["task"]["stride"] != 1:
        raise ValueError("Common cold frame and all-start EF readout are fixed")
    if (
        not 1 <= d["pilot_train"] <= c["cohorts"]["train"]
        or not 1 <= d["pilot_dev"] <= c["cohorts"]["development"]
    ):
        raise ValueError("Pilot must be a subset of the full locked cohort")
    if any(
        d[k] <= 0
        for k in ["hidden", "history_frames", "dual_lr", "dual_max", "mse_floor", "rl_score_scale"]
    ):
        raise ValueError("Invalid stage learning parameters")
    if min(d["mse_margin"], d["p90_margin"]) < 0:
        raise ValueError("Invalid quality tolerance")
    if d["evaluation_seeds"] != [42, 31415]:
        raise ValueError("Evaluation seeds are frozen separately from the training seed")
    if not 1 <= d["checkpoint_frames"] < 64:
        raise ValueError("Checkpoint chunks must allow bounded64-frame recovery probe")
    c["value_diagnostics"] = dict(
        functional_fixture=d["functional_fixture"], checkpoint_frames=d["checkpoint_frames"]
    )
    return c


def historical_names(roots, output):
    seen = set()
    hashes = {}
    for base in map(Path, roots):
        if not base.exists():
            continue
        for p in sorted(base.rglob("manifest.json")):
            if p.resolve().is_relative_to(output.resolve()) or p.stat().st_size > 4 * 2**20:
                continue
            v = read_json(p)
            if "cohorts" in v:
                for names in v["cohorts"].values():
                    seen.update(names)
                hashes[str(p)] = sha256(p)
            elif "files" in v and isinstance(v["files"], dict):
                seen.update(v["files"])
                hashes[str(p)] = sha256(p)
    return seen, hashes


def lock_manifest(c, root):
    root = Path(root)
    p = root / "manifest.json"
    source = Path(c["stage_budget"]["source_batch"])
    if p.exists():
        m = read_json(p)
        payload = {k: v for k, v in m.items() if k != "identity"}
        if digest(payload) != m["identity"]:
            raise ValueError("Locked manifest payload corrupted")
        if sha256(source / "manifest.json") != m["source_manifest_sha256"]:
            raise ValueError("Inherited manifest provenance changed")
        for key, value in m["sources"].items():
            if sha256(c[key]) != value:
                raise ValueError("Locked labels/splits changed")
        for name, item in m["files"].items():
            if sha256(Path(c["data_root"]) / item["relative"]) != item["sha256"]:
                raise ValueError(f"Locked input changed: {name}")
        return m
    old = read_json(source / "manifest.json")
    sources = {k: sha256(c[k]) for k in ["file_list", "split_manifest"]}
    if old["sources"] != sources:
        raise ValueError("New cohort sources differ from repaired batch")
    with open(c["file_list"], encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = {Path(r["FileName"]).stem + ".hdf5": r for r in rows}
    if len(labels) != len(rows):
        raise ValueError("Duplicate label identifiers")
    splits = read_splits(Path(c["split_manifest"]))
    seen, historical = historical_names(c["stage_budget"]["historical_roots"], root)
    seen.update(old["files"])
    rng = np.random.default_rng(c["stage_budget"]["cohort_seed"])
    groups, files = {}, {}
    for group, task_split in [("train", "TRAIN"), ("development", "VAL"), ("confirmation", "TEST")]:
        permitted = (
            set(splits["train"]) if group == "train" else set(splits["val"]) | set(splits["test"])
        )
        inherited = list(old["cohorts"][group]) if group != "confirmation" else []
        wanted = c["cohorts"][group]
        if len(inherited) > wanted:
            raise ValueError("Requested cohort is smaller than the inherited research cohort")
        eligible = sorted(
            n
            for n, r in labels.items()
            if n in permitted
            and r["Split"].upper() == task_split
            and n not in inherited
            and (group == "train" or n not in seen)
        )
        pool = rng.permutation(eligible).tolist()
        # Deterministic EF-stratum round robin for the newly added cases, never by reconstruction outcome.
        bins = [
            [
                n
                for n in pool
                if (0 if float(labels[n]["EF"]) < 40 else 1 if float(labels[n]["EF"]) < 60 else 2)
                == i
            ]
            for i in range(3)
        ]
        ordered = []
        while any(bins):
            for bucket in bins:
                if bucket:
                    ordered.append(bucket.pop(0))
        selected = list(inherited)
        for name in [*inherited, *ordered]:
            if name not in permitted or labels[name]["Split"].upper() != task_split:
                raise ValueError("Inherited pretraining/task split is incompatible")
            split = next(s for s in ["train", "val", "test"] if name in splits[s])
            file = Path(c["data_root"]) / split / name
            if not file.exists():
                if name in inherited:
                    raise FileNotFoundError(file)
                continue
            with h5py.File(file) as h:
                shape = h["data/image"].shape
            minimum = 63 if group == "train" else 64
            if len(shape) != 3 or tuple(shape[1:]) != (112, 112) or shape[0] < minimum:
                if name in inherited:
                    raise ValueError(f"Inherited video too short/incompatible: {name}")
                continue
            if name not in inherited and len(selected) >= wanted:
                break
            ef = float(labels[name]["EF"])
            if not np.isfinite(ef) or not 0 <= ef <= 100:
                raise ValueError("Invalid EF label")
            files[name] = dict(
                relative=f"{split}/{name}",
                frames=shape[0],
                ef=ef,
                sha256=sha256(file),
                task_split=task_split,
            )
            if name not in inherited:
                selected.append(name)
        if len(selected) != wanted:
            raise ValueError(
                f"Insufficient fresh compatible {group} videos: {len(selected)}/{wanted}; no silent downsizing"
            )
        groups[group] = selected
    if any(
        set(groups[a]) & set(groups[b])
        for a, b in [
            ("train", "development"),
            ("train", "confirmation"),
            ("development", "confirmation"),
        ]
    ):
        raise AssertionError("Cohort leakage")
    stream, ends = curriculum(groups["train"], c)
    m = dict(
        cohorts=groups,
        files=files,
        sources=sources,
        curriculum=stream,
        pass_ends={str(k): v for k, v in ends.items()},
        pilot_train=list(
            dict.fromkeys(x["case"] for x in stream[: c["stage_budget"]["pilot_train"] * 2])
        ),
        pilot_development=groups["development"][: c["stage_budget"]["pilot_dev"]],
        source_manifest_sha256=sha256(source / "manifest.json"),
        historical_registry=historical,
        locked_before_results=True,
        fresh_confirmation=True,
        pretraining_split_assumption="Official asset provenance, not independently certified",
    )
    m["identity"] = digest(m)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(p, m)
    return m


def prepare(c, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "requested_config.json").exists() and read_json(root / "requested_config.json") != c:
        raise ValueError("Configuration changed; use a new batch directory")
    atomic_json(root / "requested_config.json", c)
    c = json.loads(json.dumps(c))
    if not c["stage_budget"]["functional_fixture"]:
        old = read_json(Path(c["stage_budget"]["source_batch"]) / "config.json")
        c["feature_scales"] = old["feature_scales"]
        prior = read_json(Path(c["stage_budget"]["source_batch"]) / "identity.json")
        for key in ["ef_weights", "ef_stats"]:
            if not Path(c[key]).exists():
                raise FileNotFoundError(
                    f"Required frozen asset missing: {c[key]}; no automatic download"
                )
            if sha256(c[key]) != prior[key]:
                raise ValueError(f"Frozen source asset changed: {key}")
        actual = {p.name: sha256(p) for p in Path(c["checkpoint"]).iterdir() if p.is_file()}
        if actual != prior["checkpoint"]:
            raise ValueError("Frozen CASL weights differ from the repaired batch")
    if (root / "config.json").exists() and read_json(root / "config.json") != c:
        raise ValueError("Frozen calibration changed; do not overwrite existing batch")
    atomic_json(root / "config.json", c)
    m = lock_manifest(c, root)
    from .suite import identity

    if c["stage_budget"]["functional_fixture"]:
        from ..config import ROOT

        current = dict(
            config=digest(c),
            manifest=m["identity"],
            functional_fixture=True,
            source={
                str(p.relative_to(ROOT)): sha256(p)
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        )
    else:
        current = identity(c, m)
    if (root / "identity.json").exists() and read_json(root / "identity.json") != current:
        raise ValueError(
            "Code/models/hardware changed; preserve old results and use a new directory"
        )
    atomic_json(root / "identity.json", current)
    return c, m
