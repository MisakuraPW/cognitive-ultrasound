"""Read-only AutoDL input inventory. Does not create data, environments or experiments."""

import csv
import platform
import shutil
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .config import load, path
from .provenance import command, sha256, write_json


def overlaps(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a in b.parents or b in a.parents


def protected_paths(cfg):
    return [
        cfg["raw_root"],
        *cfg["legacy_cache_roots"],
        *cfg["protected_project_roots"],
        *cfg.get("protected_output_roots", []),
    ]


def validate_paths(cfg, report=None):
    destinations = [cfg[k] for k in ("code_root", "polar_root", "output_root", "backup_root")]
    if report is not None:
        destinations.append(str(report))
    for destination in destinations:
        for source in protected_paths(cfg):
            if overlaps(destination, source):
                raise ValueError(
                    f"CASL destination overlaps a protected source: {destination} / {source}"
                )
    return cfg


def video_id(name):
    # CSV filenames are identifiers, never paths. Preserve case and leading zeros.
    if not isinstance(name, str) or not name or any(c in name for c in ("/", "\\", ":")):
        raise ValueError(f"Invalid video ID: {name!r}")
    stem = Path(name).stem
    if stem in ("", ".", ".."):
        raise ValueError(f"Invalid video ID: {name!r}")
    return stem


def inspect_inputs(cfg):
    """Only read paths/CSV/NPY headers. Ready means ready to attempt conversion, not decoded."""
    validate_paths(cfg)
    raw = Path(cfg["raw_root"])
    manifest = path(cfg["split_manifest"])
    import yaml

    splits = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    required = {}
    seen = set()
    for split in ("train", "val", "test"):
        required[split] = [video_id(v) for v in splits[split]]
        if len(set(required[split])) != len(required[split]) or seen.intersection(required[split]):
            raise ValueError("Duplicate CASL video IDs across/within split lists")
        seen.update(required[split])
    problems = []
    rows = []
    csv_file = raw / "FileList.csv"
    if csv_file.is_file():
        with open(csv_file, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"FileName", "Split"}.issubset(reader.fieldnames or []):
                problems.append("FileList.csv missing FileName/Split columns")
            else:
                rows = list(reader)
    else:
        problems.append("FileList.csv missing; verify shared mount and raw data location")
    ids, original = [], {}
    for row in rows:
        try:
            name = video_id(row["FileName"])
        except ValueError as error:
            problems.append(str(error))
            continue
        ids.append(name)
        original[name] = (row["Split"] or "").strip().upper()
        if original[name] not in ("TRAIN", "VAL", "TEST"):
            problems.append(f"Invalid original Split for {name}: {original[name]!r}")
    duplicates = sorted(k for k, n in Counter(ids).items() if n > 1)
    if duplicates:
        problems.append(f"Duplicate FileList video IDs: {len(duplicates)}")
    if not (raw / "Videos").is_dir():
        problems.append("Videos directory missing; old NPY cache is not a CASL polar dataset")
    crosswalk = {}
    missing_avi, missing_csv, empty_avi = {}, {}, []
    for split, names in required.items():
        missing_avi[split] = [n for n in names if not (raw / "Videos" / f"{n}.avi").is_file()]
        missing_csv[split] = [n for n in names if n not in original]
        for name in names:
            p = raw / "Videos" / f"{name}.avi"
            if p.is_file() and p.stat().st_size == 0:
                empty_avi.append(name)
        crosswalk[split] = dict(Counter(original.get(n, "MISSING") for n in names))
    if any(missing_avi.values()) or any(missing_csv.values()) or empty_avi:
        problems.append(
            "CASL required inputs are missing or empty; do not start conversion/training"
        )
    caches = []
    for folder in cfg["legacy_cache_roots"]:
        folder = Path(folder)
        files = sorted((folder / "npy").glob("*.npy"))
        entry = {
            "path": str(folder),
            "exists": folder.is_dir(),
            "npy_files": len(files),
            "role": "read-only inventory; NOT used as CASL input",
            "samples": [],
        }
        # Small header/value inspection; never treat this as a complete cache certification.
        for file in files[:3]:
            try:
                import numpy as np

                array = np.load(file, mmap_mode="r", allow_pickle=False)
                shape = list(array.shape)
                layout = "THW" if array.ndim == 3 else "THWC" if array.ndim == 4 else "unknown"
                valid = (
                    array.dtype == np.uint8
                    and len(shape) in (3, 4)
                    and shape[0] > 0
                    and shape[1:3] == [112, 112]
                )
                if len(shape) == 4:
                    valid &= shape[-1] == 3
                entry["samples"].append(
                    {
                        "file": file.name,
                        "shape": shape,
                        "dtype": str(array.dtype),
                        "layout": layout,
                        "header_valid": bool(valid),
                        "color_space": "not certified by shape/filename",
                    }
                )
            except Exception as error:
                entry["samples"].append({"file": file.name, "error": str(error)})
        caches.append(entry)
    return {
        "ready_for_conversion_inventory": not problems,
        "problems": problems,
        "raw_root": str(raw),
        "original_split_counts": dict(Counter(original.values())),
        "duplicate_csv_ids": duplicates,
        "casl_split_counts": {k: len(v) for k, v in required.items()},
        "casl_vs_original_split_crosswalk": crosswalk,
        "missing_avi": missing_avi,
        "missing_csv": missing_csv,
        "empty_avi": empty_avi,
        "tracings_present": (raw / "VolumeTracings.csv").is_file(),
        "split_manifest_sha256": sha256(manifest),
        "filelist_sha256": sha256(csv_file) if csv_file.is_file() else None,
        "legacy_caches": caches,
        "limitations": [
            "AVI presence/size only; decoding and geometry filtering happen during official conversion",
            "No independent patient identity audit; split comparison uses video IDs",
            "Original EchoNet Split is recorded, not substituted for the CASL split",
        ],
    }


def preflight(config_file, output, include_system=True):
    cfg = load(config_file)
    output = Path(output).resolve()
    validate_paths(cfg, output)
    result = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "path_conventions": cfg,
        "inputs": inspect_inputs(cfg),
    }
    if include_system:
        result["system"] = {
            "nvidia_smi": command(["nvidia-smi"]),
            "memory": command(["free", "-h"]),
            "disk": command(["df", "-h", "/", "/root/autodl-tmp", "/root/autodl-fs"]),
            "inodes": command(["df", "-i", "/root/autodl-tmp", "/root/autodl-fs"]),
            "mounts": command(["findmnt", "-T", str(Path(cfg["raw_root"]))]),
        }
        disks = {}
        for key in ("polar_root", "output_root", "backup_root"):
            directory = Path(cfg[key])
            while not directory.exists() and directory != directory.parent:
                directory = directory.parent
            usage = shutil.disk_usage(directory)
            disks[key] = {"checked_existing_parent": str(directory), "free_bytes": usage.free}
        result["disk_capacity"] = disks
    write_json(output, result)
    return result
