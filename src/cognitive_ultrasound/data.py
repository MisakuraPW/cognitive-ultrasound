"""Use the official polar converter and official patient split; never resplit frames."""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

from .config import MODEL_REVISION, ROOT, SPLIT_REVISION
from .provenance import sha256, write_json

SPLIT_COUNTS = {"train": 6985, "val": 500, "test": 500}


def read_splits(file):
    raw = yaml.safe_load(Path(file).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected mapping train/val/test -> HDF5 filenames")
    result = {}
    seen = set()
    for split in SPLIT_COUNTS:
        names = raw[split]
        if not isinstance(names, list):
            raise ValueError(f"Invalid {split} list")
        if any(
            not isinstance(n, str)
            or Path(n).name != n
            or not n.endswith(".hdf5")
            or "/" in n
            or "\\" in n
            for n in names
        ):
            raise ValueError("Split entries must be plain .hdf5 filenames")
        if len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("Duplicate patient or patient leakage across splits")
        seen.update(names)
        result[split] = sorted(names)
    return result


def fetch_assets(checkpoint_dir, split_dir, weights=True):
    from huggingface_hub import hf_hub_download, snapshot_download

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    merged = {}
    for split in SPLIT_COUNTS:
        file = hf_hub_download(
            "zeahub/echonet-dynamic", f"{split}.yaml", repo_type="dataset", revision=SPLIT_REVISION
        )
        contents = yaml.safe_load(Path(file).read_text(encoding="utf-8"))
        merged[split] = contents["file_paths"] if isinstance(contents, dict) else contents
    file = split_dir / "split.yaml"
    file.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    splits = read_splits(file)
    if {k: len(v) for k, v in splits.items()} != SPLIT_COUNTS:
        raise ValueError("Official split counts differ from the paper")
    if weights:
        snapshot_download(
            "zeahub/ulsa",
            revision=MODEL_REVISION,
            local_dir=checkpoint_dir,
            allow_patterns=["config.json", "model.weights.h5"],
        )
        write_json(
            Path(checkpoint_dir) / "provenance.json",
            {
                "repo": "zeahub/ulsa",
                "revision": MODEL_REVISION,
                "files": {
                    p.name: sha256(p)
                    for p in Path(checkpoint_dir).glob("*")
                    if p.is_file() and p.name != "provenance.json"
                },
            },
        )


def convert(raw, output, manifest):
    from .official import activate

    activate("jax")
    raw, output, manifest = Path(raw).resolve(), Path(output).resolve(), Path(manifest).resolve()
    read_splits(manifest)
    if not (raw / "Videos").is_dir():
        raise FileNotFoundError(f"Expected licensed EchoNet data at {raw / 'Videos'}")
    if raw.name != "EchoNet-Dynamic":
        raise ValueError(
            "The upstream converter requires an extracted EchoNet-Dynamic/Videos layout; "
            "--raw must point to that EchoNet-Dynamic directory"
        )
    if manifest.name != "split.yaml":
        raise ValueError("The upstream converter requires a directory containing split.yaml")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh destination to avoid mixing conversions")
    # unzip(src, "echonet") appends EchoNet-Dynamic/Videos itself. Pass the
    # dataset's parent, not Videos or the dataset root. The check above ensures
    # the extracted folder exists, so upstream never extracts into shared storage.
    # Upstream --split_path is also a DIRECTORY, despite the README example.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "zea.data.convert",
            "echonet",
            str(raw.parent),
            str(output),
            "--split_path",
            str(manifest.parent),
            "--no_hyperthreading",
        ],
        check=True,
        cwd=ROOT,
    )


def inspect_file(file, min_frames=1):
    with h5py.File(file, "r") as handle:
        dataset = handle["data/image"]
        if dataset.ndim != 3 or dataset.shape[1:] != (112, 112):
            raise ValueError(f"{file}: expected (frames,112,112), got {dataset.shape}")
        if dataset.shape[0] < min_frames:
            raise ValueError(f"{file}: fewer than {min_frames} frames")
        low, high = float("inf"), float("-inf")
        for start in range(0, len(dataset), 64):
            block = dataset[start : start + 64]
            if not np.isfinite(block).all():
                raise ValueError(f"{file}: nonfinite pixels")
            low, high = min(low, float(block.min())), max(high, float(block.max()))
        if low < -60.001 or high > 0.001:
            raise ValueError(f"{file}: expected dB range [-60,0], got [{low},{high}]")
        return {"frames": len(dataset), "resolution": [112, 112], "min": low, "max": high}


def audit_dataset(root, manifest, report, require_complete=True):
    root = Path(root)
    splits = read_splits(manifest)
    statistics = {"manifest_sha256": sha256(manifest), "splits": {}, "complete": True}
    for split, names in splits.items():
        files = {p.name: p for p in (root / split).glob("*.hdf5")}
        missing = sorted(set(names) - files.keys())
        extra = sorted(files.keys() - set(names))
        if extra:
            raise ValueError(f"{split}: files outside pinned split: {extra[:5]}")
        details = {name: inspect_file(file) for name, file in sorted(files.items())}
        statistics["splits"][split] = {
            "expected": len(names),
            "videos": len(files),
            "frames": sum(v["frames"] for v in details.values()),
            "missing": missing,
            "files": details,
        }
        statistics["complete"] &= not missing and len(names) == SPLIT_COUNTS[split]
    report = Path(report)
    write_json(report.with_suffix(".json"), statistics)
    lines = [
        "# Dataset statistics",
        "",
        "Polar 112×112; official cubic conversion; data/image in [-60,0].",
        "No patient/temporal resplitting. Full sequences for training, first 100 frames for evaluation.",
        "",
        "| Split | Expected | Available | Frames |",
        "|---|---:|---:|---:|",
    ]
    for split, s in statistics["splits"].items():
        lines.append(f"| {split} | {s['expected']} | {s['videos']} | {s['frames']} |")
    lines += ["", f"Complete paper split: {statistics['complete']}"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if require_complete and not statistics["complete"]:
        raise ValueError(
            "Dataset incomplete; see dataset statistics. --allow-partial is for demos only"
        )
    return statistics
