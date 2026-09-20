"""Pure utilities shared by isolated JAX/TF workers and offline analyses."""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from ..config import ROOT, path
from ..data import read_splits
from ..provenance import sha256


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(file)


def atomic_npz(file, **arrays):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(file)


def read_json(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def read_frames(cfg, split, name, count, start=0):
    with h5py.File(Path(cfg["data_root"]) / split / name) as handle:
        data = handle["data/image"][start : start + count].astype("float32")
    if data.ndim != 3 or data.shape[1:] != (112, 112) or len(data) != count:
        raise ValueError(f"Invalid/short input {split}/{name}: {data.shape}")
    if not np.isfinite(data).all() or data.min() < -60.001 or data.max() > 0.001:
        raise ValueError(f"Expected polar [-60,0] dB: {name}")
    return (data / 30 + 1)[..., None]


def make_manifest(cfg):
    splits = read_splits(cfg["split_manifest"])
    if splits != read_splits(ROOT / "configs/splits/split.yaml"):
        raise ValueError("Preparation requires the pinned official patient splits")
    rng = np.random.default_rng(cfg["seed"])
    cohorts, files = {}, {}
    counts = cfg["cohorts"]
    for split, groups in [("val", ["debug", "development", "confirmation"]), ("train", ["train"])]:
        order = rng.permutation(splits[split]).tolist()
        if split == "val":
            excluded = set(cfg.get("excluded_validation_cases", []))
            order = [name for name in order if name not in excluded]
        offset = 0
        for group in groups:
            n = counts[group]
            if offset + n > len(order):
                raise ValueError("Requested more independent cases than available")
            cohorts[group] = order[offset : offset + n]
            offset += n
            for name in cohorts[group]:
                file = Path(cfg["data_root"]) / split / name
                with h5py.File(file) as h5:
                    ds = h5["data/image"]
                    if ds.ndim != 3 or ds.shape[1:] != (112, 112):
                        raise ValueError(f"Wrong dataset shape: {file}")
                    length = len(ds)
                    sample = ds[: min(length, 3)]
                    if (
                        not np.isfinite(sample).all()
                        or sample.min() < -60.001
                        or sample.max() > 0.001
                    ):
                        raise ValueError(f"Invalid sampled dB values: {file}")
                if length < cfg["frames"][group]:
                    raise ValueError(f"Preselected case too short; do not silently replace: {file}")
                st = file.stat()
                files[f"{split}/{name}"] = dict(
                    bytes=st.st_size, mtime_ns=st.st_mtime_ns, frames=length
                )
    return dict(
        cohorts=cohorts,
        files=files,
        split_sha256=sha256(cfg["split_manifest"]),
        note="Seeded video-level cohorts; frame zero cold start; no test data used",
    )


def verify_manifest(cfg, manifest):
    if sha256(cfg["split_manifest"]) != manifest["split_sha256"]:
        raise ValueError("Split manifest changed since this run")
    for relative, identity in manifest["files"].items():
        st = (Path(cfg["data_root"]) / relative).stat()
        if (st.st_size, st.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
            raise ValueError(f"Data changed since selection: {relative}")


def source_identity():
    paths = [
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("scripts/*.sh"),
        *ROOT.glob("configs/**/*.yaml"),
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def normalized_config(cfg):
    cfg = dict(cfg)
    for key in ("data_root", "split_manifest", "checkpoint"):
        cfg[key] = str(path(cfg[key]).resolve())
    for key, value in cfg["cohorts"].items():
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid cohort {key}")
    for key, value in cfg["frames"].items():
        if type(value) is not int or value < 3:
            raise ValueError(f"At least 3 frames needed for {key}")
    if not 1 <= cfg["budget"] <= 112 or not 0 < cfg["max_hours"] <= 24:
        raise ValueError("Invalid budget or wall-clock cap (maximum 24 h per run)")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]) or len(cfg["seeds"]) < 1:
        raise ValueError("Independent, nonempty seeds required")
    from .sampling import VARIANTS

    if set(cfg["candidates"]) - (set(VARIANTS) - {"reference", "wrapper", "profile"}):
        raise ValueError("Unknown acceleration candidate")
    if len(set(cfg["candidates"])) != len(cfg["candidates"]):
        raise ValueError("Duplicate acceleration candidates")
    if any(not 0 < v <= 24 for v in cfg["phase_hours"].values()) or not 0 < cfg["job_hours"] <= 24:
        raise ValueError("Invalid phase/job time budget")
    if cfg["min_free_gib"] < 1 or cfg["max_output_gib"] < 1:
        raise ValueError("Disk bounds must be positive")
    budgets = cfg["branches"]["budgets"]
    if (
        len(budgets) != 3
        or budgets != sorted(set(budgets))
        or not 1 <= min(budgets) <= max(budgets) < 112
    ):
        raise ValueError("Three ascending distinct budgets smaller than image width required")
    if (
        not cfg["branches"]["seeds"]
        or min(cfg["branches"]["cases"], cfg["branches"]["states_per_case"]) < 1
    ):
        raise ValueError("Empty branch study")
    if (
        cfg["risk"]["ridge"] <= 0
        or cfg["bf"]["chunk_steps"] < 1
        or any(v < 1 for v in cfg["bf"]["steps"].values())
    ):
        raise ValueError("Invalid training settings")
    return cfg


def metrics(target, prediction, mask):
    from ..evaluation.metrics import Metrics

    target, prediction = np.asarray(target), np.asarray(prediction)
    mask = np.broadcast_to(np.asarray(mask).astype(bool), target.shape)
    error = np.abs(target - prediction)
    scores = Metrics(["psnr", "ssim"])(target, prediction)
    # Perfect-match infinity is explicit rather than invalid JSON.
    scores["psnr_perfect"] = bool(np.isinf(scores["psnr"]))
    if scores["psnr_perfect"]:
        scores["psnr"] = 100.0  # reporting ceiling, flagged above
    return {
        **scores,
        "mae": float(error.mean()),
        "unobserved_mae": float(error[~mask].mean()) if (~mask).any() else None,
        "actual_lines": int(mask[0, :, 0].sum()),
    }


FEATURES = [
    "uncertainty_mean",
    "uncertainty_q90",
    "uncertainty_std",
    "observed_residual",
    "image_change",
    "budget_fraction",
]


def observable_features(uncertainty, prediction, previous, residual, actual_lines):
    u = np.asarray(uncertainty, dtype=float)
    return dict(
        uncertainty_mean=float(u.mean()),
        uncertainty_q90=float(np.quantile(u, 0.9)),
        uncertainty_std=float(u.std()),
        observed_residual=float(residual),
        image_change=float(np.abs(prediction - previous).mean()),
        budget_fraction=actual_lines / 112,
    )


def frame_files(directory):
    return sorted(Path(directory).glob("*/*/frame_*.npz"))


def rows_from(directory):
    rows = []
    for file in frame_files(directory):
        with np.load(file, allow_pickle=False) as archive:
            rows.append(json.loads(str(archive["row"])))
    return rows


def emit(message, **fields):
    print(json.dumps(clean(dict(event=message, **fields)), ensure_ascii=False), flush=True)
