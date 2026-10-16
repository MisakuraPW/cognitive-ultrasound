"""Video-disjoint manifests, streaming HDF5 clips and explicit per-step sampling."""

import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import yaml
from torch.utils.data import Dataset


@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if cfg["selection"]["groups"] != [10, 4]:
        raise ValueError(
            "This reproduction batch locks10+4; use a separate protocol for new budgets"
        )
    if cfg["precision"] != "fp32":
        raise ValueError("First baseline is FP32; approximate precision needs a separate version")
    for k in ["width", "latent_channels"]:
        if type(cfg["model"][k]) is not int or cfg["model"][k] < 2:
            raise ValueError(k)
    for stage in ["codec", "prior", "filter"]:
        c = cfg["training"][stage]
        if any(type(c[k]) is not int or c[k] < 1 for k in ["epochs", "batch", "frames"]):
            raise ValueError("Invalid training capacity/budget")
    f = cfg["training"]["filter"]
    if not 0 <= f["burn_in"] < f["frames"] or f["bptt"] < 1:
        raise ValueError("Invalid warmup/BPTT")
    if cfg["training"]["prior"]["frames"] < 2:
        raise ValueError("Prior needs consecutive frames")
    if cfg["runtime"]["threads"] < 1 or cfg["runtime"]["workers"] < 0:
        raise ValueError("Invalid runtime")
    for k in ["checkpoint_updates", "validate_every", "patience"]:
        if cfg["training"][k] < 1:
            raise ValueError(k)
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("Invalid seed")
    for group, count in cfg["cohorts"].items():
        if count is None and group == "train":
            continue
        if type(count) is not int or count < 1:
            raise ValueError("Invalid cohort size")
    positive = [
        cfg["training"]["learning_rate"],
        cfg["training"]["gradient_clip"],
        cfg["model"]["delta_limit"],
        cfg["selection"]["redundancy_sigma"],
        cfg["selection"]["residual_sigma"],
        cfg["runtime"]["max_output_gib"],
    ]
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        raise ValueError("Expected finite positive hyperparameters")
    if any(not math.isfinite(v) or v < 0 for v in cfg["loss"].values()):
        raise ValueError("Invalid loss weight")
    if cfg["loss"]["rec"] <= 0 or cfg["loss"]["unc"] <= 0:
        raise ValueError("Reconstruction and uncertainty heads require nonzero supervision")
    if not 0 <= cfg["selection"]["value_decay"] < 1:
        raise ValueError("Invalid value EMA")
    if not cfg["runtime"]["calibrate_threads"] or any(
        type(v) is not int or v < 1 for v in cfg["runtime"]["calibrate_threads"]
    ):
        raise ValueError("Invalid thread candidates")
    if cfg["runtime"]["prefetch"] < 1 or cfg["evaluation"]["component_frames"] < 2:
        raise ValueError("Invalid prefetch/component frame limit")
    return cfg


def fingerprint(path):
    """Fast identity: size,mtime,head/tail; explicitly not a full-file checksum."""
    path = Path(path)
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(65536))
        f.seek(max(0, st.st_size - 65536))
        h.update(f.read(65536))
    return dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, head_tail_sha256=h.hexdigest())


def prepare(cfg):
    root = Path(cfg["data_root"])
    raw = yaml.safe_load(Path(cfg["split_manifest"]).read_text(encoding="utf-8-sig"))
    seen = set()
    for split in ["train", "val", "test"]:
        names = raw[split]
        if not names or len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("Invalid/overlapping CASL video split")
        if any(
            Path(n).name != n or "/" in n or "\\" in n or not n.endswith(".hdf5") for n in names
        ):
            raise ValueError("Split path must be a plain HDF5 filename")
        seen.update(names)
    original = {}
    with open(cfg["file_list"], encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = Path(row["FileName"]).stem
            if name in original:
                raise ValueError("Duplicate original patient")
            original[name] = dict(split=row["Split"].lower(), ef=float(row["EF"]))
            if not math.isfinite(original[name]["ef"]):
                raise ValueError("Nonfinite EF label")
    rng = np.random.default_rng(cfg["seed"])
    cohorts, files = {}, {}
    min_frames = max(x["frames"] for x in cfg["training"].values() if isinstance(x, dict))
    for group, split in [("train", "train"), ("development", "val"), ("test", "test")]:
        eligible = sorted(
            n for n in raw[split] if original.get(Path(n).stem, {}).get("split") == split
        )
        eligible = [eligible[i] for i in rng.permutation(len(eligible))]
        count = cfg["cohorts"][group]
        if count is not None and len(eligible) < count:
            raise ValueError(f"Insufficient {group} intersection: {len(eligible)} < {count}")
        selected = eligible if count is None else eligible[:count]
        if not selected:
            raise ValueError(f"Empty {group} cohort")
        cohorts[group] = selected
        for i, name in enumerate(selected):
            path = root / split / name
            with h5py.File(path) as h:
                ds = h["data/image"]
                if ds.ndim != 3 or ds.shape[1:] != (112, 112) or len(ds) < min_frames:
                    raise ValueError(f"Invalid/short preselected video: {path}")
                pixels = ds[0]
                frames = len(ds)
            normalize(pixels)
            files[name] = dict(
                split=split, frames=frames, ef=original[Path(name).stem]["ef"], **fingerprint(path)
            )
            if i % 100 == 0:
                print(
                    json.dumps(
                        dict(event="INDEX", cohort=group, indexed=i + 1, total=len(selected))
                    ),
                    flush=True,
                )
    return dict(
        cohorts=cohorts,
        files=files,
        root=str(root.resolve()),
        split_sha256=sha(cfg["split_manifest"]),
        file_list_sha256=sha(cfg["file_list"]),
        fingerprint_scope="size/mtime/head-tail hash, not full HDF5 checksum",
    )


def verify(manifest):
    for name, info in manifest["files"].items():
        value = fingerprint(Path(manifest["root"]) / info["split"] / name)
        if any(value[k] != info[k] for k in value):
            raise ValueError(f"Data identity changed: {name}")


def normalize(pixels):
    pixels = np.asarray(pixels, np.float32)
    if not np.isfinite(pixels).all() or pixels.min() < -60.001 or pixels.max() > 0.001:
        raise ValueError("Expected finite CASL polar B-mode[-60,0]dB; no silent normalization")
    return pixels / 30 + 1


def frames(manifest, name, start=0, length=None):
    info = manifest["files"][name]
    end = info["frames"] if length is None else start + length
    if start < 0 or end > info["frames"]:
        raise ValueError("Clip outside video")
    with h5py.File(Path(manifest["root"]) / info["split"] / name) as h:
        x = normalize(h["data/image"][start:end])
    return np.ascontiguousarray(x[:, None])


class Clips(Dataset):
    def __init__(self, manifest, names, length, seed, epoch):
        self.manifest, self.names, self.length = manifest, list(names), length
        self.seed, self.epoch = seed, epoch
        self.order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(
            len(names)
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        index = int(self.order[i])
        name = self.names[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 17]))
        start = int(rng.integers(self.manifest["files"][name]["frames"] - self.length + 1))
        return frames(self.manifest, name, start, self.length), start
