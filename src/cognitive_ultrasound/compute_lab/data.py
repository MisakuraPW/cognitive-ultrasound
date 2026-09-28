"""Bounded ordered I/O and immutable cohort lock. Never selects on model outcomes."""

import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from ..data import read_splits
from ..preparation.common import atomic_json, digest, read_json
from ..provenance import sha256


def historical_usage(roots):
    cases, sources = set(), []
    # Include tasks, case lists, manifests and reports; exclude binary frame caches.
    pattern = re.compile(r"0X[0-9A-Fa-f]+(?:\.hdf5)?")
    for root in map(Path, roots):
        if not root.exists():
            raise FileNotFoundError(
                f"Historical results must be present before cohort lock: {root}"
            )
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix in (".json", ".jsonl", ".csv", ".yaml", ".md"):
                found = {
                    x.removesuffix(".hdf5") + ".hdf5"
                    for x in pattern.findall(f.read_text(encoding="utf-8-sig", errors="replace"))
                }
                if found:
                    cases.update(found)
                    sources.append(dict(path=str(f), sha256=sha256(f), cases=sorted(found)))
    if not sources:
        raise ValueError("No historical case evidence found; refusing an unaudited fresh split")
    return dict(cases=sorted(cases), sources=sources)


def lock_manifest(cfg, output):
    file = output / "manifest.json"
    if file.exists():
        manifest = read_json(file)
        if manifest["split_sha256"] != sha256(Path(cfg["split_manifest"])):
            raise ValueError("Official split changed")
        for name, record in manifest["files"].items():
            f = Path(cfg["data_root"]) / "val" / name
            if f.stat().st_size != record["bytes"] or sha256(f) != record["sha256"]:
                raise ValueError(f"Locked input changed: {name}")
        return manifest
    audit = historical_usage(cfg["history_roots"])
    splits = read_splits(Path(cfg["split_manifest"]))
    names = sorted(set(splits["val"]) - set(audit["cases"]))
    names = np.random.default_rng(cfg["cohort_seed"]).permutation(names).tolist()
    if len(names) < 42:
        raise ValueError("Need 42 previously unused validation cases")
    groups = dict(debug=names[:2], development=names[2:10], confirmation=names[10:42])
    # Select before inspecting duration or quality. A missing/invalid video blocks;
    # it is never silently replaced by a more convenient patient.
    files = {}
    for name in names[:42]:
        f = Path(cfg["data_root"]) / "val" / name
        with h5py.File(f) as h5:
            shape = h5["data/image"].shape
            if shape[1:] != (112, 112) or shape[0] < 3:
                raise ValueError(f"Invalid locked video {name}: {shape}")
        files[name] = dict(frames=shape[0], bytes=f.stat().st_size, sha256=sha256(f))
    manifest = dict(
        cohorts=groups,
        files=files,
        history=audit,
        split_sha256=sha256(Path(cfg["split_manifest"])),
        seeds=dict(debug=[42], development=[42, 31415], confirmation=[42, 31415, 271828]),
        budgets=dict(debug=[14], development=[14], confirmation=[7, 14, 28]),
        dev_frames=128,
        confirmation_frames="all",
        locked_before_results=True,
    )
    manifest["identity"] = digest(manifest)
    atomic_json(file, manifest)
    return manifest


def read_one(file, index):
    with h5py.File(file) as h5:
        x = h5["data/image"][index].astype(np.float32)
    if x.shape != (112, 112) or not np.isfinite(x).all() or x.min() < -60.001 or x.max() > 0.001:
        raise ValueError(f"Invalid polar frame {file}:{index}")
    return (x / 30 + 1)[..., None]


def frames(file, count, mode="serial", depth=4):
    if mode == "cache":
        with h5py.File(file) as h5:
            x = h5["data/image"][:count].astype(np.float32)
        if not np.isfinite(x).all() or x.min() < -60.001 or x.max() > 0.001:
            raise ValueError("Invalid cached polar data")
        yield from (x / 30 + 1)[..., None]
    elif mode == "serial":
        for i in range(count):
            yield read_one(file, i)
    elif mode == "prefetch":
        # Bounded queue; workers own file handles; consume in input order.
        with ThreadPoolExecutor(max_workers=min(depth, 4)) as pool:
            pending = {}
            for i in range(min(depth, count)):
                pending[i] = pool.submit(read_one, file, i)
            for i in range(count):
                value = pending.pop(i).result()
                if i + depth < count:
                    pending[i + depth] = pool.submit(read_one, file, i + depth)
                yield value
    else:
        raise ValueError(mode)


class Writer:
    """Single ordered writer with bounded outstanding work; copies mutable buffers."""

    def __init__(self, file, asynchronous=False):
        self.h5 = h5py.File(file, "w")
        # HDF5 sec2 opens inheritable descriptors. Torch's C++ addr2line helper
        # uses exec without Python's close_fds default and would retain our lock.
        if os.name == "posix" and self.h5.driver == "sec2":
            os.set_inheritable(self.h5.id.get_vfd_handle(), False)
        self.pool = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.pending = []
        self.index = 0

    def append(self, arrays):
        copied = {k: np.array(v, copy=True) for k, v in arrays.items()}
        if any(not np.isfinite(v).all() for v in copied.values()):
            raise FloatingPointError("Refusing nonfinite output cache")
        i = self.index
        self.index += 1

        def save():
            group = self.h5.create_group(str(i))
            for k, v in copied.items():
                group.create_dataset(k, data=v, compression="lzf" if v.ndim else None)

        if self.pool:
            self.pending.append(self.pool.submit(save))
            if len(self.pending) >= 3:
                self.pending.pop(0).result()
        else:
            save()

    def close(self):
        try:
            for task in self.pending:
                task.result()
        finally:
            if self.pool:
                self.pool.shutdown(wait=True)
            self.h5.close()


def throughput(cfg, manifest, output):
    records, expected = [], {}
    for mode, depth in (
        ("serial", 1),
        ("prefetch", 1),
        ("prefetch", 2),
        ("prefetch", 4),
        ("cache", 1),
    ):
        for repeat in range(2):
            tick, count = time.perf_counter(), 0
            for name in manifest["cohorts"]["debug"]:
                n = min(128, manifest["files"][name]["frames"])
                hasher = hashlib.sha256()
                for i, x in enumerate(
                    frames(Path(cfg["data_root"]) / "val" / name, n, mode, depth)
                ):
                    hasher.update(i.to_bytes(8, "little"))
                    hasher.update(x.tobytes())
                    count += 1
                key = hasher.hexdigest()
                if name in expected and expected[name] != key:
                    raise ValueError("I/O candidate changed input content/order")
                expected[name] = key
            seconds = time.perf_counter() - tick
            records.append(
                dict(
                    mode=mode,
                    depth=depth,
                    repetition=repeat,
                    seconds=seconds,
                    frames=count,
                    passed=True,
                )
            )
    result = dict(
        records=records,
        hashes=expected,
        cache_caveat="First application read vs repeat; OS page cache not forcibly evicted.",
        recommendation=min(records, key=lambda r: r["seconds"]),
        scientific_configuration_changed=False,
    )
    atomic_json(output / "throughput.json", result)
    return result
