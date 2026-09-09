"""Bounded parallelism around the unchanged upstream H5Processor, with atomic outputs."""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import h5py
import numpy as np
import yaml

from .config import ZEA_COMMIT
from .data import read_splits


def atomic_json(file, value):
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(file)


@contextmanager
def conversion_lock(output):
    with open(output / ".conversion.lock", "a+b") as lock:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt

            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        yield


def verify_h5(file, accepted, frames=None):
    """Read every image chunk, including scan-converted images, to detect truncated writes."""
    with h5py.File(file, "r") as handle:
        keys = ["data/image_sc", "data/image"] if accepted else ["data/image_sc"]
        n = None
        for key in keys:
            data = handle[key]
            if data.ndim != 3 or data.shape[1:] != (112, 112) or len(data) < 1:
                raise ValueError(f"Invalid shape for {key}: {data.shape}")
            if n is not None and len(data) != n:
                raise ValueError("Polar and scan-converted frame counts differ")
            n = len(data)
            if frames is not None and n != frames:
                raise ValueError(f"Frame count differs from source AVI: {n} != {frames}")
            for start in range(0, n, 64):
                block = data[start : start + 64]
                if not np.isfinite(block).all() or block.min() < -60.001 or block.max() > 0.001:
                    raise ValueError(f"Invalid pixels in {key}")
        if not accepted and "data/image" in handle:
            raise ValueError("Rejected file unexpectedly contains polar images")
    return n


def init_worker(splits):
    # Spawn avoids forking initialized ML runtimes. Import only once in each worker.
    from .official import activate

    activate("jax")
    global PROCESSOR, SPLITS
    from zea.data.convert.echonet import H5Processor

    PROCESSOR, SPLITS = H5Processor, splits


def convert_one(source, output, split, identity_digest):
    source, output = Path(source), Path(output)
    started = time.monotonic()
    stat = source.stat()
    # Upstream writes only in a private directory. Publish after it closes and validation passes.
    with TemporaryDirectory(prefix=f"{source.stem}-", dir=output / ".conversion-tmp") as work:
        PROCESSOR(path_out_h5=work, splits=SPLITS)(source)
        file = Path(work) / split / f"{source.stem}.hdf5"
        frames = verify_h5(file, split != "rejected")
        with h5py.File(file, "a") as handle:
            handle.attrs["casl_conversion_identity"] = identity_digest
            handle.attrs["casl_source_size"] = stat.st_size
            handle.attrs["casl_source_mtime_ns"] = stat.st_mtime_ns
        current = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise ValueError(f"Source changed during conversion: {source}")
        destination = output / split / file.name
        if destination.exists():
            raise FileExistsError(f"Refusing to replace a published output: {destination}")
        file.replace(destination)
    return {
        "file": destination.name,
        "split": split,
        "frames": frames,
        "seconds": time.monotonic() - started,
    }


def run(raw, output, manifest, workers=8, resume=False):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Conversion output overlaps raw data")
    splits = read_splits(manifest)
    assigned = {name: split for split, names in splits.items() for name in names}
    sources = sorted((raw / "Videos").glob("*.avi"))
    if not sources:
        raise ValueError("No AVI sources found")
    inventory = {
        f"{p.stem}.hdf5": [p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in sources
    }
    if set(assigned) - inventory.keys():
        raise ValueError("Source AVI files are missing from the requested split")
    identity = {"zea_commit": ZEA_COMMIT, "raw": str(raw), "splits": splits, "sources": inventory}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    workers = min(
        workers,
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1),
    )
    if workers < 1 or workers > 32:
        raise ValueError("Conversion workers must be between 1 and 32")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Use --resume to inspect existing conversions")
    output.mkdir(parents=True, exist_ok=True)
    # Reject symlinked output entries before any quarantine, writes or temporary cleanup.
    if any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("Symlinks are not allowed inside the conversion destination")
    with conversion_lock(output):
        return run_locked(output, sources, assigned, identity, digest, splits, workers)


def run_locked(output, sources, assigned, identity, digest, splits, workers):
    metadata = output / "conversion_manifest.json"
    legacy = not metadata.exists()
    if not legacy and json.loads(metadata.read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError(
            "Conversion source, split or upstream revision changed; refusing mixed data"
        )
    for folder in ("train", "val", "test", "rejected", ".conversion-tmp"):
        (output / folder).mkdir(exist_ok=True)
    files = [
        p for split in ("train", "val", "test", "rejected") for p in (output / split).glob("*.hdf5")
    ]
    for file in files:
        if file.name not in identity["sources"] or file.parent.name != assigned.get(
            file.name, "rejected"
        ):
            raise ValueError(f"Unexpected file or split: {file}")
    # The old serial writer had no atomic publication. Redo its latest file even
    # if an interrupted HDF5 write happens to look structurally readable.
    last_legacy = max(files, key=lambda p: p.stat().st_mtime_ns) if legacy and files else None
    quarantine = output / ".conversion-quarantine" / uuid4().hex
    complete, repaired = set(), []
    print(
        f"Validating {len(files)} existing files; workers={workers}, CPU threads per worker=1",
        flush=True,
    )
    for index, file in enumerate(files, 1):
        try:
            if file == last_legacy:
                raise ValueError("Rechecking the last file from the interrupted serial converter")
            with h5py.File(file, "r") as handle:
                recorded = handle.attrs.get("casl_conversion_identity")
                if recorded is not None and recorded != digest:
                    raise RuntimeError(f"Foreign conversion identity: {file}")
            frames = None
            if recorded is None:
                # Only legacy files lack an atomic-completion identity. Check against real AVI count.
                from imageio_ffmpeg import count_frames_and_secs

                avi = Path(identity["raw"]) / "Videos" / identity["sources"][file.name][0]
                frames, _ = count_frames_and_secs(str(avi))
            verify_h5(file, file.parent.name != "rejected", frames)
        except (OSError, ValueError, KeyError) as error:
            destination = quarantine / file.parent.name / file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            file.replace(destination)
            repaired.append(file.name)
            print(f"REBUILD {file.name}: {error}; saved old file to {destination}", flush=True)
        else:
            complete.add(file.name)
        if index % 100 == 0:
            print(f"Validated {index}/{len(files)} existing files", flush=True)
    pending = [p for p in sources if f"{p.stem}.hdf5" not in complete]
    record = {
        "identity": identity,
        "status": "running",
        "workers": workers,
        "reused": len(complete),
        "rebuilt": repaired,
    }
    atomic_json(metadata, record)
    progress_file = output / "conversion_progress.json"
    started = time.monotonic()
    finished = 0
    print(
        f"Reusing {len(complete)} files; {len(pending)} pending; starting {workers} processes",
        flush=True,
    )

    def progress():
        elapsed = time.monotonic() - started
        atomic_json(
            progress_file,
            {
                "total": len(sources),
                "reused": len(complete),
                "converted_this_run": finished,
                "remaining": len(pending) - finished,
                "workers": workers,
                "elapsed_seconds": elapsed,
                "estimated_remaining_hours": ((len(pending) - finished) * elapsed / finished / 3600)
                if finished
                else None,
            },
        )

    progress()
    try:
        if pending:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=init_worker,
                initargs=(splits,),
            ) as pool:
                iterator = iter(pending)
                active = set()

                def submit_next():
                    source = next(iterator, None)
                    if source is not None:
                        active.add(
                            pool.submit(
                                convert_one,
                                str(source),
                                str(output),
                                assigned.get(f"{source.stem}.hdf5", "rejected"),
                                digest,
                            )
                        )

                for _ in range(workers):
                    submit_next()
                while active:
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = future.result()  # Propagate failures; never silently skip a video.
                        finished += 1
                        progress()
                        print(
                            f"[{len(complete) + finished}/{len(sources)}] {item['split']}/{item['file']} "
                            f"frames={item['frames']} worker_seconds={item['seconds']:.1f}",
                            flush=True,
                        )
                    for _ in done:
                        submit_next()
        produced = {
            split: sorted(p.name for p in (output / split).glob("*.hdf5"))
            for split in ("train", "val", "test", "rejected")
        }
        if any(produced[split] != splits[split] for split in splits):
            raise ValueError("Converted output does not match requested train/val/test split")
        temporary = output / "split.yaml.tmp"
        temporary.write_text(yaml.safe_dump(produced), encoding="utf-8")
        temporary.replace(output / "split.yaml")
        record["status"] = "completed"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["converted_this_run"] = finished
        atomic_json(metadata, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("raw", "output", "manifest"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.raw, args.output, args.manifest, args.workers, args.resume)


if __name__ == "__main__":
    main()
