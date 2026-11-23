"""Validate dataset access and smoke outputs on the SEU compute nodes."""

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter


def cpu_check(share, nvme):
    from cognitive_ultrasound.config import MODEL_REVISION
    from cognitive_ultrasound.data import SPLIT_COUNTS, inspect_file, read_splits
    from cognitive_ultrasound.official import activate
    from cognitive_ultrasound.provenance import sha256

    activate("jax")
    weights = Path("checkpoints/official")
    record = json.loads((weights / "provenance.json").read_text(encoding="utf-8"))
    if record["revision"] != MODEL_REVISION:
        raise SystemExit("Checkpoint revision mismatch")
    for name in ("config.json", "model.weights.h5"):
        if sha256(weights / name) != record["files"][name]:
            raise SystemExit(f"Checkpoint checksum mismatch: {name}")
    print("Official checkpoint SHA-256 OK")

    splits = read_splits("configs/splits/split.yaml")
    for label, root in (("SHARE", share), ("NVMe", nvme)):
        for split, names in splits.items():
            actual = {p.name for p in (root / split).glob("*.hdf5")}
            expected = set(names)
            if len(expected) != SPLIT_COUNTS[split] or actual != expected:
                raise SystemExit(
                    f"{label} {split}: missing={len(expected - actual)}, "
                    f"extra={len(actual - expected)}, manifest={len(expected)}"
                )
            print(f"{label} {split}: {len(actual)} HDF5")

        start = perf_counter()
        samples = [
            inspect_file(root / split / name)
            for split, names in splits.items()
            for name in names[:4]
        ]
        frames = sum(sample["frames"] for sample in samples)
        print(f"{label}: {len(samples)} videos, {frames} frames, {perf_counter() - start:.2f} s")
    print("CPU_CHECK_OK")


def gpu_check(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    case, = manifest["identity"]["cases"]
    folder = root / "casl" / "lines_007" / Path(case).stem
    with (folder / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if manifest["status"] != "completed" or len(rows) != 5:
        raise SystemExit("Evaluation incomplete")
    for row in rows:
        if float(row["actual_lines"]) != 7:
            raise SystemExit("Unexpected acquisition budget")
        if not all(math.isfinite(float(row[key])) for key in ("psnr", "ssim", "total_s")):
            raise SystemExit("Non-finite evaluation output")
    print("GPU_SMOKE_OK", root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checks = parser.add_subparsers(dest="check", required=True)
    cpu = checks.add_parser("cpu")
    cpu.add_argument("share", type=Path)
    cpu.add_argument("nvme", type=Path)
    gpu = checks.add_parser("gpu")
    gpu.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.check == "cpu":
        cpu_check(args.share, args.nvme)
    else:
        gpu_check(args.output)


if __name__ == "__main__":
    main()
