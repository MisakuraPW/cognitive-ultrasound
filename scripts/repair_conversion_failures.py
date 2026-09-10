"""Repair diagnosed missing cases only after the main conversion finishes."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["JAX_PLATFORMS"] = "cpu"

import psutil  # noqa: E402
import yaml  # noqa: E402

from cognitive_ultrasound import conversion as conv  # noqa: E402
from cognitive_ultrasound.conversion_compat import compatible_processor, split_digest  # noqa: E402
from cognitive_ultrasound.data import audit_dataset, read_splits  # noqa: E402


def main():
    state_dir = ROOT / "results/local_conversion"
    state = json.loads((state_dir / "status.json").read_text())
    if psutil.pid_exists(state["pid"]):
        raise RuntimeError("Main conversion PID still exists; wait for it to exit")
    output, raw = Path(state["output"]), Path(state["raw"])
    registry = json.loads((ROOT / "configs/conversion_compatibility.json").read_text())
    splits = read_splits(ROOT / "configs/splits/split.yaml")
    if split_digest(splits) != registry["splits_sha256"]:
        raise ValueError("Pinned splits changed")
    with conv.conversion_lock(state_dir), conv.conversion_lock(output):
        progress = json.loads((output / "conversion_progress.json").read_text())
        metadata = json.loads((output / "conversion_manifest.json").read_text())
        failures = json.loads((output / "conversion_failures.json").read_text())
        if progress["remaining"] != 0 or metadata["status"] != "failed":
            raise RuntimeError("Full conversion must finish before targeted repair")
        if not failures or any(
            f["source"] not in registry["cases"]
            or f["error"] != "AssertionError: Rejection mismatch"
            or f["split"] != registry["cases"][f["source"]]["split"]
            for f in failures
        ):
            raise ValueError("Undiagnosed conversion failure; refusing automatic exception")
        identity = metadata["identity"]
        inventory = {
            f"{p.stem}.hdf5": [p.name, p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted((raw / "Videos").glob("*.avi"))
        }
        if identity != {
            "zea_commit": conv.ZEA_COMMIT,
            "raw": str(raw),
            "splits": splits,
            "sources": inventory,
        }:
            raise ValueError("Source inventory or pinned identity changed")
        if any(p.is_symlink() for p in output.rglob("*")):
            raise ValueError("Symlink in conversion output")
        old_files = {
            p: (p.stat().st_size, p.stat().st_mtime_ns)
            for s in ("train", "val", "test", "rejected")
            for p in (output / s).glob("*.hdf5")
        }
        missing = set(inventory) - {p.name for p in old_files}
        if missing != {Path(f["source"]).stem + ".hdf5" for f in failures}:
            raise ValueError("Missing files do not exactly match diagnosed failures")
        conv.atomic_json(state_dir / "failures_before_repair.json", failures)
        conv.init_worker(splits)
        conv.PROCESSOR = compatible_processor(conv.PROCESSOR, registry)
        digest = (
            __import__("hashlib").sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        )
        repaired = []
        state.update(status="running", pid=os.getpid(), stage="compatibility_repair")
        conv.atomic_json(state_dir / "status.json", state)
        try:
            for failure in failures:
                item = conv.convert_one(
                    raw / "Videos" / failure["source"], output, failure["split"], digest
                )
                conv.verify_h5(
                    output / failure["split"] / item["file"],
                    failure["split"] != "rejected",
                    registry["cases"][failure["source"]]["frames"],
                )
                repaired.append(item)
                print(f"Repaired {len(repaired)}/{len(failures)}: {item['file']}", flush=True)
            if any(
                (p.stat().st_size, p.stat().st_mtime_ns) != stat for p, stat in old_files.items()
            ):
                raise ValueError("Previously published output changed during repair")
            produced = {
                s: sorted(p.name for p in (output / s).glob("*.hdf5"))
                for s in ("train", "val", "test", "rejected")
            }
            if any(produced[s] != splits[s] for s in splits) or sum(
                map(len, produced.values())
            ) != len(inventory):
                raise ValueError("Final split or total file count mismatch")
            (output / "split.yaml").write_text(yaml.safe_dump(produced), encoding="utf-8")
            # Main run validated existing files; new files verified at atomic publication.
            state["stage"] = "audit"
            conv.atomic_json(state_dir / "status.json", state)
            audit_dataset(
                output, ROOT / "configs/splits/split.yaml", state_dir / "dataset_statistics.md"
            )
            conv.atomic_json(
                output / "conversion_compatibility_report.json",
                {"registry": registry, "repaired": repaired, "retained_unchanged": len(old_files)},
            )
            metadata.update(status="completed", failures=[], compatibility=registry)
            conv.atomic_json(output / "conversion_manifest.json", metadata)
            conv.atomic_json(output / "conversion_failures.json", [])
            progress.update(
                reused=len(old_files),
                converted_this_run=len(repaired),
                failed_this_run=0,
                remaining=0,
                estimated_remaining_hours=0,
            )
            conv.atomic_json(output / "conversion_progress.json", progress)
            state.update(status="completed", stage="audit", error=None)
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            conv.atomic_json(state_dir / "status.json", state)


if __name__ == "__main__":
    main()
