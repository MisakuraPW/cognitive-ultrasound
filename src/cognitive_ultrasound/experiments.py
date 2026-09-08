import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from .config import ROOT, path, validate
from .data import SPLIT_COUNTS, inspect_file, read_splits
from .evaluation.metrics import Metrics
from .provenance import environment, sha256, write_json
from .visualization import save_frame


def case_seed(seed, name):
    return (seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")) % (2**31)


def evaluate(cfg, resume=False):
    validate(cfg)
    from .evaluation.segmentation import Segmentation, segmentation_failure
    from .models.diffusion import CASLLoop

    root, output = path(cfg["data_root"]), path(cfg["output"])
    manifest = path(cfg["split_manifest"])
    names = read_splits(manifest)[cfg["split"]]
    if not cfg.get("synthetic_input") and read_splits(manifest) != read_splits(
        ROOT / "configs/splits/split.yaml"
    ):
        raise ValueError("Research evaluation must use the pinned official patient split")
    if cfg["limit_cases"] is not None:
        names = names[: cfg["limit_cases"]]
    if not names:
        raise ValueError("No cases to evaluate")
    # Fail before costly model loading. Never silently skip missing patients.
    for name in names:
        inspect_file(root / cfg["split"] / name)
    checkpoint = path(cfg["checkpoint"])
    weight_hashes = {n: sha256(checkpoint / n) for n in ("config.json", "model.weights.h5")}
    identity = {
        "config": cfg,
        "checkpoint_sha256": weight_hashes,
        "split_sha256": sha256(manifest),
        "cases": names,
        "data_sha256": {name: sha256(root / cfg["split"] / name) for name in names},
    }
    from .assets import asset_hashes

    asset_names = (["lpips"] if "lpips" in cfg["metrics"] else []) + (
        ["segmentation"] if cfg["segmentation"] else []
    )
    identity["evaluation_checkpoint_sha256"] = asset_hashes(
        path(cfg["evaluation_checkpoints"]), asset_names
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "manifest.json"
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not resume or existing["identity"] != identity:
            raise FileExistsError(
                "Existing run: use --resume with identical configuration or new output"
            )
    else:
        if any(output.iterdir()):
            raise FileExistsError("Output is not empty and has no run manifest")
        write_json(
            manifest_file,
            {
                "identity": identity,
                "environment": environment(),
                "timing": "Synchronized, excludes I/O, metrics and visualization; first frame separate",
                "status": "running",
            },
        )
    metrics = Metrics(cfg["metrics"], path(cfg["evaluation_checkpoints"]))
    segmentation = (
        Segmentation(path(cfg["evaluation_checkpoints"])) if cfg["segmentation"] else None
    )
    for method in cfg["methods"]:
        for budget in cfg["budgets"]:
            loop = None
            for name in names:
                folder = output / method / f"lines_{budget:03d}" / Path(name).stem
                done = folder / "complete.json"
                if resume and done.exists():
                    continue
                if loop is None:
                    loop = CASLLoop(cfg, method, budget)
                seed = case_seed(cfg["seed"], name)
                loop.reset(seed)
                with h5py.File(root / cfg["split"] / name, "r") as handle:
                    frames = handle["data/image"][: cfg["frames"]].astype("float32")
                frames = (frames + 60.0) / 30.0 - 1.0
                frames = frames[..., None]
                folder.mkdir(parents=True, exist_ok=True)
                print(f"{method} K={budget} {name} ({len(frames)} frames)", flush=True)
                # Compile both first-frame and SeqDiff branches, then reset all patient state.
                start = perf_counter()
                for frame in frames[:2]:
                    loop.step(frame)
                warmup = perf_counter() - start
                loop.reset(seed)
                rows, reference_masks = [], []
                for index, frame in enumerate(frames):
                    state, timing = loop.step(frame)
                    row = {
                        "method": method,
                        "budget": budget,
                        "case": Path(name).stem,
                        "frame": index,
                        "seed": seed,
                        "actual_lines": state["actual_lines"],
                        **metrics(frame, state["reconstruction"]),
                        **timing,
                    }
                    if segmentation:
                        gt_mask, prediction, agreement = segmentation(
                            frame, state["reconstruction"]
                        )
                        state.update(
                            segmentation_reference=gt_mask, segmentation_prediction=prediction
                        )
                        reference_masks.append(gt_mask)
                        row["dice_agreement"] = agreement
                    frame_dir = folder / "trajectory" / f"frame_{index:04d}"
                    if cfg["save_trajectory"]:
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(frame_dir / "state.npz", **state)
                    if cfg["visualize_every"] > 0 and index % cfg["visualize_every"] == 0:
                        save_frame(frame_dir, state)
                    rows.append(row)
                # A completion marker is written only after all outputs succeed.
                with open(folder / "frames.csv", "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                write_json(
                    done,
                    {
                        "frames": len(rows),
                        "warmup_compile_s": warmup,
                        "short_sequence": len(rows) < cfg["frames"],
                        "segmentation_excluded": segmentation_failure(reference_masks)
                        if segmentation
                        else None,
                        "budget_mismatch_frames": sum(r["actual_lines"] != budget for r in rows),
                    },
                )
    record = json.loads(manifest_file.read_text(encoding="utf-8"))
    record["status"] = "completed"
    record["full_paper_case_count"] = len(names) == SPLIT_COUNTS[cfg["split"]]
    write_json(manifest_file, record)
    from .evaluation.report import generate_report

    generate_report(output)


def synthetic_smoke(output):
    """Exercise artifacts and metrics only. No toy model masquerades as CASL."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh synthetic output directory")
    y, x = np.mgrid[:112, :112]
    reference = (2 * np.exp(-((x - 56) ** 2 + (y - 60) ** 2) / 500) - 1).astype("float32")
    mask = np.zeros((112, 112), dtype="float32")
    mask[:, ::16] = 1
    # A perturbed reference is for metric/serialization tests only, never a reconstruction baseline.
    candidate = np.clip(reference + 0.05 * np.sin(x), -1, 1).astype("float32")
    state = {
        "ground_truth": reference,
        "observation": reference * mask,
        "observation_mask": mask,
        "reconstruction": candidate,
        "entropy_map": np.abs(candidate - reference),
        "selected_action": mask[0],
    }
    save_frame(output / "frame_0000", state)
    np.savez_compressed(output / "frame_0000/state.npz", **state)
    result = {
        "status": "SYNTHETIC_ARTIFACT_TEST_ONLY",
        "casl_executed": False,
        "metrics": Metrics(["psnr", "ssim"])(reference, candidate),
    }
    write_json(output / "smoke.json", result)
    return result
