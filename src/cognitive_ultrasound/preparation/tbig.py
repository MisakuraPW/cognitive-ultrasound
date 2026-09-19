"""Optional official TBIG bridge, in a separate user-provided compatible environment."""

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..provenance import sha256
from .common import atomic_json, atomic_npz, read_json


def run(task, cfg, output):
    spec = cfg["tbig"]
    repo, config = Path(spec["repo"]).resolve(), Path(spec["config"]).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != spec["expected_commit"]:
        raise ValueError("TBIG commit differs from explicit expected_commit")
    if subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only"], text=True).strip():
        raise ValueError("TBIG has tracked changes; use an audited clean checkout")
    if not (repo / "users.yaml").exists():
        raise FileNotFoundError(
            "TBIG requires its own users.yaml and existing compatible weights/data"
        )
    required_assets = spec.get("asset_files", [])
    if not required_assets or any(not Path(f).is_file() for f in required_assets):
        raise ValueError(
            "TBIG needs explicit asset_files listing frozen diffusion and downstream checkpoint files"
        )
    identity = dict(
        commit=actual,
        config_sha256=sha256(config),
        users_sha256=sha256(repo / "users.yaml"),
        asset_sha256={str(f): sha256(f) for f in required_assets},
        sequence_sha256={str(f): sha256(f) for f in spec.get("sequences", [])},
    )
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("TBIG input/assets changed; do not merge with old results")
    atomic_json(output / "identity.json", identity)
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path[:0] = [str(repo), str(repo / "zea")]
    os.chdir(repo)
    official = importlib.import_module("active_sampling_temporal")
    cases = spec.get("sequences", [])
    if not 1 <= len(cases) <= 2 or not 3 <= spec["frames"] <= 24:
        raise ValueError("TBIG MVP needs 1–2 explicit compatible sequences, 3–24 frames")
    results = []
    for i, sequence in enumerate(cases):
        result_file = output / f"case_{i}.npz"
        if result_file.exists():
            continue
        tick = time.perf_counter()
        values = official.active_sampling_single_file(
            str(config),
            target_sequence=str(Path(sequence).resolve()),
            seed=cfg["seed"],
            override_config={"io_config": {"frame_cutoff": spec["frames"]}},
        )
        if len(values) != 8:
            raise ValueError("Unsupported official TBIG API; expected eight return values")
        result, downstream, target_task, reconstruction_task, beliefs_task, _, _, _ = values
        arrays = {
            k: np.asarray(getattr(result, k))
            for k in (
                "masks",
                "target_imgs",
                "reconstructions",
                "belief_distributions",
                "measurements",
            )
        }
        if any(v.dtype == object or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Invalid TBIG numeric output")
        if downstream is None:
            raise ValueError(
                "Config has no downstream task; this would not test TBIG task-directed behavior"
            )
        for key, value in [
            ("full_image_task_reference", target_task),
            ("reconstruction_task", reconstruction_task),
            ("beliefs_task", beliefs_task),
        ]:
            array = np.asarray(value)
            if array.dtype == object:
                raise ValueError("Unsupported non-array downstream output")
            arrays[key] = array
        atomic_npz(result_file, **arrays)
        results.append(dict(sequence=str(sequence), seconds=time.perf_counter() - tick))
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            commit=actual,
            config_sha256=sha256(config),
            sequences=cases,
            completed_this_attempt=results,
            caveat="Official single-sequence task smoke/diagnostic, not paper replication; full-image task output is a model reference, not clinical ground truth",
        ),
    )
