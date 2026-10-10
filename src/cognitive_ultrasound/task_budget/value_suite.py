"""Diagnostic lifecycle reusing task-budget identity, locks and owned processes."""

import copy
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

from ..config import ROOT, path
from ..data import read_splits
from ..preparation.common import atomic_json, digest, emit, read_json
from ..preparation.suite import run_lock, stop_process
from ..provenance import sha256
from .data import configuration
from .suite import identity, terminate_owned_worker
from .suite import process_alive as process_alive

PHASES = ["P0", "P1", "P2", "P3", "P4", "P5"]


def config(file):
    cfg = configuration(file)
    d = cfg["value_diagnostics"]
    for key in ["source_batch", "expansion_parent"]:
        d[key] = str(path(d[key]).resolve())
    d["historical_roots"] = [str(path(p).resolve()) for p in d["historical_roots"]]
    if d["phases"] != PHASES or d["seed"] != 42 or d["verification_seed"] != 31415:
        raise ValueError("Frozen diagnostic phases/seeds changed")
    if d["checkpoint_frames"] < 1 or d["debug_frames"] != 64:
        raise ValueError("Invalid diagnostic checkpoint/debug protocol")
    if d["extension_strata"] != [6, 5, 5] or d["min_extension_frames"] != 96:
        raise ValueError("Locked16-case extension coverage changed")
    if any(
        not np.isfinite(d[k]) or d[k] <= 0
        for k in ["gate_ef_pp", "gate_mse_abs", "mse_margin", "p90_margin", "ef_margin_pp"]
    ):
        raise ValueError("Positive predeclared diagnostic margins required")
    if (
        d["bootstrap_seed"] != 20261010
        or d["bootstrap_replicates"] != 5000
        or d["predictor_alpha"] != 1
    ):
        raise ValueError("Frozen statistical protocol changed")
    if cfg["perception"] != {"warm_steps": 25, "particles": 2, "precision": "float32", "omega": 10}:
        raise ValueError("Diagnostics keep the shared25-step FP32 perception")
    if (
        cfg["runtime"].get("ef_execution") != "eager"
        or cfg["runtime"].get("evaluation_workers") != 1
        or cfg["runtime"]["threads"] != 2
    ):
        raise ValueError(
            "Only the previously qualified serial eager engineering profile is enabled"
        )
    return cfg


def _used_names(value):
    if isinstance(value, dict):
        return set().union(*(_used_names(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_used_names(v) for v in value)) if value else set()
    return (
        {value.replace("\\", "/").split("/")[-1]}
        if isinstance(value, str) and value.endswith(".hdf5")
        else set()
    )


def extension(cfg, source):
    d = cfg["value_diagnostics"]
    parent = Path(d["expansion_parent"])
    if (
        not (parent / "status.json").exists()
        or read_json(parent / "status.json").get("status") != "completed"
    ):
        raise ValueError("Explicit expansion needs a completed initial diagnostic batch")
    excluded = _used_names(source["cohorts"])
    origins = []
    for root in d["historical_roots"]:
        for p in sorted(Path(root).glob("*/manifest.json")):
            excluded.update(_used_names(read_json(p).get("cohorts", {})))
            origins.append(dict(path=str(p), sha256=sha256(p)))
    splits = read_splits(Path(cfg["split_manifest"]))
    held = set(splits["val"]) | set(splits["test"])
    with open(cfg["file_list"], encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    pools = [[], [], []]
    for row in labels:
        name = Path(row["FileName"]).stem + ".hdf5"
        ef = float(row["EF"])
        if row["Split"].upper() != "VAL" or name not in held or name in excluded:
            continue
        split = "val" if name in set(splits["val"]) else "test"
        file = Path(cfg["data_root"]) / split / name
        if not file.exists():
            continue
        with h5py.File(file) as h:
            n = int(h["data/image"].shape[0])
        if n < d["min_extension_frames"]:
            continue
        pools[0 if ef < 40 else 1 if ef < 60 else 2].append(
            (name, dict(relative=f"{split}/{name}", frames=n, ef=ef, task_split="VAL"))
        )
    rng = np.random.default_rng(d["extension_seed"])
    selected = []
    files = {}
    for pool, count in zip(pools, d["extension_strata"]):
        pool = sorted(pool)
        if len(pool) < count:
            raise ValueError(f"Insufficient independent diagnostic stratum:{len(pool)}<{count}")
        for i in rng.choice(len(pool), count, replace=False):
            name, item = pool[int(i)]
            selected.append(name)
            item["sha256"] = sha256(Path(cfg["data_root"]) / item["relative"])
            files[name] = item
    return (
        selected,
        files,
        dict(
            excluded_count=len(excluded),
            history=origins,
            parent_identity=read_json(parent / "identity.json"),
            strata=d["extension_strata"],
            scope="new validation diagnostics; not blind clinical test",
        ),
    )


def prepare(cfg, root, expand=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    required_free = 2 if (root / "identity.json").exists() else 24
    if (
        not cfg["value_diagnostics"]["functional_fixture"]
        and shutil.disk_usage(root).free < required_free * 2**30
    ):
        raise OSError(
            "Diagnostic full-frame/state cache requires at least24GiB free; old data are never deleted"
        )
    source = Path(cfg["value_diagnostics"]["source_batch"])
    original = read_json(source / "manifest.json")
    old_cfg = read_json(source / "config.json")
    if old_cfg.get("functional_fixture") and not cfg["value_diagnostics"]["functional_fixture"]:
        raise ValueError("Synthetic source is not a scientific cohort")
    if read_json(source / "status.json")["status"] != "completed":
        raise ValueError("Source repair batch incomplete")
    for key in ["feature_scales", "feature_calibration"]:
        if key in old_cfg:
            cfg[key] = copy.deepcopy(old_cfg[key])
    cfg["value_diagnostics"]["expanded"] = bool(expand)
    if (root / "config.json").exists() and read_json(root / "config.json") != cfg:
        raise ValueError("Diagnostic config changed; use a separate output directory")
    atomic_json(root / "config.json", cfg)
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        if manifest["source_manifest_sha256"] != sha256(source / "manifest.json"):
            raise ValueError("Source cohort changed")
    else:
        if expand:
            names, files, selection = extension(cfg, original)
        else:
            names = original["cohorts"]["development"]
            files = {n: original["files"][n] for n in names}
            if len(names) != 8 and not cfg["value_diagnostics"]["functional_fixture"]:
                raise ValueError("Expected the8 locked development cases")
            selection = dict(scope="existing development discovery; not unseen validation")
        manifest = dict(
            cohorts={"development": names},
            files=files,
            sources=original["sources"],
            source_manifest_sha256=sha256(source / "manifest.json"),
            selection=selection,
        )
        manifest["identity"] = digest(manifest)
        atomic_json(root / "manifest.json", manifest)
    if not cfg["value_diagnostics"]["functional_fixture"]:
        for key in ["file_list", "split_manifest"]:
            if sha256(cfg[key]) != manifest["sources"][key]:
                raise ValueError("Label/split source changed")
        for key in ["ef_weights", "ef_stats"]:
            if sha256(cfg[key]) != read_json(source / "identity.json")[key]:
                raise ValueError("Task asset changed")
        for p in Path(cfg["checkpoint"]).iterdir():
            if p.is_file() and sha256(p) != read_json(source / "identity.json")["checkpoint"].get(
                p.name
            ):
                raise ValueError("Frozen diffusion asset changed")
        with open(cfg["file_list"], encoding="utf-8-sig", newline="") as f:
            labels = {Path(r["FileName"]).stem + ".hdf5": r for r in csv.DictReader(f)}
        split = read_splits(Path(cfg["split_manifest"]))
        held = set(split["val"]) | set(split["test"])
        for name, item in manifest["files"].items():
            if (
                name not in held
                or labels[name]["Split"].upper() != "VAL"
                or float(labels[name]["EF"]) != item["ef"]
            ):
                raise ValueError(
                    "Diagnostic cohort/label no longer matches held-out validation sources"
                )
    for name, item in manifest["files"].items():
        p = Path(cfg["data_root"]) / item["relative"]
        if sha256(p) != item["sha256"]:
            raise ValueError(f"Diagnostic input changed:{name}")
        with h5py.File(p) as h:
            shape = tuple(h["data/image"].shape)
        if (
            shape != (item["frames"], 112, 112)
            or item["frames"] < 5
            or not np.isfinite(item["ef"])
            or not 0 <= item["ef"] <= 100
        ):
            raise ValueError(f"Invalid diagnostic geometry/label:{name}")
    if cfg["value_diagnostics"]["functional_fixture"]:
        current = dict(
            config=digest(cfg),
            manifest=manifest["identity"],
            functional_fixture=True,
            source={
                str(p.relative_to(ROOT)): sha256(p)
                for p in sorted(
                    [
                        *Path(__file__).parent.glob("*.py"),
                        ROOT / "src/cognitive_ultrasound/preparation/common.py",
                    ]
                )
            },
        )
    else:
        current = identity(cfg, manifest)
    if (root / "identity.json").exists() and read_json(root / "identity.json") != current:
        raise ValueError(
            "Source/assets/hardware changed; keep results and use a new output directory"
        )
    atomic_json(root / "identity.json", current)
    atomic_json(
        root / "plan.json",
        dict(
            phases=PHASES,
            initial_cases=len(manifest["files"]),
            variant_caps=dict(P1=3, P2=18, P3=8, P4_seed42=8, P4_seed31415=8),
            no_budget_training=True,
            no_automatic_expansion=True,
            statistical_unit="video",
        ),
    )
    return cfg, manifest


def cleanup_worker(process):
    """A Linux worker has its own session; kill its EF group even after an abort."""
    if process.poll() is None:
        stop_process(process)
    elif os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # No original members may escape the phase after a native/JAX abort.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(cfg, root, action="run", expand=False, launch=None, probe_only=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        old = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if action == "resume":
            terminate_owned_worker(old)
            (root / "STOP").unlink(missing_ok=True)
        elif (root / "STOP").exists():
            raise ValueError("Stopped batch; explicitly resume")
        cfg, manifest = prepare(cfg, root, expand)
        state = dict(
            status="running",
            pid=os.getpid(),
            created=psutil.Process().create_time(),
            started=old.get("started", time.time()),
            failures=[],
            completed_phases=[],
            functional_fixture=cfg["value_diagnostics"]["functional_fixture"],
        )
        atomic_json(root / "status.json", state)
        try:
            for phase in PHASES[:1] if probe_only else PHASES:
                p = root / "jobs" / phase / "result.json"
                if p.exists() and read_json(p).get("status") in ["completed", "skipped"]:
                    state["completed_phases"].append(phase)
                    emit("REUSE", phase=phase)
                    continue
                if (root / "STOP").exists():
                    raise InterruptedError("Stop requested")
                state.update(phase=phase)
                atomic_json(root / "status.json", state)
                emit("STAGE", phase=phase)
                if launch is not None:
                    launch(phase, cfg, manifest, root)
                else:
                    directory = root / "jobs" / phase
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory / "console.log").open("a", encoding="utf-8") as log:
                        proc = subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                "-m",
                                "cognitive_ultrasound.task_budget",
                                "value-worker",
                                "--output",
                                str(root),
                                "--phase",
                                phase,
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=os.name != "nt",
                            env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                        )
                        state.update(
                            worker_pid=proc.pid,
                            worker_created=psutil.Process(proc.pid).create_time(),
                        )
                        atomic_json(root / "status.json", state)
                        start = time.monotonic()
                        last = None
                        try:
                            while proc.poll() is None:
                                if (root / "STOP").exists():
                                    raise InterruptedError("Stop requested")
                                if (
                                    time.monotonic() - start
                                    > cfg["runtime"]["job_timeout_hours"] * 3600
                                ):
                                    raise TimeoutError("Phase timeout")
                                progress = root / "worker_progress.json"
                                if progress.exists():
                                    detail = read_json(progress)
                                    if detail != last and detail.get("phase") == phase:
                                        state["progress"] = detail
                                        atomic_json(root / "status.json", state)
                                        if (
                                            detail.get("frame", 0) % 16 == 0
                                            or "condition" in detail
                                        ):
                                            emit("PROGRESS", **detail)
                                        last = detail
                                time.sleep(0.5)
                            if proc.returncode:
                                raise RuntimeError(
                                    f"{phase} worker failed; see{directory}/console.log"
                                )
                        finally:
                            cleanup_worker(proc)
                            state.pop("worker_pid", None)
                            state.pop("worker_created", None)
                if not p.exists() or read_json(p).get("status") not in ["completed", "skipped"]:
                    raise RuntimeError(f"{phase} failed to commit its result")
                state["completed_phases"].append(phase)
                atomic_json(root / "status.json", state)
                emit("STAGE_END", phase=phase, status=read_json(p)["status"])
                from .value_report import report

                report(root)
            state.update(
                status="probe_completed" if probe_only else "completed", finished=time.time()
            )
        except InterruptedError as error:
            state.update(status="stopped", error=str(error), finished=time.time())
        except Exception as error:
            state.update(
                status="failed",
                error=f"{type(error).__name__}:{error}",
                finished=time.time(),
                failures=[state.get("phase")],
            )
            raise
        finally:
            atomic_json(root / "status.json", state)
            from .value_report import report

            report(root)
        if state["status"] == "completed":
            from .value_report import bundle

            emit("BUNDLE", **bundle(root))
    return state
