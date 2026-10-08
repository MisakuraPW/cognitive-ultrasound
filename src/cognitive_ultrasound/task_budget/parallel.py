"""Bounded independent-case evaluation; causal frames inside a video remain serial."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..preparation.common import atomic_json, read_json
from ..preparation.suite import stop_process


def evaluate_parallel(spec, cfg, manifest, root, output):
    workers = int(cfg["runtime"]["evaluation_workers"])
    if workers != 2:
        raise ValueError("Only calibrated evaluation concurrency 1 or 2 is supported")
    names = manifest["cohorts"][spec["cohort"]]
    pending = [n for n in names if not (output / Path(n).stem / "complete.json").exists()]
    children = []
    streams = []
    started = time.perf_counter()
    try:
        for index in range(min(workers, len(pending))):
            directory = output / ".workers" / str(index)
            directory.mkdir(parents=True, exist_ok=True)
            child = dict(spec, _case_names=pending[index::workers])
            task = directory / "task.json"
            atomic_json(task, child)
            stream = (directory / "console.log").open("a", encoding="utf-8")
            streams.append(stream)
            p = subprocess.Popen(
                [sys.executable, "-u", "-m", "cognitive_ultrasound.task_budget", "worker",
                 "--output", str(root), "--task", str(task), "--worker-output", str(directory)],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name != "nt",
            )
            children.append((p, directory))
        while any(p.poll() is None for p, _ in children):
            # Copy only atomically committed cases; interrupted cases never masquerade as complete.
            for _, directory in children:
                for complete in directory.glob("*/complete.json"):
                    dest = output / complete.parent.name
                    if not (dest / "complete.json").exists():
                        dest.mkdir(exist_ok=True)
                        for f in complete.parent.iterdir():
                            if f.name != "complete.json":
                                shutil.copy2(f, dest / f.name)
                        atomic_json(dest / "complete.json", read_json(complete))
            if (root / "STOP").exists():
                raise InterruptedError("STOP during parallel evaluation")
            if any(p.poll() not in (None, 0) for p, _ in children):
                raise RuntimeError("Parallel worker failed; see .workers/*/console.log")
            if time.perf_counter() - started > cfg["runtime"]["job_timeout_hours"] * 3600:
                raise TimeoutError("Parallel evaluation timeout")
            time.sleep(0.5)
        if any(p.returncode for p, _ in children):
            raise RuntimeError("Parallel worker failed")
        # The final poll may race with the last case commit; reconcile once more.
        for _, directory in children:
            for complete in directory.glob("*/complete.json"):
                dest = output / complete.parent.name
                if not (dest / "complete.json").exists():
                    shutil.copytree(complete.parent, dest, dirs_exist_ok=True)
        records = [read_json(output / Path(n).stem / "complete.json") for n in names]
        atomic_json(output / "result.json", dict(
            status="completed", method=spec["method"], cohort=spec["cohort"], seed=spec["seed"],
            cost_weight=spec.get("lambda"), fixed=spec.get("fixed"), records=records,
            evaluation_workers=workers, parallel_wall_s=time.perf_counter()-started,
            timing_scope="Independent cases concurrent; per-case time includes contention",
        ))
    finally:
        for p, _ in children:
            if p.poll() is None:
                stop_process(p)
        for stream in streams:
            stream.close()
