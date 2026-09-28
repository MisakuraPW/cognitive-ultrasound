"""Retry only four failed low-precision SHORT jobs with eager rounding preserved.

Original jobs, identities, thresholds and locked confirmation selection stay intact.
The compiler option is explicit evidence, not a precision fallback or new search.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.preparation.suite import run_lock
from cognitive_ultrasound.compute_lab.suite import owned_alive, stop_process
from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.provenance import sha256

JOBS = tuple(f"short_torch_{steps}_{dtype}_b14" for steps in (50, 25) for dtype in ("fp16", "bf16"))


def validate_task(name, task, result):
    p = task["profile"]
    if name not in JOBS or task["kind"] != "inference" or not task.get("short"):
        raise ValueError("Only the four original SHORT jobs are authorized")
    if task["cohort"] != "debug" or task["budget"] != 14:
        raise ValueError("Debug-only retry; no cohort expansion")
    if p["backend"] != "torch" or p["mode"] != "graph" or p["precision"] not in ("fp16", "bf16"):
        raise ValueError("Original low precision graph profile required")
    if name != f"short_torch_{p['steps']}_{p['precision']}_b14" or p["dps"] != p["steps"]:
        raise ValueError("Scientific profile mismatch")
    if result["status"] != "failed" or result.get("failure_kind") != "numerical_correctness_failure":
        raise ValueError("Never rerun completed or unrelated jobs")
    if "Tensor-likes are not close" not in result.get("error", ""):
        raise ValueError("Failure is not the investigated compiler/eager mismatch")


def run(root, attempt="gap_repair", emulate=True):
    if attempt not in ("gap_repair", "gap_repair_classification"):
        raise ValueError("Unknown bounded repair attempt")
    output = root / attempt
    output.mkdir(exist_ok=True)
    with run_lock(root), run_lock(output):
        state = read_json(root / "status.json")
        if any(owned_alive(state.get(k), state.get(t)) for k, t in (("pid", "created"), ("worker_pid", "worker_created"))):
            raise RuntimeError("Original coordinator/worker still active")
        if (root / "STOP").exists():
            raise RuntimeError("STOP retained")
        if read_json(root / "scope_amendment.json")["confirmation_budgets"] != [7]:
            raise RuntimeError("Requires completed budget7 scope")
        cfg = read_json(root / "config.json")
        plan = dict(
            jobs=list(JOBS),
            compiler_option={"emulate_precision_casts": emulate},
            reason="Preserve eager rounding hypothesis" if emulate else "Original compiler settings; record B numerical rejection separately from runtime failure; unchanged strict tolerance",
            unchanged_tolerance=dict(atol=2e-4, rtol=2e-4),
            scientific_config_sha256=sha256(root / "config.json"),
            original_identity_sha256=sha256(root / "identity.json"),
            original_selection_sha256=sha256(root / "selection.json"),
            original_results={name: sha256(root / "jobs" / name / "result.json") for name in JOBS},
            launcher_sha256=sha256(Path(__file__)),
            runtime_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            automatic_adoption=False,
        )
        plan_file = output / "plan.json"
        if plan_file.exists() and read_json(plan_file) != plan:
            raise RuntimeError("Retry identity changed; preserve previous attempt")
        atomic_json(plan_file, plan)
        records = {}
        for name in JOBS:
            task = read_json(root / "jobs" / name / "task.json")
            validate_task(name, task, read_json(root / "jobs" / name / "result.json"))
            directory = output / "jobs" / name
            directory.mkdir(parents=True, exist_ok=True)
            result = directory / "result.json"
            if not result.exists():
                task = dict(task, output=str(directory))
                atomic_json(directory / "task.json", task)
                env = dict(os.environ, TORCHINDUCTOR_EMULATE_PRECISION_CASTS="1" if emulate else "0", TORCHINDUCTOR_COMPILE_THREADS="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8", NVIDIA_TF32_OVERRIDE="0",
                           OMP_NUM_THREADS=str(cfg["threads"]), PYTHONUNBUFFERED="1")
                command = [cfg["pythons"]["torch"], "-m", "cognitive_ultrasound.compute_lab", "worker",
                           "--config", str(root / "config.json"), "--output", str(root), "--task", str(directory / "task.json")]
                start = time.perf_counter()
                print("RETRY", name, flush=True)
                with (directory / "console.log").open("a", encoding="utf-8") as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    atomic_json(output / "status.json", dict(status="running", job=name, worker_pid=child.pid))
                    try:
                        while child.poll() is None:
                            if (root / "STOP").exists():
                                raise RuntimeError("STOP requested")
                            time.sleep(1)
                    finally:
                        if child.poll() is None:
                            stop_process(child)
                value = read_json(result) if result.exists() else dict(status="failed", error="Worker exited without result")
                value.update(process_wall_s=time.perf_counter()-start, returncode=child.returncode,
                             compiler_option=plan["compiler_option"], original_failed_result_sha256=plan["original_results"][name])
                atomic_json(result, value)
            value = read_json(result)
            records[name] = dict(status=value["status"], error=value.get("error"),
                                 operator_checks=value.get("operator_checks"), result_sha256=sha256(result),
                                 original_retained=True, quality_scope="debug SHORT only; no confirmation or adoption")
            print("RETRY_END", name, value["status"], flush=True)
            atomic_json(output / "summary.json", records)
        for name, digest in plan["original_results"].items():
            if sha256(root / "jobs" / name / "result.json") != digest:
                raise RuntimeError("Original failure evidence changed")
        atomic_json(output / "status.json", dict(status="completed" if all(v["status"]=="completed" for v in records.values()) else "completed_with_rejections",
                                                original_evidence_preserved=True, automatic_adoption=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--attempt", choices=("gap_repair","gap_repair_classification"), default="gap_repair")
    p.add_argument("--emulate-precision-casts", choices=(0,1), type=int, default=1)
    args=p.parse_args()
    run(args.output.resolve(),args.attempt,bool(args.emulate_precision_casts))
