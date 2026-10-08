"""Same-case GPU controls for failed GS replay; verify partial E2 forward compatibility."""

import argparse
import hashlib
import subprocess
import time
import types
from pathlib import Path

import numpy as np

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.task_budget.data import read_episode
from cognitive_ultrasound.task_budget.episode import gs_gradient, rollout
from cognitive_ultrasound.task_budget.experiment import setup
from cognitive_ultrasound.task_budget.policy import initialize, restore


def legacy_rollout(base):
    source = subprocess.check_output(
        ["git", "show", f"{base}:src/cognitive_ultrasound/task_budget/episode.py"], cwd=ROOT
    )
    module = types.ModuleType("cognitive_ultrasound.task_budget._legacy_episode")
    module.__package__ = "cognitive_ultrasound.task_budget"
    exec(compile(source, "reviewed_legacy_episode.py", "exec"), module.__dict__)
    return module.rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", default="a8f4d26")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg, manifest = read_json(source / "config.json"), read_json(source / "manifest.json")
    perception, task = setup(cfg, output)
    record = dict(kind="CONTROLLED_GPU_REPAIR_DIAGNOSTIC_NOT_EFFICACY", base=args.base)
    try:
        name = manifest["cohorts"]["development"][0]
        params = initialize(42, cfg)
        images, rows, contexts = rollout(
            cfg,
            perception,
            task,
            read_episode(cfg, manifest, name, count=3),
            params,
            "E0",
            42,
            fixed=[10, 4],
        )
        prediction, adjoint, _ = task.video(images, gradient=True)
        record.update(case=name, ef_prediction=prediction, rows=rows)
        replay = perception.infer_for_replay
        for mode in ("legacy_ad_forward", "same_primal_custom_vjp"):
            perception.infer_for_replay = (
                perception.infer if mode == "legacy_ad_forward" else replay
            )
            start = time.perf_counter()
            try:
                _, gate = gs_gradient(params, contexts, adjoint, perception, cfg, 1.0, 0.0)
                record[mode] = dict(
                    passed=all(v > 0 for v in gate["task_gradient_norms"].values()), **gate
                )
            except Exception as error:
                record[mode] = dict(passed=False, error=f"{type(error).__name__}: {error}")
            record[mode]["seconds"] = time.perf_counter() - start
            atomic_json(output / "diagnostic.json", record)
            print(mode, record[mode], flush=True)
        perception.infer_for_replay = replay
        checkpoint = sorted((source / "jobs/E2_l0_s42_train/updates").glob("*.npz"))[-1]
        params, opt, _ = restore(checkpoint)
        rng = np.random.default_rng(42 * 1000003 + opt["step"])
        name = manifest["cohorts"]["train"][int(rng.integers(len(manifest["cohorts"]["train"])))]
        n = min(cfg["training"]["clip_frames"], manifest["files"][name]["frames"])
        start = int(rng.integers(manifest["files"][name]["frames"] - n + 1))
        seed = int(rng.integers(2**31))
        frames = read_episode(cfg, manifest, name, start, 3)
        old_images, old_rows, _ = legacy_rollout(args.base)(
            cfg, perception, task, frames, params, "E2", seed, training=True
        )
        new_images, new_rows, _ = rollout(
            cfg, perception, task, frames, params, "E2", seed, training=True
        )
        actions = all(
            a[k] == b[k]
            for a, b in zip(old_rows, new_rows)
            for k in ("k1", "k2", "lines1", "lines2")
        )
        exact = old_images.tobytes() == new_images.tobytes()
        record["e2_forward"] = dict(
            passed=actions and exact,
            actions_identical=actions,
            images_bitwise=exact,
            max_abs=float(np.max(np.abs(old_images - new_images))),
            case=name,
            start=start,
            checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            resume_step=opt["step"],
            scope="first3 frames of the next training episode; supplemented by AST compatibility audit",
        )
        atomic_json(output / "diagnostic.json", record)
        print("e2_forward", record["e2_forward"], flush=True)
        if not record["same_primal_custom_vjp"]["passed"] or not record["e2_forward"]["passed"]:
            raise RuntimeError("Repair gate failed; do not inherit or start batch")
    finally:
        task.close()


if __name__ == "__main__":
    main()
