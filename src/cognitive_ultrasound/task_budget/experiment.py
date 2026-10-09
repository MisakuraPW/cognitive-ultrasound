"""Bounded training/evaluation workers; never optimize perception or task parameters."""

import time
from pathlib import Path

import jax
import numpy as np

from ..experiments import case_seed
from ..preparation.common import atomic_json, atomic_npz, emit, read_json
from .data import read_episode
from .episode import gs_gradient, rollout
from .perception import CASLPerception
from .policy import adam, adam_state, initialize, restore, rl_loss, save, probabilities
from .task import EFService


def check_stop(root):
    if (root / "STOP").exists():
        raise InterruptedError("User requested STOP")


def setup(cfg, root):
    perception = CASLPerception(cfg)
    return perception, EFService(cfg, root, perception.coordinates)


def probe(cfg, manifest, root, output):
    perception, task = setup(cfg, output)
    try:
        seed = cfg["seeds"][0]
        params = initialize(seed, cfg)
        name = manifest["cohorts"]["development"][0]
        frames = read_episode(cfg, manifest, name, count=3)
        images, rows, contexts = rollout(
            cfg, perception, task, frames, params, "E0", seed, progress=lambda i: check_stop(root)
        )
        pred, adjoint, _ = task.video(images, gradient=True)
        result = dict(
            status="completed",
            production_gpu=True,
            task_frozen=True,
            perception_frozen=True,
            causal_window=True,
            frame_records=rows,
            ef_prediction=pred,
            task_service_wall_s=task.seconds,
            gs_status="not_probed",
        )
        try:
            _, gate = gs_gradient(params, contexts, adjoint, perception, cfg, 1.0, 0.0)
            if not all(v > 0 for v in gate["task_gradient_norms"].values()):
                raise AssertionError("EF task loss fails to reach both budget heads")
            result.update(gs_status="passed", gs_gate=gate)
        except Exception as error:
            result.update(gs_status="failed", gs_error=f"{type(error).__name__}: {error}")
        step_s = np.mean([r["frame_wall_s"] for r in rows[1:]])
        backward_s = result.get("gs_gate", {}).get("backward_wall_s", 0) / 2
        n = cfg["training"]["clip_frames"]
        rl_episode = rows[0]["frame_wall_s"] + (n - 1) * step_s
        result["estimates"] = dict(
            rl_episode_s=rl_episode,
            gs_episode_s=rl_episode + (n - 1) * backward_s
            if result["gs_status"] == "passed"
            else None,
            limitation="Three-frame probe includes initial JIT; extrapolation, not guarantee. "
            "Full EF clips, validation, confirmation and output are extra.",
        )
        atomic_json(output / "result.json", result)
    finally:
        task.close()


def train(spec, cfg, manifest, root, output, end_update=None, audit_callback=None):
    perception, task = setup(cfg, output)
    method, seed, weight = spec["method"], spec["seed"], spec["lambda"]
    params = initialize(seed, cfg)
    optimizer, baseline = adam_state(params), 0.0
    completed = sorted((output / "updates").glob("*.npz")) if (output / "updates").exists() else []
    if completed:
        params, optimizer, baseline = restore(completed[-1])
    start_update = optimizer["step"]
    try:
        if spec.get("resume_checkpoint") and not completed:
            source=Path(spec["resume_checkpoint"])
            params,optimizer,baseline=restore(source)
            start_update=optimizer["step"]
        final_update = cfg["training"]["updates"] if end_update is None else min(end_update, cfg["training"]["updates"])
        for update in range(start_update, final_update):
            check_stop(root)
            start = time.perf_counter()
            # Global update-index RNG makes resumed and uninterrupted data/action streams identical.
            rng = np.random.default_rng(seed * 1000003 + update)
            names = manifest["cohorts"]["train"]
            name = names[int(rng.integers(len(names)))]
            count = min(cfg["training"]["clip_frames"], manifest["files"][name]["frames"])
            first_frame = int(rng.integers(manifest["files"][name]["frames"] - count + 1))
            frames = read_episode(cfg, manifest, name, first_frame, count)
            action_seed = int(rng.integers(2**31))
            images, rows, contexts = rollout(
                cfg,
                perception,
                task,
                frames,
                params,
                method,
                action_seed,
                training=True,
                progress=lambda i: check_stop(root),
            )
            prediction, adjoint, _ = task.video(images, gradient=method == "E1")
            truth = manifest["files"][name]["ef"]
            task_loss = abs(prediction - truth)
            acquisition = sum(r["k1"] + r["k2"] for r in rows) / (112 * count)
            objective = task_loss + weight * acquisition
            diagnostics = {}
            if method == "E1":
                span = max(1, cfg["training"].get("temperature_updates",cfg["training"]["updates"]) - 1)
                temperature = cfg["training"]["temperature_start"] * (
                    cfg["training"]["temperature_end"] / cfg["training"]["temperature_start"]
                ) ** (min(update,span) / span)
                gradients, diagnostics = gs_gradient(
                    params,
                    contexts,
                    adjoint * np.sign(prediction - truth),
                    perception,
                    cfg,
                    temperature,
                    weight,
                )
                if (
                    update == 0
                    and task_loss > 1e-7
                    and not all(x > 0 for x in diagnostics["task_gradient_norms"].values())
                ):
                    raise RuntimeError("GS task gradients absent; do not train cost-only fallback")
            elif method == "E2":
                reference_error=0.0
                if cfg["training"].get("rl_reference")=="full_input":
                    reference_prediction,_,_=task.video(frames)
                    reference_error=abs(reference_prediction-truth)
                reward = -objective+reference_error
                # Baseline contains previous episodes only, so current reward is not subtracted from itself.
                advantage = reward - baseline
                scale=2*(cfg["training"]["clip_frames"]-1) if cfg["training"].get("rl_score_scale")=="fixed_decisions" else 1.0
                gradients = jax.grad(rl_loss)(params, contexts, advantage,scale)
                baseline = (
                    cfg["training"]["rl_baseline_decay"] * baseline
                    + (1 - cfg["training"]["rl_baseline_decay"]) * reward
                )
                diagnostics = dict(reward=reward, advantage=advantage, baseline=baseline,
                                   reference_error=reference_error,score_scale=scale,
                                   baseline_scope="full-input action-independent reference plus past-excess EMA" if cfg["training"].get("rl_reference")=="full_input" else "EMA of previous episode rewards")
            else:
                raise ValueError("Training only E1/E2")
            policy_stats={}
            if cfg.get("execution",{}).get("repair_milestones"):
                for i,head in enumerate(("first","second")):
                    values=np.stack([np.asarray(probabilities(params[head],c[f"state{i}"],c[f"legal{i}"])) for c in contexts if not c["cold"]])
                    policy_stats[head]=dict(mean_max_probability=float(values.max(1).mean()),
                                           mean_entropy=float(-(values*np.log(values+1e-20)).sum(1).mean()))
            params, optimizer, norm = adam(params, gradients, optimizer, cfg)
            record = dict(
                update=update + 1,
                case=name,
                start=first_frame,
                frames=count,
                action_seed=action_seed,
                ef_prediction=prediction,
                ef_truth=truth,
                ef_absolute_error=task_loss,
                acquisition_fraction=acquisition,
                objective=objective,
                gradient_norm=norm,
                mean_lines=acquisition * 112,
                zero_second_fraction=np.mean([r["k2"] == 0 for r in rows]),
                seconds=time.perf_counter() - start,
                **diagnostics,
                policy_stats=policy_stats,
                gradient_clipped=norm>cfg["training"]["gradient_clip"],
            )
            # Commit checkpoint last. A crash before it replays just this update and replaces its record.
            atomic_json(output / "updates" / f"{update + 1:05d}.json", record)
            save(output / "updates" / f"{update + 1:05d}.npz", params, optimizer, baseline)
            if audit_callback is not None:
                audit_callback(update + 1, images, rows, gradients)
            emit("train", method=method, seed=seed, cost_weight=weight, **record)
        save(output / "policy.npz", params, optimizer, baseline)
        atomic_json(
            output / "result.json",
            dict(
                status="completed",
                method=method,
                seed=seed,
                cost_weight=weight,
                updates=optimizer["step"],
                checkpoint=str(output / "policy.npz"),
                task_service_calls=task.calls,
                task_service_wall_s=task.seconds,
                perception_optimized=False,
                ef_optimized=False,
                gs_gradient_scope=cfg["training"]["gs_gradient"] if method == "E1" else None,
                restore_scope="policy, Adam, global update, reward baseline; per-update explicit RNG; "
                "resume at optimizer-update boundary; GPU numerical reproducibility unverified",
            ),
        )
    finally:
        task.close()


def balanced_schedule(total, frames, pairs, initial):
    """Outcome-independent periodic control matched post hoc to realized acquisition total."""
    if frames < 2:
        return [tuple(initial)]
    target = (total - sum(initial)) / (frames - 1)
    ordered = sorted(pairs, key=sum)
    lower = max((p for p in ordered if sum(p) <= target), key=sum, default=ordered[0])
    upper = min((p for p in ordered if sum(p) >= target), key=sum, default=ordered[-1])
    fraction = 0 if sum(upper) == sum(lower) else (target - sum(lower)) / (sum(upper) - sum(lower))
    return [tuple(initial)] + [
        tuple(upper if round((i + 1) * fraction) > round(i * fraction) else lower)
        for i in range(frames - 1)
    ]


def full_reference(cfg, task, root, name, frames, frame_limit=None):
    import cv2

    from ..provenance import sha256

    raw = Path(cfg["raw_videos"]) / (Path(name).stem + ".avi")
    suffix = "" if frame_limit is None else f".prefix{frame_limit}"
    file = root / ".cache" / "full_ef_reference" / f"{Path(name).stem}{suffix}.json"
    raw_hash = sha256(raw)
    if file.exists():
        value = read_json(file)
        if value["raw_sha256"] != raw_hash:
            raise ValueError("Original EF reference input changed")
        return value
    full, _, _ = task.video(frames)
    cap = cv2.VideoCapture(str(raw))
    originals = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        originals.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if frame_limit is not None and len(originals) == frame_limit:
            break
    cap.release()
    if len(originals) != len(frames) or any(x.shape != (112, 112, 3) for x in originals):
        raise ValueError("Original/polar time indices or input geometry differ")
    original, _, _ = task.video(np.stack(originals), domain="cartesian_rgb")
    value = dict(full_polar_ef=full, original_cartesian_ef=original, raw_sha256=raw_hash)
    atomic_json(file, value)
    return value


def evaluate(spec, cfg, manifest, root, output):
    if cfg["runtime"].get("evaluation_workers", 1) > 1 and "_case_names" not in spec:
        from .parallel import evaluate_parallel

        return evaluate_parallel(spec, cfg, manifest, root, output)
    perception, task = setup(cfg, output)
    params = (
        initialize(spec["seed"], cfg) if spec["method"] == "E0" or spec.get("untrained") else restore(spec["checkpoint"])[0]
    )
    records = []
    try:
        for name in spec.get("_case_names", manifest["cohorts"][spec["cohort"]]):
            check_stop(root)
            directory = output / Path(name).stem
            result_file = directory / "complete.json"
            if result_file.exists():
                records.append(read_json(result_file))
                continue
            start = time.perf_counter()
            frames = read_episode(cfg, manifest, name, count=spec.get("_calibration_frames"))
            io_s = time.perf_counter() - start
            start = time.perf_counter()
            seed = case_seed(spec["seed"], name)
            images, rows, _ = rollout(
                cfg,
                perception,
                task,
                frames,
                params,
                spec["method"],
                seed,
                fixed=spec.get("fixed", cfg["budgets"]["fixed"]),
                progress=lambda i: check_stop(root),
            )
            task_start = time.perf_counter()
            pred, _, clips = task.video(images)
            final_task_s = time.perf_counter() - task_start
            computation_s = time.perf_counter() - start
            reference_start = time.perf_counter()
            reference = full_reference(cfg, task, root, name, frames, frame_limit=spec.get("_calibration_frames"))
            full = reference["full_polar_ef"]
            truth = manifest["files"][name]["ef"]
            record = dict(
                case=name,
                seed=spec["seed"],
                frames=len(frames),
                ef_prediction=pred,
                ef_truth=truth,
                full_polar_ef=full,
                absolute_error=abs(pred - truth),
                full_input_absolute_error=abs(full - truth),
                original_cartesian_ef=reference["original_cartesian_ef"],
                original_input_absolute_error=abs(reference["original_cartesian_ef"] - truth),
                conversion_prediction_shift=abs(full - reference["original_cartesian_ef"]),
                prediction_preservation_error=abs(pred - full),
                total_lines=sum(r["k1"] + r["k2"] for r in rows),
                mean_lines=float(np.mean([r["k1"] + r["k2"] for r in rows])),
                mean_psnr=float(np.mean([r["psnr"] for r in rows])),
                mean_ssim=float(np.mean([r["ssim"] for r in rows])),
                mean_mae=float(np.mean([r["mae"] for r in rows])),
                perception_calls=sum(r["perception_calls"] for r in rows),
                seconds=computation_s,
                final_task_s=final_task_s,
                controller_s=sum(r["controller_s"] for r in rows),
                ef_selection_s=sum(r["score_s"] for r in rows),
                perception_s=sum(r["stage1_s"] + r["stage2_s"] for r in rows),
                io_s=io_s,
                full_input_reference_s=time.perf_counter() - reference_start,
                timing_scope="preloaded full-video acquisition/EF inference; diagnostic full-input EF excluded",
                clip_predictions=clips,
            )
            warm=[(r["k1"],r["k2"]) for r in rows[1:]]
            record["warm_budget_pairs"]=sorted(set(warm))
            record["budget_switch_rate"]=sum(a!=b for a,b in zip(warm,warm[1:]))/max(1,len(warm)-1)
            if spec["method"] != "E0":
                if cfg.get("execution",{}).get("repair_milestones"):
                    from .repair import joint_budget_schedule
                    schedule=joint_budget_schedule(rows)
                else:
                    schedule = balanced_schedule(
                        record["total_lines"],len(frames),cfg["budgets"]["fixed_sweep"],cfg["budgets"]["fixed"],
                    )
                t = time.perf_counter()
                identical=all((r["k1"],r["k2"])==tuple(k) for r,k in zip(rows,schedule))
                if identical:
                    matched_rows=rows;matched_pred=pred
                else:
                    matched_images, matched_rows, _ = rollout(
                        cfg,perception,task,frames,params,"E0",seed,fixed=schedule,
                        progress=lambda i: check_stop(root),
                    )
                    matched_pred, _, _ = task.video(matched_images)
                matched_total = sum(r["k1"] + r["k2"] for r in matched_rows)
                record["matched"] = dict(
                    ef_prediction=matched_pred,
                    absolute_error=abs(matched_pred - truth),
                    total_lines=matched_total,
                    mean_lines=matched_total / len(frames),
                    seconds=None if identical else time.perf_counter() - t,
                    mean_line_mismatch=abs(matched_total - record["total_lines"]) / len(frames),
                    scope="post-hoc resource-matched periodic fixed-budget control; "
                    "uses realized total only, no ground-truth/image/task-error information",
                    identical_hard_trajectory=identical,
                    timing_scope="not remeasured: identical control reused" if identical else "measured independent control",
                    joint_pair_histogram_matched=bool(cfg.get("execution",{}).get("repair_milestones")),
                )
                atomic_json(directory / "matched_trajectory.json", matched_rows)
            atomic_json(directory / "trajectory.json", rows)
            if spec.get("_calibration_save_full"):
                atomic_npz(directory / "calibration_images.npz", images=images)
            atomic_npz(
                directory / "snapshots.npz",
                indices=np.array([0, len(images) // 2, len(images) - 1]),
                reconstructions=images[[0, len(images) // 2, len(images) - 1]],
            )
            atomic_json(result_file, record)
            records.append(record)
            emit("evaluation", job=spec["id"], **record)
        atomic_json(
            output / ("pilot_result.json" if spec.get("_pilot") else "result.json"),
            dict(
                status="completed",
                method=spec["method"],
                cohort=spec["cohort"],
                seed=spec["seed"],
                cost_weight=spec.get("lambda"),
                fixed=spec.get("fixed"),
                records=records,
                task_service_wall_s=task.seconds,
            ),
        )
    finally:
        task.close()


def worker(spec, cfg, manifest, root, output):
    output.mkdir(parents=True, exist_ok=True)
    function = {"probe": probe, "train": train, "evaluate": evaluate}[spec["kind"]]
    if spec["kind"] == "probe":
        function(cfg, manifest, root, output)
    elif spec["kind"] == "train":
        function(spec,cfg,manifest,root,output,end_update=spec.get("end_update"))
    else:
        function(spec, cfg, manifest, root, output)
