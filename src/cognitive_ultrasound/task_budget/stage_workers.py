"""Actual whole-video hard rollouts, independent objectives and committed updates."""

import time
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from ..preparation.common import atomic_json, digest, read_json
from .data import read_episode
from .policy import adam, adam_state
from .repair import joint_budget_schedule
from .stage_policy import Controller, initialize
from .stage_protocol import ARMS, checkpoint_choice, dual_update, training_objective
from .value_protocol import uniform
from .value_runtime import Runtime, trajectory
from .value_storage import commit, committed, load_tree, save_tree, tree_digest


def metrics(r, prediction, truth, rows):
    return dict(
        ef_prediction=prediction,
        ef_error=abs(prediction - truth),
        mse=r["quality"]["mse"],
        p90=r["quality"]["frame_mse_p90"],
        psnr=r["psnr"],
        ssim=r["ssim"],
        mean_lines=r["mean_lines"],
        warm_mean_lines=float(np.mean([x["k1"] + x["k2"] for x in rows[1:]])),
        total_lines=r["total_lines"],
        perception_calls=r["perception_calls"],
        inference_seconds=r["inference_seconds"],
        pairs={
            f"{a}+{b}": n for (a, b), n in Counter((x["k1"], x["k2"]) for x in rows[1:]).items()
        },
        zero_second_fraction=float(np.mean([x["k2"] == 0 for x in rows[1:]])),
    )


def reference(runtime, name, seed, pair=(7, 7), count=None):
    n = runtime.manifest["files"][name]["frames"] if count is None else count
    images, masks, rows, r, directory = trajectory(
        runtime, name, seed, uniform(n, pair), tag=f"fixed_{pair[0]}_{pair[1]}", frames_limit=count
    )
    target = read_episode(runtime.cfg, runtime.manifest, name, count=count)
    truth = runtime.manifest["files"][name]["ef"]
    full = runtime.readout(target, truth, condition="full_input")
    readouts = runtime.readout(images, truth, reference=full, condition="fixed_readout")
    x = metrics(r, readouts["all_starts_v1"]["prediction"], truth, rows)
    x.update(
        full_ef_error=full["all_starts_v1"]["absolute_error"],
        preservation_error=readouts["all_starts_v1"]["preservation_error"],
        coverage=readouts["all_starts_v1"]["coverage"],
        trajectory=str(directory),
    )
    return x


def policy_episode(runtime, arm, params, name, seed, count=None):
    n = runtime.manifest["files"][name]["frames"] if count is None else count
    controller = Controller(params, runtime.cfg, arm, n)
    images, masks, rows, r, directory = trajectory(
        runtime, name, seed, uniform(n, (7, 7)), tag=arm, frames_limit=count, controller=controller
    )
    truth = runtime.manifest["files"][name]["ef"]
    task_readouts = runtime.readout(images, truth, condition=f"{arm}_readout")
    x = metrics(r, task_readouts["all_starts_v1"]["prediction"], truth, rows)
    x.update(
        coverage=task_readouts["all_starts_v1"]["coverage"],
        trajectory=str(directory),
        frame_mse=r["quality"]["frame_mse"],
        quota_forced_fraction=float(np.mean([t["quota_forced"] for t in controller.trace])),
        quota_constrained_fraction=float(
            np.mean([t["quota_constrained"] for t in controller.trace])
        ),
        probability_max=float(np.mean([t["max_probability"] for t in controller.trace])),
    )
    if arm != "Q2" and (x["total_lines"] != 14 * n or controller.remaining != 0):
        raise AssertionError("Temporal policy changed total acquisition budget")
    return x, rows, controller


def checkpoint(root, arm, update):
    p = root / "training" / arm / "updates" / f"{update:05d}"
    return p


def checkpoint_state(runtime, arm, update):
    directory = checkpoint(runtime.root, arm, update)
    value = committed(directory, dict(batch=runtime.identity, arm=arm, update=update))
    if value is None:
        raise FileNotFoundError(f"Required committed checkpoint: {directory}")
    return load_tree(directory / "state.npz")


def train(runtime, arm, end):
    cfg, root = runtime.cfg, runtime.root
    d = cfg["stage_budget"]
    params = initialize(d["training_seed"], cfg, arm)
    state = dict(
        params=params,
        optimizer=adam_state(params),
        baseline=0.0,
        task_baseline=0.0,
        dual=np.array([1.0, 1.0]),
        update=0,
        frozen_initial=tree_digest(params),
    )
    complete = [
        p
        for p in (root / "training" / arm / "updates").glob("*/complete.json")
        if int(p.parent.name) <= end
    ]
    if complete:
        state = checkpoint_state(runtime, arm, max(int(p.parent.name) for p in complete))
    for i in range(state["update"], end):
        runtime.stopped()
        entry = runtime.manifest["curriculum"][i]
        name = entry["case"]
        seed = d["training_seed"] + entry["visit"] * 100003
        started = time.monotonic()
        # Same explicit case/visit noise across arms; references cached once per visit.
        ref = reference(runtime, name, seed)
        x, rows, controller = policy_episode(runtime, arm, state["params"], name, seed)
        objective, violation = training_objective(arm, x, ref, state["dual"], cfg)
        reward = -objective
        advantage = reward - state["baseline"]
        gradients = controller.gradients(advantage)
        task_reward = reward if violation is None else -float(np.asarray(state["dual"]) @ violation)
        task_gradients = (
            gradients
            if violation is None
            else controller.gradients(task_reward - state["task_baseline"])
        )
        task_norm = float(
            np.sqrt(
                sum(
                    np.sum(np.asarray(g, dtype=np.float64) ** 2)
                    for g in jax.tree_util.tree_leaves(task_gradients)
                )
            )
        )
        if not np.isfinite(task_norm):
            raise FloatingPointError("Nonfinite objective-specific task gradient")
        updated, optimizer, norm = adam(state["params"], gradients, state["optimizer"], cfg)
        dual = (
            dual_update(state["dual"], violation, cfg) if violation is not None else state["dual"]
        )
        record = dict(
            arm=arm,
            update=i + 1,
            case=name,
            visit=entry["visit"],
            section=entry["section"],
            cohort="train",
            objective=objective,
            reward=reward,
            advantage=advantage,
            gradient_norm=norm,
            task_gradient_norm=task_norm,
            task_reward=task_reward,
            dual_before=np.asarray(state["dual"]).tolist(),
            dual_after=np.asarray(dual).tolist(),
            violations=None if violation is None else violation.tolist(),
            seconds=time.monotonic() - started,
            **x,
        )
        new = dict(
            params=updated,
            optimizer=optimizer,
            baseline=cfg["training"]["rl_baseline_decay"] * state["baseline"]
            + (1 - cfg["training"]["rl_baseline_decay"]) * reward,
            task_baseline=cfg["training"]["rl_baseline_decay"] * state["task_baseline"]
            + (1 - cfg["training"]["rl_baseline_decay"]) * task_reward,
            dual=dual,
            update=i + 1,
            frozen_initial=state["frozen_initial"],
        )
        directory = checkpoint(root, arm, i + 1)
        atomic_json(directory / "record.json", record)
        save_tree(directory / "state.npz", new)
        commit(
            directory,
            dict(batch=runtime.identity, arm=arm, update=i + 1),
            record,
            ["record.json", "state.npz"],
        )
        state = new
        runtime.progress(
            arm=arm,
            operation="TRAIN_UPDATE",
            update=i + 1,
            end_update=end,
            case=name,
            mean_lines=x["mean_lines"],
            ef_error=x["ef_error"],
            gradient_norm=norm,
            objective=objective,
        )
    return dict(
        arm=arm,
        update=state["update"],
        parameters_changed=tree_digest(state["params"]) != state["frozen_initial"],
        frozen_casl=True,
        frozen_ef=True,
        objective="EF_exact_budget" if arm != "Q2" else "reconstruction_constrained_acquisition",
        checkpoint=str(checkpoint(root, arm, end)),
        restore_scope="GRU, Adam, global curriculum index, past reward EMA, quality duals; explicit case/visit RNG; trajectory chunks restore posterior/history/controller/RNG",
    )


def evaluate_policy(runtime, arm, update, names, seeds, cohort, controls):
    params = checkpoint_state(runtime, arm, update)["params"]
    results = []
    for seed in seeds:
        for name in names:
            unit = dict(
                batch=runtime.identity,
                arm=arm,
                update=update,
                case=name,
                seed=seed,
                controls=controls,
                cohort=cohort,
            )
            directory = runtime.root / "evaluations" / digest(unit)[:24]
            previous = committed(directory, unit)
            if previous is not None:
                results.append(previous)
                continue
            runtime.progress(
                arm=arm,
                operation="EVALUATE_CASE",
                update=update,
                case=name,
                seed=seed,
                cohort=cohort,
            )
            ref = reference(runtime, name, seed)
            x, rows, controller = policy_episode(runtime, arm, params, name, seed)
            x.update(
                arm=arm,
                update=update,
                case=name,
                seed=seed,
                cohort=cohort,
                mse_ratio=(x["mse"] - ref["mse"])
                / max(ref["mse"], runtime.cfg["stage_budget"]["mse_floor"]),
                p90_ratio=(x["p90"] - ref["p90"])
                / max(ref["p90"], runtime.cfg["stage_budget"]["mse_floor"]),
                ratio_floor_applied=bool(
                    min(ref["mse"], ref["p90"]) < runtime.cfg["stage_budget"]["mse_floor"]
                ),
                ef_vs_fixed=x["ef_error"] - ref["ef_error"],
                reference=ref,
                policy_state_inputs="causal observed-state only; no EF truth/full frame/future pixels",
                policy_mode="sampled_main",
                training_seed=runtime.cfg["stage_budget"]["training_seed"],
            )
            if controls:
                schedule = joint_budget_schedule(rows)
                assert int(schedule.sum()) == x["total_lines"]
                if np.array_equal(schedule, np.array([[r["k1"], r["k2"]] for r in rows])):
                    other = dict(x)
                    other.pop("reference", None)
                    other.pop("periodic", None)
                    matched_reused = True
                else:
                    images, masks, cyclic, r, directory2 = trajectory(
                        runtime, name, seed, schedule, tag=f"{arm}_matched_periodic"
                    )
                    y = runtime.readout(
                        images,
                        runtime.manifest["files"][name]["ef"],
                        condition="matched_periodic_readout",
                    )
                    other = metrics(
                        r,
                        y["all_starts_v1"]["prediction"],
                        runtime.manifest["files"][name]["ef"],
                        cyclic,
                    )
                    matched_reused = False
                if (
                    other["total_lines"] != x["total_lines"]
                    or other["perception_calls"] != x["perception_calls"]
                    or other["pairs"] != x["pairs"]
                ):
                    raise AssertionError("Joint acquisition/call matching failed")
                x.update(
                    periodic=other,
                    periodic_reused=matched_reused,
                    ef_vs_periodic=x["ef_error"] - other["ef_error"],
                    mse_vs_periodic=x["mse"] - other["mse"],
                )
            import csv

            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "budget_trace.csv").open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "k1", "k2", "mse", "perception_calls"])
                w.writeheader()
                w.writerows(
                    dict(
                        frame=r["frame"],
                        k1=r["k1"],
                        k2=r["k2"],
                        mse=err,
                        perception_calls=r["perception_calls"],
                    )
                    for r, err in zip(rows, x["frame_mse"])
                )
            x["budget_trace"] = str(directory / "budget_trace.csv")
            commit(directory, unit, x, ["budget_trace.csv"])
            results.append(x)
    return results


def worker(phase, cfg, manifest, root):
    runtime = Runtime(cfg, manifest, root, phase)
    output = root / "jobs" / phase
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        d = cfg["stage_budget"]
        pilot_end = d["pilot_train"] * d["pilot_passes"]
        if phase == "probe":
            result = probe(runtime)
        elif phase in ["pilot_baselines", "full_baselines", "confirmation_baselines"]:
            cohort = "confirmation" if phase.startswith("confirmation") else "development"
            names = (
                manifest["pilot_development"]
                if phase.startswith("pilot")
                else manifest["cohorts"][cohort]
            )
            seeds = d["evaluation_seeds"] if cohort == "confirmation" else [42]
            values = []
            for seed in seeds:
                for name in names:
                    for pair in d["fixed_controls"]:
                        runtime.progress(operation="BASELINE", case=name, seed=seed, pair=pair)
                        values.append(
                            dict(
                                case=name,
                                seed=seed,
                                cohort=cohort,
                                pair=pair,
                                **reference(runtime, name, seed, tuple(pair)),
                            )
                        )
            result = dict(cases=values)
        elif phase.endswith("_train"):
            end = (
                pilot_end
                if phase.startswith("pilot")
                else manifest["pass_ends"][phase.split("_")[2]]
            )
            result = dict(models=[train(runtime, arm, end) for arm in ARMS])
        elif phase.endswith("_evaluate"):
            end = (
                pilot_end
                if phase.startswith("pilot")
                else manifest["pass_ends"][phase.split("_")[2]]
            )
            names = (
                manifest["pilot_development"]
                if phase.startswith("pilot")
                else manifest["cohorts"]["development"]
            )
            result = dict(
                cases=[
                    x
                    for arm in ARMS
                    for x in evaluate_policy(
                        runtime, arm, end, names, [42], "development", phase.startswith("pilot")
                    )
                ]
            )
        elif phase == "select":
            all_rows = []
            for e in [1, 2, 4]:
                all_rows.extend(
                    read_json(root / "jobs" / f"full_pass_{e}_evaluate" / "result.json")["cases"]
                )
            result = dict(
                selection={
                    arm: checkpoint_choice(arm, [x for x in all_rows if x["arm"] == arm], cfg)
                    for arm in ARMS
                }
            )
        elif phase == "confirmation_policies":
            selection = read_json(root / "jobs/select/result.json")["selection"]
            result = dict(
                cases=[
                    x
                    for arm in ARMS
                    for x in evaluate_policy(
                        runtime,
                        arm,
                        selection[arm]["chosen"]["update"],
                        manifest["cohorts"]["confirmation"],
                        d["evaluation_seeds"],
                        "confirmation",
                        True,
                    )
                ]
            )
        else:
            raise ValueError(f"Unknown phase: {phase}")
        result.update(
            status="completed",
            phase=phase,
            seconds=time.monotonic() - started,
            functional_fixture=d["functional_fixture"],
        )
        atomic_json(output / "result.json", result)
        commit(output, dict(batch=runtime.identity, phase=phase), result, ["result.json"])
    finally:
        if runtime.task is not None:
            runtime.task.close()


def probe(runtime):
    """Short real-model test, including one recurrent policy gradient per arm."""
    cfg, m = runtime.cfg, runtime.manifest
    name = m["pilot_development"][0]
    count = min(64, m["files"][name]["frames"])
    ref = reference(runtime, name, 42, count=count)
    models = []
    for arm in ARMS:
        params = initialize(cfg["stage_budget"]["training_seed"], cfg, arm)
        x, rows, controller = policy_episode(runtime, arm, params, name, 42, count=count)
        g = controller.gradients(1.0)
        _, _, norm = adam(params, g, adam_state(params), cfg)
        if norm <= 0 or len(controller.trace) != count - 1:
            raise AssertionError("No genuine warm-frame policy gradient")
        models.append(
            dict(
                arm=arm,
                gradient_norm=norm,
                trace_frames=len(controller.trace),
                total_lines=x["total_lines"],
                quota_valid=arm == "Q2" or x["total_lines"] == 14 * count,
            )
        )
        if arm == "T2":
            directory = Path(x["trajectory"])
            with np.load(directory / "images.npz", allow_pickle=False) as z:
                original_images, original_masks = z["images"].copy(), z["masks"].copy()
            at = cfg["stage_budget"]["checkpoint_frames"]
            saved = load_tree(directory / "chunks" / f"{at:05d}" / "chunk.npz")["state"]
            restored = Controller(params, cfg, arm, count)
            replay, replay_masks, replay_rows, _, _ = trajectory(
                runtime,
                name,
                42,
                uniform(count, (7, 7)),
                initial=saved,
                prefix=(original_images[:at], original_masks[:at], rows[:at]),
                controller=restored,
                frames_limit=count,
                tag="recovery_probe",
            )
            np.testing.assert_allclose(replay, original_images, atol=2e-4, rtol=2e-4)
            np.testing.assert_array_equal(replay_masks, original_masks)
            assert [[r["lines1"], r["lines2"]] for r in replay_rows] == [
                [r["lines1"], r["lines2"]] for r in rows
            ]
            assert tree_digest(restored.trace) == tree_digest(controller.trace)
    warm_s = ref["inference_seconds"] / count
    total_updates = len(m["curriculum"])
    train_frames = sum(m["files"][x["case"]]["frames"] for x in m["curriculum"])
    return dict(
        models=models,
        production_gpu=not cfg["stage_budget"]["functional_fixture"],
        estimates=dict(
            measured_frame_s=warm_s,
            updates_per_arm=total_updates,
            all_three_training_hours=train_frames * warm_s * 4 / 3600,
            note="Three learned arms plus shared fixed training references; excludes repeated cold setup, EF readouts, gradients, evaluation and I/O. Not a guarantee.",
        ),
    )
