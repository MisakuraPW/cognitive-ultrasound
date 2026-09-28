"""Finite sequence worker with shared-noise diagnostics and own-history rollouts."""

import json
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

from ..experiments import case_seed
from ..preparation.common import atomic_json, metrics, read_json
from ..provenance import sha256
from .data import Writer, frames
from .protocol import Profile, numeric


def cache_path(root, cohort, seed, budget, name):
    return root / ".cache" / cohort / f"b{budget}" / str(seed) / (Path(name).stem + ".h5")


def state_checks(actual, expected):
    names = (
        "prediction",
        "uncertainty",
        "next_action",
        "mask",
        "resume_mask",
        "resume_buffer",
        "resume_posterior_samples",
    )
    return {
        k: numeric(actual[k], expected[k], discrete=k in ("next_action", "mask", "resume_mask"))
        for k in names
    }


def read_group(h5, i):
    return {k: v[()] for k, v in h5[str(i)].items()}


def explicit_replay(engine, cfg, profile, budget, previous, target, noise):
    """RNG outside sampler boundary, identical observations/history/noise on both sides."""
    if profile.backend == "torch":
        engine.restore(previous)
        return engine.step(target, noise)[1]
    import jax
    import jax.numpy as jnp
    import keras

    from ..preparation.sampling import attach_sampler
    from ..torch_casl.reference import load_model, matched_frame

    if not hasattr(engine, "replay_kernel"):
        keras.mixed_precision.set_global_policy(profile.variant()["precision"])
        model = load_model(cfg["checkpoint"])
        if profile.dps != profile.steps:
            attach_sampler(model, profile.variant())
        engine.replay_kernel = jax.jit(matched_frame(model, budget), static_argnames=("steps",))
    mask = previous["resume_mask"][None]
    observation = target * mask[0, ..., -1, None]
    history = np.concatenate((previous["resume_buffer"][..., 1:], observation), -1)
    args = (np.repeat(history[None], 2, 0), mask, previous["resume_posterior_samples"], noise)
    samples, prediction, action, entropy = engine.replay_kernel(
        *map(jnp.asarray, args), steps=profile.steps
    )
    jax.block_until_ready(samples)
    return dict(
        prediction=np.asarray(prediction, dtype=np.float32)[..., None],
        next_action=np.asarray(action),
        uncertainty=np.asarray(entropy, dtype=np.float32),
        resume_posterior_samples=np.asarray(samples, dtype=np.float32),
    )


def run(task, cfg, root, output):
    from .engines import JaxEngine, TorchEngine

    profile = Profile(**task["profile"])
    cohort, budget = task["cohort"], task["budget"]
    manifest = read_json(root / "manifest.json")
    engine = None
    process = psutil.Process()
    all_rows, replay_records, micro = [], [], []
    peak_ram, load_s, task_start = 0, 0, time.perf_counter()
    short = task.get("short", False)
    for seed in manifest["seeds"][cohort]:
        for name in manifest["cohorts"][cohort]:
            directory = output / str(seed) / Path(name).stem
            receipt = directory / "complete.json"
            if receipt.exists():
                saved = read_json(receipt)
                all_rows.extend(saved["rows"])
                replay_records.extend(saved.get("replay", []))
                micro.extend(saved.get("micro", []))
                continue
            directory.mkdir(parents=True, exist_ok=True)
            count = manifest["files"][name]["frames"]
            if cohort != "confirmation":
                count = min(count, 4 if short else 128)
            if engine is None:
                start = time.perf_counter()
                cls = JaxEngine if profile.backend == "jax" else TorchEngine
                engine = cls(cfg, profile, budget, root / "export")
                load_s = time.perf_counter() - start
                if short and profile.backend == "jax":
                    (output / "operators").mkdir(exist_ok=True)
                    engine.operator_checks(output / "operators")
            ref_file = cache_path(root, cohort, seed, budget, name)
            is_ref = profile.name == "official"
            reference = None if is_ref else h5py.File(ref_file, "r")
            if not is_ref:
                expected_hash = ref_file.with_suffix(".sha256").read_text(encoding="ascii").strip()
                if sha256(ref_file) != expected_hash:
                    reference.close()
                    raise ValueError("Reference state/noise cache checksum mismatch")
            if is_ref:
                engine.reset(case_seed(seed, name))
                initial = engine.snapshot()
                ref_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = ref_file.with_suffix(".partial")
                writer = Writer(temporary, profile.async_output)
                group = writer.h5.create_group("initial")
                for key, value in initial.items():
                    group.create_dataset(key, data=value)
            else:
                initial = {k: v[()] for k, v in reference["initial"].items()}
                engine.reset(case_seed(seed, name), initial)
                # Same full trajectory write workload. Only reference cache is retained;
                # candidate scratch is verified/read below before deletion.
                temporary = directory / "trajectory.partial"
                writer = Writer(temporary, profile.async_output)
            rows, replays, micro_here = [], [], []
            before = time.perf_counter()
            previous_prediction, previous_target = None, None
            iterator = iter(frames(Path(cfg["data_root"]) / "val" / name, count, profile.io))
            try:
                for index in range(count):
                    t_read = time.perf_counter()
                    target = next(iterator)
                    io_s = time.perf_counter() - t_read
                    noise = engine.noise() if is_ref else reference[str(index)]["noise"][()]
                    previous = engine.snapshot()
                    previous_native = engine.capture() if short and index == 2 else None
                    timing, arrays = engine.step(target, noise)
                    if any(not np.isfinite(v).all() for v in arrays.values()):
                        raise FloatingPointError("Nonfinite output/gradient trajectory")
                    row = dict(
                        case=name,
                        seed=seed,
                        frame=index,
                        budget=budget,
                        io_s=io_s,
                        cold=index == 0,
                        warm_signature_first=index == 1,
                        **timing,
                        **metrics(target, arrays["prediction"], arrays["mask"]),
                    )
                    row["temporal_error"] = (
                        None
                        if index == 0
                        else float(
                            np.mean(
                                np.abs(
                                    (arrays["prediction"] - previous_prediction)
                                    - (target - previous_target)
                                )
                            )
                        )
                    )
                    previous_prediction, previous_target = arrays["prediction"], target
                    expected = arrays if is_ref else read_group(reference, index)
                    row["state_checks"] = state_checks(arrays, expected)
                    common = ~(arrays["mask"].astype(bool) | expected["mask"].astype(bool))
                    row["common_unobserved_mae"] = (
                        float(np.abs(target - arrays["prediction"])[common].mean())
                        if common.any()
                        else None
                    )
                    row["reference_common_unobserved_mae"] = (
                        float(np.abs(target - expected["prediction"])[common].mean())
                        if common.any()
                        else None
                    )
                    row["action_changed"] = not np.array_equal(
                        arrays["next_action"], expected["next_action"]
                    )
                    # Replay and fixed-input microbenchmark are outside sequence timing.
                    diagnostic_s = 0.0
                    if short and index == 2:
                        diagnostic_start = time.perf_counter()
                        own = engine.capture()
                        ref_previous = previous if is_ref else read_group(reference, index - 1)
                        replay = explicit_replay(
                            engine, cfg, profile, budget, ref_previous, target, noise
                        )
                        replay_keys = (
                            "prediction",
                            "next_action",
                            "uncertainty",
                            "resume_posterior_samples",
                        )
                        replay_record = dict(
                            case=name,
                            seed=seed,
                            frame=index,
                            rng_boundary="same external explicit noise on both sides; boundary drift recorded separately",
                            checks={
                                k: numeric(
                                    replay[k],
                                    replay[k] if is_ref else expected["replay_" + k],
                                    k == "next_action",
                                )
                                for k in replay_keys
                            },
                            rng_boundary_checks={
                                k: numeric(replay[k], expected[k], k == "next_action")
                                for k in replay_keys
                            },
                        )
                        if is_ref:
                            arrays.update({"replay_" + k: replay[k] for k in replay_keys})
                        replays.append(replay_record)
                        for repetition in range(23):
                            engine.reinstate(previous_native)
                            sample_t, sample_a = engine.step(target, noise)
                            if any(not np.isfinite(v).all() for v in sample_a.values()):
                                raise FloatingPointError("Nonfinite fixed-input microbenchmark")
                            if repetition >= 3:
                                micro_here.append(
                                    dict(case=name, repetition=repetition - 3, **sample_t)
                                )
                        engine.reinstate(own)
                        diagnostic_s = time.perf_counter() - diagnostic_start
                    row["diagnostic_s_excluded"] = diagnostic_s
                    rows.append(row)
                    t_write = time.perf_counter()
                    writer.append(dict(**arrays, noise=noise))
                    if index in (0, 2, count - 1):
                        from ..preparation.common import atomic_npz

                        atomic_npz(
                            directory / f"frame_{index:04d}.npz",
                            target=target,
                            prediction=arrays["prediction"],
                            mask=arrays["mask"],
                            uncertainty=arrays["uncertainty"],
                            next_action=arrays["next_action"],
                        )
                    row["output_enqueue_s"] = time.perf_counter() - t_write
                    peak_ram = max(peak_ram, process.memory_info().rss)
                    if index % 16 == 0 or index + 1 == count:
                        print(
                            json.dumps(
                                dict(
                                    event="frame",
                                    profile=profile.name,
                                    cohort=cohort,
                                    budget=budget,
                                    case=name,
                                    seed=seed,
                                    frame=index + 1,
                                    total=count,
                                    seconds=row["closed_loop_s"],
                                )
                            ),
                            flush=True,
                        )
            finally:
                try:
                    writer.close()
                finally:
                    if reference is not None:
                        reference.close()
                    iterator.close()
            if is_ref:
                temporary.replace(ref_file)
                ref_file.with_suffix(".sha256").write_text(
                    sha256(ref_file) + "\n", encoding="ascii"
                )
            else:
                with h5py.File(temporary) as check:
                    if len(check) != count:
                        raise ValueError("Output writer lost trajectory frames")
                temporary.unlink()
            record = dict(
                rows=rows,
                replay=replays,
                micro=micro_here,
                sequence_wall_s=time.perf_counter()
                - before
                - sum(r["diagnostic_s_excluded"] for r in rows),
                cache_output_included=True,
                retained_reference_cache=is_ref,
                vram_peak_bytes=engine.peak(),
                ram_peak_bytes=peak_ram,
            )
            atomic_json(receipt, record)
            if hasattr(engine, "probes"):
                atomic_json(output / "operator_checks.json", engine.probes)
            all_rows.extend(rows)
            replay_records.extend(replays)
            micro.extend(micro_here)
    checks = [c for r in all_rows for c in r["state_checks"].values()]
    numerical = bool(checks and all(c["passed"] for c in checks))
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            profile=profile.record(),
            cohort=cohort,
            budget=budget,
            rows=all_rows,
            replay=replay_records,
            micro=micro,
            equivalence_passed=numerical,
            bitwise=bool(checks and all(c["bitwise"] for c in checks)),
            operator_checks=(
                getattr(engine, "probes", None)
                if engine
                else read_json(output / "operator_checks.json")
                if (output / "operator_checks.json").exists()
                else None
            ),
            model_load_s=load_s,
            task_wall_s=time.perf_counter() - task_start,
            ram_peak_bytes=peak_ram,
            vram_peak_bytes=engine.peak() if engine else None,
            closed_loop_s=sum(r["closed_loop_s"] for r in all_rows),
            lpips="not_evaluated: verified protocol not configured",
            task_metrics="not_evaluated: verified labels/protocol not configured",
        ),
    )


def export(cfg, output):
    from ..torch_casl.reference import export_network, load_model, probes

    model = load_model(cfg["checkpoint"])
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    export_network(model, cfg["checkpoint"], output)
    probes(model, output)
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            particles="already vmapped upstream",
            new_particle_vectorization=False,
        ),
    )
