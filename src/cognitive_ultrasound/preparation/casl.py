"""Original CASL adapters, causal snapshots and bounded branch diagnostics."""

import json
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from ..config import load, path
from ..experiments import case_seed
from ..official import activate
from .common import (
    atomic_json,
    atomic_npz,
    emit,
    metrics,
    observable_features,
    read_frames,
    read_json,
)
from .sampling import VARIANTS, attach_sampler


class Adapter:
    def __init__(self, cfg, variant="reference", method="casl"):
        activate("jax")
        import jax
        import keras
        from ulsa.agent import ActionSelectionConfig, AgentConfig, get_operator_dict, setup_agent
        from zea import Config
        from zea.models.diffusion import DiffusionModel

        self.jax, self.cfg = jax, cfg
        if jax.default_backend() != "gpu":
            raise RuntimeError(
                "CASL preparation requires a visible GPU; CPU fallback is not allowed"
            )
        self.variant = VARIANTS[variant]
        v = self.variant
        keras.mixed_precision.set_global_policy(v["precision"])
        ac = AgentConfig(
            io_config=Config({}),
            action_selection=ActionSelectionConfig(
                n_possible_actions=112,
                n_actions=cfg["budget"],
                shape=[112, 112],
                selection_strategy={
                    "casl": "greedy_entropy",
                    "uniform": "equispaced",
                    "random": "uniform_random",
                }[method],
                kwargs={"entropy_sigma": 1.0} if method == "casl" else {},
            ),
            diffusion_inference=Config(
                dict(
                    run_dir=cfg["checkpoint"],
                    num_steps=500,
                    initial_step=v["initial_step"],
                    batch_size=2,
                    guidance_kwargs={"omega": 10},
                    reconstruction_method="choose_first",
                    hard_project=True,
                )
            ),
        )
        model = None
        if "guidance_steps" in v or "inference_steps" in v:
            model = DiffusionModel.from_preset(
                cfg["checkpoint"],
                guidance={"name": "dps", "params": {"disable_jit": True}},
                operator=get_operator_dict(ac),
            )
            attach_sampler(model, v)
        self.wrapper = None
        if v.get("wrapper"):
            from ..models.diffusion import CASLLoop

            base = load(path("configs/baseline.yaml"))
            base.update(checkpoint=cfg["checkpoint"], seed=cfg["seed"])
            self.wrapper = CASLLoop(base, method, cfg["budget"])
            self.agent, self.state = self.wrapper.agent, self.wrapper.state
        else:
            self.agent, self.state = setup_agent(
                ac,
                jax.random.PRNGKey(cfg["seed"]),
                jit_mode=v.get("jit_mode", "recover"),
                model=model,
            )
        if tuple(self.agent.input_shape) != (112, 112, 3):
            raise ValueError("Expected pinned 112x112, W=3 checkpoint")
        self.component_times = {}
        if variant == "profile":
            args = dict(self.agent.recover.keywords)
            for name in ("posterior_sample", "action_selection"):
                fn = args[name]

                def timed(*a, _fn=fn, _name=name, **k):
                    self.jax.block_until_ready((a, k))
                    t0 = time.perf_counter()
                    result = _fn(*a, **k)
                    self.jax.block_until_ready(result)
                    self.component_times[_name + "_s"] = time.perf_counter() - t0
                    return result

                args[name] = timed
            self.agent = replace(self.agent, recover=partial(self.agent.recover.func, **args))

    def reset(self, seed):
        from ulsa.agent import reset_agent_state

        self.state = reset_agent_state(self.agent, self.jax.random.PRNGKey(seed))

    def clone(self):
        # Rebuilds mutable FrameBuffer containers as well as AgentState.
        return self.jax.tree_util.tree_map(lambda a: a, self.state)

    def force_lines(self, lines):
        import jax.numpy as jnp

        lines = np.asarray(lines, dtype=int)
        if len(lines) != len(set(lines)) or len(lines) < 1 or min(lines) < 0 or max(lines) >= 112:
            raise ValueError("Invalid distinct action group")
        action = np.zeros(112, np.float32)
        action[lines] = 1
        newmask = self.state.mask.at[..., -1].set(jnp.broadcast_to(action, (112, 112)))
        self.state = replace(self.state, mask=newmask, selected_lines=jnp.asarray(action))

    def arrays(self):
        result = {"resume_buffer": np.asarray(self.state.measurement_buffer.buffer)}
        for name in (
            "mask",
            "seed",
            "selected_lines",
            "posterior_samples",
            "belief_distribution",
            "saliency_map",
        ):
            value = getattr(self.state, name)
            if value is not None:
                result["resume_" + name] = np.asarray(value)
        if self.state.pipeline_state:
            raise ValueError("Unexpected nonempty pipeline state; cannot serialize safely")
        return result

    def restore(self, archive):
        import jax.numpy as jnp

        self.reset(0)
        self.state.measurement_buffer.buffer = jnp.asarray(archive["resume_buffer"])
        values = {
            name: jnp.asarray(archive["resume_" + name]) if "resume_" + name in archive else None
            for name in (
                "mask",
                "seed",
                "selected_lines",
                "posterior_samples",
                "belief_distribution",
                "saliency_map",
            )
        }
        self.state = replace(self.state, **values)

    def step(self, target):
        import jax.numpy as jnp
        from ulsa.agent import hard_projection
        from ulsa.entropy import pixelwise_entropy

        started = time.perf_counter()
        mask = np.asarray(self.state.mask[..., -1, None])
        observation = jnp.asarray(target * mask)
        previous = (
            np.asarray(self.state.belief_distribution[0])
            if self.state.belief_distribution is not None
            else np.zeros_like(target)
        )
        cold = self.state.posterior_samples is None
        self.component_times = {}
        self.jax.block_until_ready(observation)
        t0 = time.perf_counter()
        raw, following = self.agent.recover(observation, self.state)
        self.jax.block_until_ready((raw, following))
        recover_s = time.perf_counter() - t0
        prediction = hard_projection(raw, observation)  # preserve official nonzero convention
        self.jax.block_until_ready(prediction)
        algorithm_s = time.perf_counter() - t0
        u = pixelwise_entropy(following.belief_distribution[None, ..., 0], 1.0)[0]
        u, raw, prediction = map(np.asarray, (u, raw, prediction))
        if not all(np.isfinite(x).all() for x in (raw, prediction, u)):
            raise FloatingPointError("Nonfinite CASL output")
        measured = mask.astype(bool)
        residual = float(np.abs(raw - np.asarray(observation))[measured].mean())
        self.state = following
        steps = (
            500 if cold else self.variant.get("inference_steps", 500 - self.variant["initial_step"])
        )
        gradients = 500 if cold else self.variant.get("guidance_steps", steps)
        row = {
            **metrics(target, prediction, mask),
            **observable_features(u, prediction, previous, residual, int(mask[0].sum())),
            "cold": cold,
            "algorithm_s": algorithm_s,
            "recover_s": recover_s,
            "adapter_wall_s": time.perf_counter() - started,
            "reverse_steps": steps,
            "dps_gradient_calls_per_particle": gradients,
            "particles": 2,
            **self.component_times,
        }
        arrays = dict(
            target=target,
            prediction=prediction,
            raw=raw,
            mask=mask,
            uncertainty=u,
            next_action=np.asarray(following.selected_lines),
        )
        return row, arrays


def run_trajectory(task, cfg, manifest, output):
    cohort, variant = task["cohort"], task["variant"]
    split = "train" if cohort == "train" else "val"
    names = manifest["cohorts"][cohort]
    frames = task.get("frames", cfg["frames"][cohort])
    seeds = task.get("seeds", cfg["seeds"][:1])
    adapter = None
    for seed in seeds:
        for name in names:
            directory = output / str(seed) / Path(name).stem
            if (directory / "complete.json").exists():
                continue
            if adapter is None:
                t0 = time.perf_counter()
                adapter = Adapter(cfg, variant)
                atomic_json(output / "load.json", dict(model_setup_s=time.perf_counter() - t0))
            directory.mkdir(parents=True, exist_ok=True)
            adapter.reset(case_seed(seed, name))
            start = 0
            for index in range(frames):
                if not (directory / f"frame_{index:04d}.npz").exists():
                    break
                start = index + 1
            if start:
                with np.load(
                    directory / f"frame_{start - 1:04d}.npz", allow_pickle=False
                ) as archive:
                    adapter.restore(archive)
            for index in range(start, frames):
                tick = time.perf_counter()
                target = read_frames(cfg, split, name, 1, index)[0]
                io_s = time.perf_counter() - tick
                row, arrays = adapter.step(target)
                # First two calls in this process/case may compile either cold/warm signature.
                row.update(
                    case=name,
                    frame=index,
                    seed=seed,
                    variant=variant,
                    cohort=cohort,
                    io_s=io_s,
                    timing_valid=index >= start + 2,
                )
                arrays.update(adapter.arrays())
                atomic_npz(
                    directory / f"frame_{index:04d}.npz", row=np.array(json.dumps(row)), **arrays
                )
                emit(
                    "frame",
                    job=task["id"],
                    case=name,
                    frame=index + 1,
                    total=frames,
                    psnr=row["psnr"],
                    seconds=row["algorithm_s"],
                )
            atomic_json(directory / "complete.json", dict(frames=frames, case=name))
    atomic_json(
        output / "result.json",
        dict(status="completed", variant=variant, cohort=cohort, cases=len(names), frames=frames),
    )


def candidate_actions(state, uncertainty, budget, seed):
    greedy = np.flatnonzero(np.asarray(state.selected_lines))
    uniform = np.linspace(0, 111, budget, dtype=int)
    random = np.sort(np.random.default_rng(seed).choice(112, budget, replace=False))
    scores = np.asarray(uncertainty).reshape(112, 112).mean(axis=0)
    alternative = np.argsort(-scores, kind="stable")[:budget]
    if len(greedy) != budget:
        # Do not silently repair the official selection in a supposedly matched comparison.
        raise ValueError("Original greedy action has wrong cardinality; branch study rejected")
    result = {}
    for label, lines in [
        ("greedy", greedy),
        ("uniform", uniform),
        ("random", random),
        ("topk", alternative),
    ]:
        key = tuple(sorted(lines.tolist()))
        if key not in [tuple(sorted(v)) for v in result.values()]:
            result[label] = lines.tolist()
    return result


def fixed_history(task, cfg, manifest, output, root):
    """Replay identical causal state, measurements and random key; no cold-start confound."""
    variant = task["variant"]
    adapter = Adapter(cfg, variant)
    records = []
    for name in manifest["cohorts"]["debug"]:
        source = root / "jobs/debug_reference" / str(cfg["seeds"][0]) / Path(name).stem
        files = sorted(source.glob("frame_*.npz"))
        for index in (2, min(6, len(files) - 2)):
            if index < 2 or index + 1 >= len(files):
                raise ValueError("Insufficient reference frames for common-history comparison")
            file = output / f"{Path(name).stem}_{index}.json"
            if file.exists():
                records.append(read_json(file))
                continue
            with np.load(files[index], allow_pickle=False) as archive:
                adapter.restore(archive)
            with np.load(files[index + 1], allow_pickle=False) as archive:
                baseline = json.loads(str(archive["row"]))
                expected = archive["next_action"].copy()
            row, arrays = adapter.step(read_frames(cfg, "val", name, 1, index + 1)[0])
            action = arrays["next_action"].astype(bool)
            expected = expected.astype(bool)
            record = dict(
                case=name,
                frame=index + 1,
                reference=baseline,
                candidate=row,
                next_action_jaccard=float(
                    np.count_nonzero(action & expected)
                    / max(1, np.count_nonzero(action | expected))
                ),
            )
            atomic_json(file, record)
            records.append(record)
    psnr = np.mean([r["candidate"]["psnr"] - r["reference"]["psnr"] for r in records])
    ssim = np.mean([r["candidate"]["ssim"] - r["reference"]["ssim"] for r in records])
    a = np.mean([r["reference"]["unobserved_mae"] for r in records])
    b = np.mean([r["candidate"]["unobserved_mae"] for r in records])
    passed = (
        psnr >= -cfg["gate"]["psnr_drop"]
        and ssim >= -cfg["gate"]["ssim_drop"]
        and b <= a * cfg["gate"]["mae_ratio"]
    )
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            passed=bool(passed),
            records=records,
            psnr_delta=psnr,
            ssim_delta=ssim,
            mae_ratio=b / max(a, 1e-12),
            caveat="Identical history diagnostic; action agreement is descriptive, not an optimality test",
        ),
    )


def closed_loop(task, cfg, manifest, output, root):
    """Conditional MVP: same spatial rule, past-only budget decisions, matched random schedule."""
    from .analysis import predict
    from .common import FEATURES

    fitted = read_json(root / "jobs/risk_probe/model.json")
    adapter = Adapter(cfg, fitted["variant"])
    budgets = cfg["branches"]["budgets"]
    if len(budgets) != 3:
        raise ValueError("This exploratory policy requires three budgets")
    records = []
    # Entire-case checkpoint: at most one tiny case/policy is recomputed on interruption.
    for name in manifest["cohorts"]["confirmation"][:2]:
        for seed in cfg["seeds"]:
            directory = output / str(seed) / Path(name).stem
            result_file = directory / "complete.json"
            if result_file.exists():
                records.append(read_json(result_file))
                continue
            frames = read_frames(cfg, "val", name, min(16, cfg["frames"]["confirmation"]))
            schedules, scores = {}, {}
            for policy in ("adaptive", "random_matched", "fixed"):
                adapter.reset(case_seed(seed, name))
                previous_row = None
                rows, schedule = [], []
                if policy == "random_matched":
                    shuffled = (
                        np.random.default_rng(seed).permutation(schedules["adaptive"][1:]).tolist()
                    )
                    scheduled = [cfg["budget"], *shuffled]
                for index, target in enumerate(frames):
                    k = cfg["budget"]
                    ood = False
                    if index and policy == "adaptive":
                        x = np.array([previous_row[f] for f in FEATURES])
                        ood = bool(
                            np.any(x < fitted["train_min"]) or np.any(x > fitted["train_max"])
                        )
                        if not ood:
                            risk = float(predict(fitted["model"], x[fitted["indices"]])[0])
                            k = budgets[int(np.searchsorted(fitted["thresholds"], risk))]
                    elif policy == "random_matched":
                        k = scheduled[index]
                    # Budget-only experiment: all arms use the same equispaced WHERE rule.
                    adapter.force_lines(np.linspace(0, 111, k, dtype=int))
                    row, arrays = adapter.step(target)
                    row.update(
                        frame=index, policy=policy, requested_lines=k, out_of_training_range=ood
                    )
                    previous_row = row
                    rows.append(row)
                    schedule.append(k)
                    atomic_npz(
                        directory / policy / f"frame_{index:04d}.npz",
                        row=np.array(json.dumps(row)),
                        **arrays,
                    )
                schedules[policy] = schedule
                scores[policy] = dict(
                    mean_unobserved_mae=float(np.mean([r["unobserved_mae"] for r in rows])),
                    total_lines=sum(r["actual_lines"] for r in rows),
                    algorithm_s=sum(r["algorithm_s"] for r in rows),
                    ood_fallback_frames=sum(r["out_of_training_range"] for r in rows),
                )
            record = dict(case=name, seed=seed, scores=scores, schedules=schedules)
            atomic_json(result_file, record)
            records.append(record)
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            records=records,
            caveat="Exploratory reuse of confirmation cases, not independent efficacy evidence; predictor trained under CASL is policy-shifted here; random arm matches adaptive total budget post hoc",
        ),
    )


def branches(task, cfg, manifest, output, run_root):
    chosen = read_json(run_root / "selection.json")["variant"]
    source = run_root / "jobs" / ("confirm_" + chosen)
    if not source.exists():
        source = run_root / "jobs" / ("debug_" + chosen)
    adapter = Adapter(cfg, chosen)
    records = []
    spec = cfg["branches"]
    names = (
        manifest["cohorts"]["confirmation"]
        if source.name.startswith("confirm")
        else manifest["cohorts"]["debug"]
    )
    for name in names[: spec["cases"]]:
        files = sorted((source / str(cfg["seeds"][0]) / Path(name).stem).glob("frame_*.npz"))
        usable = files[2:-2]
        if len(usable) < spec["states_per_case"]:
            raise ValueError("Insufficient completed trajectory for branch experiments")
        selected = [
            usable[i] for i in np.linspace(0, len(usable) - 1, spec["states_per_case"], dtype=int)
        ]
        for file in selected:
            with np.load(file, allow_pickle=False) as archive:
                adapter.restore(archive)
                index = json.loads(str(archive["row"]))["frame"]
                u = archive["uncertainty"].copy()
            parent = adapter.clone()
            targets = read_frames(cfg, "val", name, 2, index + 1)
            actions = candidate_actions(parent, u, cfg["budget"], cfg["seed"] + index)
            for kind, groups in [
                (
                    "budget",
                    {str(k): np.linspace(0, 111, k, dtype=int).tolist() for k in spec["budgets"]},
                ),
                ("action", actions),
            ]:
                for label, lines in groups.items():
                    for seed in spec["seeds"]:
                        token = f"{Path(name).stem}_{index}_{kind}_{label}_{seed}"
                        record_file = output / (token + ".json")
                        if record_file.exists():
                            records.append(read_json(record_file))
                            continue
                        adapter.state = adapter.jax.tree_util.tree_map(lambda a: a, parent)
                        adapter.state = replace(
                            adapter.state, seed=adapter.jax.random.PRNGKey(seed)
                        )
                        adapter.force_lines(lines)
                        first, _ = adapter.step(targets[0])
                        second = adapter.step(targets[1])[0] if kind == "action" else None
                        record = dict(
                            case=name,
                            frame=index,
                            kind=kind,
                            action=label,
                            seed=seed,
                            lines=lines,
                            one=first,
                            two=second,
                            hindsight_only=True,
                            common_history=True,
                        )
                        atomic_json(record_file, record)
                        records.append(record)
                        emit(
                            "branch",
                            job=task["id"],
                            case=name,
                            frame=index,
                            kind=kind,
                            action=label,
                        )
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            records=records,
            interpretation="Restricted hindsight diagnostic, not an executable policy or global planning bound",
        ),
    )
