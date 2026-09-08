"""Official CASL recovery, with a policy boundary and synchronized profiling hooks."""

from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from time import perf_counter
from unittest.mock import patch

import numpy as np

from ..acquisition import OfficialSampler
from ..config import path
from ..official import activate

STRATEGIES = {"random": "uniform_random", "uniform": "equispaced", "casl": "greedy_entropy"}


class CASLLoop:
    def __init__(self, cfg, method, budget):
        activate("jax")
        import jax
        import keras
        from ulsa.agent import ActionSelectionConfig, AgentConfig, setup_agent
        from zea import Config

        self.cfg, self.method, self.budget = cfg, method, budget
        keras.mixed_precision.set_global_policy(cfg["precision"])
        agent_cfg = AgentConfig(
            io_config=Config({}),
            action_selection=ActionSelectionConfig(
                n_possible_actions=112,
                n_actions=budget,
                shape=[112, 112],
                selection_strategy=STRATEGIES[method],
                kwargs={"entropy_sigma": cfg["entropy_sigma"]} if method == "casl" else {},
            ),
            diffusion_inference=Config(
                {
                    "run_dir": str(path(cfg["checkpoint"])),
                    "num_steps": cfg["num_steps"],
                    "initial_step": cfg["initial_step"],
                    "batch_size": cfg["particles"],
                    "guidance_kwargs": {"omega": cfg["omega"]},
                    "reconstruction_method": cfg["reconstruction"],
                    "hard_project": cfg["hard_project"],
                }
            ),
        )
        self.agent, self.state = setup_agent(
            agent_cfg,
            jax.random.PRNGKey(cfg["seed"]),
            jit_mode="posterior_sample" if cfg["profile"] else "off",
        )
        if tuple(self.agent.input_shape) != (112, 112, cfg["temporal_window"]):
            raise ValueError(f"Unexpected checkpoint input shape {self.agent.input_shape}")
        if tuple(self.agent.input_range) != (-1, 1):
            raise ValueError("Checkpoint input range must be [-1,1]")
        self.times = {}
        kwargs = dict(self.agent.recover.keywords)
        self.sampler = OfficialSampler(kwargs["action_selection"])
        kwargs["action_selection"] = self.sampler.bridge
        if cfg["profile"]:
            kwargs["posterior_sample"] = self.timed("diffusion_s", kwargs["posterior_sample"])
            kwargs["action_selection"] = self.timed(
                "action_with_entropy_s", kwargs["action_selection"]
            )
        recovery = partial(self.agent.recover.func, **kwargs)
        self.agent = replace(self.agent, recover=recovery if cfg["profile"] else jax.jit(recovery))

    def sync(self, value):
        import jax

        return jax.block_until_ready(value)

    def timed(self, key, function):
        def wrapped(*args, **kwargs):
            self.sync((args, kwargs))
            start = perf_counter()
            value = function(*args, **kwargs)
            self.sync(value)
            self.times[key] = self.times.get(key, 0.0) + perf_counter() - start
            return value

        return wrapped

    def reset(self, seed):
        import jax
        from ulsa.agent import reset_agent_state

        self.state = reset_agent_state(self.agent, jax.random.PRNGKey(seed))

    def step(self, target):
        from keras import ops
        from ulsa.agent import hard_projection
        from ulsa.entropy import pixelwise_entropy
        from zea.agent.selection import GreedyEntropy

        self.times = {}
        target = ops.convert_to_tensor(target, dtype="float32")
        self.sync(target)
        start = perf_counter()
        mask = self.state.mask[..., -1, None]
        acquired_action = self.state.selected_lines
        observation = target * mask
        entropy_fn = GreedyEntropy.compute_pixelwise_entropy
        context = (
            patch.object(
                GreedyEntropy, "compute_pixelwise_entropy", self.timed("entropy_s", entropy_fn)
            )
            if self.cfg["profile"]
            else nullcontext()
        )
        with context:
            reconstruction, new_state = self.agent.recover(observation, self.state)
            self.sync((reconstruction, new_state))
        projection_start = perf_counter()
        if self.cfg["hard_project"]:
            reconstruction = hard_projection(reconstruction, observation)
        self.sync(reconstruction)
        self.times["projection_s"] = perf_counter() - projection_start
        self.times["total_s"] = perf_counter() - start
        if self.cfg["profile"]:
            self.times.setdefault("entropy_s", 0.0)
            self.times["action_s"] = max(
                0.0, self.times.pop("action_with_entropy_s") - self.times["entropy_s"]
            )
            self.times["overhead_s"] = max(
                0.0,
                self.times["total_s"]
                - sum(
                    self.times[k] for k in ("diffusion_s", "entropy_s", "action_s", "projection_s")
                ),
            )
        # This extra visualization entropy is NOT added to algorithm timing.
        entropy = pixelwise_entropy(
            new_state.belief_distribution[None, ..., 0], self.cfg["entropy_sigma"]
        )[0]
        self.state = new_state
        result = {
            "observation_mask": mask,
            "observation": observation,
            "reconstruction": reconstruction,
            "ground_truth": target,
            "posterior_samples": new_state.posterior_samples,
            "posterior_particles": new_state.belief_distribution,
            "measurement_buffer": new_state.measurement_buffer.buffer,
            "next_mask_buffer": new_state.mask,
            "seed": new_state.seed,
            "entropy_map": entropy,
            "acquired_action": acquired_action,
            "selected_action": new_state.selected_lines,
        }
        result = {k: np.asarray(v) for k, v in result.items()}
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError("Nonfinite official CASL state")
        # Record actual cardinality. Upstream greedy selection can repeat an index for zero entropy.
        result["actual_lines"] = int(np.count_nonzero(result["acquired_action"]))
        return result, dict(self.times)
