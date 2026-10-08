"""Reuse official CASL posterior kernel; frame history shifts only once, outside kernel."""

import numpy as np


class CASLPerception:
    def __init__(self, cfg, cpu_functional_test=False):
        from ..official import activate

        activate("jax")
        import jax
        import jax.numpy as jnp
        import keras
        from ulsa.agent import ActionSelectionConfig, AgentConfig, setup_agent
        from zea import Config
        from zea.display import compute_scan_convert_2d_coordinates

        if not cpu_functional_test and jax.default_backend() != "gpu":
            raise RuntimeError("CASL GPU unavailable; production CPU fallback forbidden")
        jax.config.update("jax_default_matmul_precision", "highest")
        keras.mixed_precision.set_global_policy("float32")
        ac = AgentConfig(
            io_config=Config({}),
            action_selection=ActionSelectionConfig(
                n_possible_actions=112,
                n_actions=14,
                shape=[112, 112],
                selection_strategy="greedy_entropy",
                kwargs={"entropy_sigma": 1.0},
            ),
            diffusion_inference=Config(
                dict(
                    run_dir=cfg["checkpoint"],
                    num_steps=500,
                    initial_step=500 - cfg["perception"]["warm_steps"],
                    batch_size=2,
                    guidance_kwargs={"omega": cfg["perception"]["omega"]},
                    reconstruction_method="choose_first",
                    hard_project=True,
                )
            ),
        )
        agent, _ = setup_agent(ac, jax.random.PRNGKey(0), jit_mode="posterior_sample")
        if tuple(agent.input_shape) != (112, 112, 3):
            raise ValueError("Wrong official diffusion checkpoint shape")
        self.posterior = agent.recover.keywords["posterior_sample"]

        # Official inpainting uses a boolean where, which has no mask derivative.
        # Binary forward is retained exactly; the JVP is the declared fractional-mask
        # relaxation needed by GS. This modifies only this new experiment instance.
        @jax.custom_jvp
        def straight_through_observation(data, mask):
            return jnp.where(mask != 0, data, jnp.zeros_like(data))

        @straight_through_observation.defjvp
        def observation_jvp(primals, tangents):
            data, mask = primals
            ddata, dmask = tangents
            return straight_through_observation(data, mask), ddata * mask + data * dmask

        agent.operator.forward = straight_through_observation
        self.jax = jax
        self.coordinates = np.asarray(
            compute_scan_convert_2d_coordinates((112, 112), (0, 112), (-np.pi / 4, np.pi / 4), 1.0)[
                0
            ]
        )

        # Rematerialization bounds local-frame VJP memory; extra backward compute is recorded.
        def warm(history, masks, previous, key):
            return self.posterior(history, masks, previous, key)

        self.warm = jax.jit(jax.checkpoint(warm))
        self.zeros = jnp.zeros((112, 112, 3), jnp.float32)

    def infer(self, history, masks, previous, key, cold=False):
        import jax.numpy as jnp

        history, masks = jnp.asarray(history), jnp.asarray(masks)
        return (
            self.posterior(history, masks, None, key)
            if cold
            else self.warm(history, masks, jnp.asarray(previous), key)
        )
