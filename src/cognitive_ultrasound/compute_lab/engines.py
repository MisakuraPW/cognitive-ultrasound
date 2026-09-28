"""Framework-isolated engines. Preserve DPS input derivatives and owned outputs."""

import time

import numpy as np


class JaxEngine:
    def __init__(self, cfg, profile, budget, export):
        from ..preparation.casl import Adapter

        self.adapter = Adapter(dict(cfg, budget=budget, seed=42), profile.variant())
        self.jax = self.adapter.jax
        self.jax.config.update("jax_default_matmul_precision", "highest")
        self.profile, self.export = profile, export
        self.empty = self.jax.numpy.zeros((112, 112, 1))
        from ulsa.entropy import pixelwise_entropy

        self.entropy = lambda p: pixelwise_entropy(p[None, ..., 0], 1.0)[0]
        if profile.constants:
            # Hoist fixed sigma/shape and entropy expression into one cached executable.
            # Main sampler was already jitted/vmapped; this is only the diagnostic kernel.
            self.entropy = self.jax.jit(self.entropy)

    def reset(self, seed, initial=None):
        self.adapter.reset(seed)

    def snapshot(self):
        return {
            k: np.asarray(v, dtype=None if k == "resume_seed" else np.float32)
            for k, v in self.adapter.arrays().items()
        }

    def restore(self, arrays):
        self.adapter.restore(arrays)

    def capture(self):
        return self.adapter.clone()

    def reinstate(self, state):
        self.adapter.state = self.jax.tree_util.tree_map(lambda x: x, state)

    def noise(self):
        import keras
        from zea.func import split_seed

        state = self.adapter.state
        keys = split_seed(split_seed(state.seed, 3)[0], 2)
        return np.stack(
            [
                np.asarray(keras.random.normal((1, 112, 112, 3), seed=split_seed(k, 2)[0]))[0]
                for k in keys
            ]
        )

    def step(self, target, noise=None):
        import jax.numpy as jnp
        from ulsa.agent import hard_projection

        tick = time.perf_counter()
        state = self.adapter.state
        mask = state.mask[..., -1, None]
        if self.profile.transfers:
            obs = jnp.asarray(target) * mask
        else:
            obs = jnp.asarray(target * np.asarray(mask))
        self.jax.block_until_ready(obs)
        start = time.perf_counter()
        raw, following = self.adapter.agent.recover(obs, state)
        prediction = hard_projection(raw, obs)
        self.jax.block_until_ready((prediction, following))
        core = time.perf_counter() - start
        entropy = self.entropy(following.belief_distribution)
        arrays = dict(
            prediction=np.asarray(prediction, dtype=np.float32),
            mask=np.asarray(mask),
            uncertainty=np.asarray(entropy, dtype=np.float32),
            next_action=np.asarray(following.selected_lines),
        )
        self.adapter.state = following
        arrays.update(self.snapshot())
        self.jax.block_until_ready(entropy)
        return dict(core_s=core, closed_loop_s=time.perf_counter() - tick), arrays

    def peak(self):
        stats = self.jax.devices()[0].memory_stats() or {}
        return stats.get("peak_bytes_in_use")

    def operator_checks(self, directory):
        from ..torch_casl.reference import load_model, probes
        from .protocol import numeric

        model = load_model(self.adapter.cfg["checkpoint"])
        expected_dtype = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[
            self.profile.precision
        ]
        layers = [x for x in model.ema_network.layers if type(x).__name__ == "Conv2D"]
        actual = sorted({str(x.compute_dtype) for x in layers})
        if actual != [expected_dtype]:
            raise RuntimeError(
                f"Requested {expected_dtype}, actual convolution dtypes {actual}; no fallback"
            )
        probes(model, directory, storage_dtype=np.float32)
        with np.load(directory / "probes.npz") as a, np.load(self.export / "probes.npz") as b:
            result = {
                k: numeric(a[k], b[k], k == "selected")
                for k in ("prediction", "gradient", "entropy", "selected")
            }
        if any(v["max_abs"] is None for v in result.values()):
            raise FloatingPointError("Nonfinite operator/gradient probe")
        result["passed"] = all(v["passed"] for v in result.values())
        result["convolution_dtypes"] = actual
        self.probes = result
        return result


class TorchEngine:
    def __init__(self, cfg, profile, budget, export):
        import os

        # All compilation happens in this worker: compiler children cannot inherit an
        # open trajectory descriptor. This changes setup cost, not scientific settings.
        if profile.mode in ("compile", "graph"):
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch

        from ..preparation.common import read_json
        from ..provenance import sha256
        from ..torch_casl.native import FrozenGraph, NativeCASL
        from ..torch_casl.worker import validate_probes

        if not torch.cuda.is_available():
            raise RuntimeError("Torch GPU unavailable")
        if profile.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; fallback forbidden")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.reset_peak_memory_stats()
        self.torch, self.profile = torch, profile
        spec = read_json(export / "network.json")
        if sha256(export / "ema.npz") != spec["ema_sha256"]:
            raise ValueError("EMA export hash changed")
        network = FrozenGraph(export).to(device="cuda", memory_format=torch.channels_last)
        # Keep state, schedule, norm and gradients FP32. Autocast the network only.
        if profile.precision != "fp32":
            original = network.forward
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[profile.precision]

            def forward(*args):
                with torch.autocast("cuda", dtype=dtype):
                    value = original(*args)
                return value.float()

            network.forward = forward
        self.model = NativeCASL(network, budget).to(device="cuda")
        # Probes use 14-line selector irrespective of sequence budget.
        self.model.budget = 14
        self.probes = validate_probes(self.model, export, "cuda")
        if any(not np.isfinite(v["max_abs"]) for v in self.probes.values() if isinstance(v, dict)):
            raise FloatingPointError("Nonfinite Torch operator/input gradient")
        self.model.budget = budget
        if profile.mode in ("compile", "graph"):
            import torch._inductor.config as compiler_config

            compiler_config.compile_threads = 1
            compiler_config.worker_start_method = "spawn"
            torch.use_deterministic_algorithms(True)
            self.model.step_fn = torch.compile(self.model.dps_step, fullgraph=True)
            self.model.select = torch.compile(self.model.select, fullgraph=True)
            self.validate_compiled_gradient(export)
        self.graph = None

    def validate_compiled_gradient(self, export):
        from ..torch_casl.native import nchw

        torch = self.torch
        with np.load(export / "probes.npz") as data:
            base = tuple(
                nchw(data[k], "cuda") for k in ("x", "measurement", "mask", "noise", "signal")
            )
        for factor in (1.0, 0.73, 0.91):
            args = (base[0] * factor, base[1] * factor, *base[2:], base[3] * 0.9, base[4] * 1.01)
            expected = self.model.dps_step(*args)
            actual = self.model.step_fn(*args)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
        self.probes["compiled_input_gradient"] = True
        self.probes["internal_correctness"] = self.profile.mode == "compile"
        self.probes["deterministic_algorithms"] = True
        self.probes["compiler_threads"] = 1

    def reset(self, seed, initial=None):
        if initial is None:
            raise ValueError("Torch reset requires official initial mask and history")
        self.restore(initial)

    def restore(self, arrays):
        from ..torch_casl.native import nchw

        self.mask = nchw(np.asarray(arrays["resume_mask"])[None], "cuda")
        self.history = nchw(np.asarray(arrays["resume_buffer"])[None], "cuda")
        samples = arrays.get("resume_posterior_samples")
        self.previous = None if samples is None else nchw(samples, "cuda")

    def capture(self):
        return (
            self.mask.clone(),
            self.history.clone(),
            None if self.previous is None else self.previous.clone(),
        )

    def reinstate(self, state):
        self.mask, self.history, self.previous = (None if x is None else x.clone() for x in state)

    def snapshot(self):
        result = dict(
            resume_mask=self.mask[0].permute(1, 2, 0).cpu().numpy(),
            resume_buffer=self.history[0].permute(1, 2, 0).cpu().numpy(),
        )
        if self.previous is not None:
            result["resume_posterior_samples"] = self.previous.permute(0, 2, 3, 1).cpu().numpy()
        return result

    def step(self, target, noise):
        from ..torch_casl.native import CapturedFrame, nchw

        torch = self.torch
        tick = time.perf_counter()
        z = nchw(noise, "cuda")
        mask = self.mask[:, -1:]
        observation = nchw(target[None], "cuda") * mask
        self.history = torch.cat((self.history[:, 1:], observation), 1)
        cold = self.previous is None
        previous = torch.zeros_like(z) if cold else self.previous
        args = (self.history.expand(2, -1, -1, -1), self.mask, previous, z)
        setup = 0.0
        if not cold and self.profile.mode == "graph" and self.graph is None:
            begin = time.perf_counter()
            self.graph = CapturedFrame(self.model, args, self.profile.steps)
            from ..torch_casl.native import validate_capture

            validate_capture(self.graph, self.model, args, self.profile.steps)
            self.probes["capture_changed_inputs_and_owned_outputs"] = True
            self.probes["internal_correctness"] = True
            setup = time.perf_counter() - begin
        fn = (
            self.graph
            if not cold and self.graph is not None
            else (lambda *a: self.model.frame(*a, steps=self.profile.steps, cold=cold))
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        samples, predicted, action, entropy = fn(*args)
        torch.cuda.synchronize()
        core = time.perf_counter() - start
        self.previous = samples.detach().clone()
        selected_mask = action[None, None, None, :].expand(1, 1, 112, 112).float()
        self.mask = torch.cat((self.mask[:, 1:], selected_mask), 1)
        result = dict(
            prediction=predicted.detach().cpu().numpy()[..., None],
            mask=mask[0].permute(1, 2, 0).cpu().numpy(),
            uncertainty=entropy.detach().cpu().numpy(),
            next_action=action.cpu().numpy(),
        )
        result.update(self.snapshot())
        return dict(
            core_s=core, graph_setup_s=setup, closed_loop_s=time.perf_counter() - tick
        ), result

    def peak(self):
        return self.torch.cuda.max_memory_allocated()
