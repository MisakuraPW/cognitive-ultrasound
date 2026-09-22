"""Frozen native Torch graph exported from the official Keras EMA network.

All runtime tensors use NCHW. No Keras/JAX imports or calls in this module.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrozenGraph(nn.Module):
    def __init__(self, folder):
        super().__init__()
        folder = Path(folder)
        self.spec = json.loads((folder / "network.json").read_text(encoding="utf-8"))
        self.nodes = self.spec["nodes"]
        self.layers = nn.ModuleDict()
        with np.load(folder / "ema.npz", allow_pickle=False) as arrays:
            for node in self.nodes:
                key, kind, cfg = node["name"], node["kind"], node["config"]
                weights = [torch.from_numpy(arrays[k].copy()) for k in node["weights"]]
                if kind == "Conv2D":
                    kernel, bias = weights
                    outc, inc = kernel.shape[-1], kernel.shape[-2]
                    layer = nn.Conv2d(
                        inc,
                        outc,
                        cfg["kernel_size"],
                        padding="same" if cfg["padding"] == "same" else 0,
                    )
                    layer.weight.data.copy_(kernel.permute(3, 2, 0, 1))
                    layer.bias.data.copy_(bias)
                    self.layers[key] = layer
                elif kind == "BatchNormalization":
                    mean, var = weights  # official center=False, scale=False
                    layer = nn.BatchNorm2d(len(mean), eps=cfg["epsilon"], affine=False)
                    layer.running_mean.copy_(mean)
                    layer.running_var.copy_(var)
                    self.layers[key] = layer
        self.eval().requires_grad_(False)
        kw = self.spec["model_config"]["network_kwargs"]
        lo, hi = kw.get("embedding_min_frequency", 1.0), kw.get("embedding_max_frequency", 1000.0)
        dims = kw.get("embedding_dims", 32)
        frequencies = torch.exp(torch.linspace(math.log(lo), math.log(hi), dims // 2))
        self.register_buffer("angular", (2 * math.pi * frequencies).reshape(1, -1, 1, 1))

    def forward(self, image, variance):
        values = dict(zip(self.spec["inputs"], (image, variance)))
        for node in self.nodes:
            kind, key, cfg = node["kind"], node["name"], node["config"]
            if kind == "InputLayer":
                continue
            args = [values[n] for n in node["inputs"]]
            x = args[0]
            if kind == "Conv2D":
                out = self.layers[key](x)
                if cfg["activation"] in ("swish", "silu"):
                    out = F.silu(out)
            elif kind == "BatchNormalization":
                out = self.layers[key](x)
            elif kind == "Lambda":
                phase = self.angular * x
                out = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
            elif kind == "UpSampling2D":
                mode = cfg["interpolation"]
                out = F.interpolate(
                    x,
                    scale_factor=tuple(cfg["size"]),
                    mode=mode,
                    **({"align_corners": False} if mode == "bilinear" else {}),
                )
            elif kind == "AveragePooling2D":
                out = F.avg_pool2d(x, tuple(cfg["pool_size"]), tuple(cfg["strides"]))
            elif kind == "Concatenate":
                out = torch.cat(args, dim=1)
            elif kind == "Add":
                out = args[0] + args[1]
            else:
                raise ValueError(f"Unsupported exported layer: {kind}")
            values[key] = out
        return values[self.spec["output"]]


class NativeCASL(nn.Module):
    """Per-particle DPS norm, deterministic DDIM, official warm start and selection."""

    def __init__(self, network, budget=14, total_steps=500):
        super().__init__()
        self.network = network
        cfg = network.spec["model_config"]
        self.start_angle = math.acos(cfg["max_signal_rate"])
        self.end_angle = math.acos(cfg["min_signal_rate"])
        self.max_t = cfg.get("max_t", 1.0)
        self.budget, self.total_steps = budget, total_steps
        self.register_buffer("columns", torch.arange(112))
        self.register_buffer("reweight", 1 - torch.exp(-0.5 * torch.arange(-2.0, 3.0) ** 2))
        # torch.func captures input gradients without building a training graph across DDIM steps.
        self.grad_fn = torch.func.grad_and_value(self.loss, has_aux=True)
        self.step_fn = self.dps_step

    def rates(self, time):
        angle = self.start_angle + time * (self.end_angle - self.start_angle)
        return angle.sin(), angle.cos()

    def loss(self, x, measurement, mask, noise, signal):
        pn = self.network(x, noise.square())
        pi = (x - noise * pn) / signal
        error = measurement - torch.where(mask.bool(), pi, 0.0)
        # SUM of individual norms, not one norm over all particles (matches JAX vmap).
        loss = 10.0 * error.square().flatten(1).sum(1).sqrt().sum()
        return loss, (pn, pi)

    def dps_step(self, x, measurement, mask, noise, signal, next_noise, next_signal):
        gradient, (_, (pn, pi)) = self.grad_fn(x, measurement, mask, noise, signal)
        return (next_signal * pi + next_noise * pn - gradient).detach(), (pi - gradient).detach()

    def select(self, samples):
        pixels = samples[:, -1]  # particles, height, width
        pairwise = torch.exp(-((pixels[:, None] - pixels[None, :]) ** 2) / 2.0)
        entropy = -pairwise.mean(0).log().mean(0)
        scores = entropy.sum(0)
        selected = torch.zeros_like(scores, dtype=torch.bool)
        for _ in range(self.budget):
            best = scores.argmax()
            selected = selected | (self.columns == best)
            distance = self.columns - best
            factor = self.reweight[(distance + 2).clamp(0, 4)]
            scores = scores * torch.where(distance.abs() <= 2, factor, 1.0)
        return selected, entropy

    def frame(self, measurement, mask, previous, initial_noise, steps=50, cold=False):
        initial = 0 if cold else self.total_steps - steps
        base = torch.ones((len(initial_noise), 1, 1, 1), device=initial_noise.device)
        dt = self.max_t / self.total_steps
        times = base * self.max_t
        if cold:
            x = initial_noise
        else:
            noise, signal = self.rates(times - (initial - 1) * dt)
            x = signal * previous + noise * initial_noise
        for index in range(initial, self.total_steps):
            t = times - index * dt
            noise, signal = self.rates(t)
            nn_, ns = self.rates(t - dt)
            x, prediction = self.step_fn(x, measurement, mask, noise, signal, nn_, ns)
        selected, entropy = self.select(prediction)
        observation = measurement[0, -1]
        projected = torch.where(observation != 0, observation, prediction[0, -1])
        return prediction, projected, selected, entropy


def nchw(array, device="cpu"):
    x = torch.from_numpy(np.asarray(array).copy()).to(device=device, dtype=torch.float32)
    return x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)


class CapturedFrame:
    """Warm full frame CUDA Graph; copies inputs, replays, returns owned output tensors."""

    def __init__(self, model, args, steps):
        if args[0].device.type != "cuda":
            raise RuntimeError("CUDA Graph requires a CUDA device")
        self.inputs = tuple(x.clone() for x in args)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.frame(*self.inputs, steps=steps)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = model.frame(*self.inputs, steps=steps)

    def __call__(self, *args):
        for target, source in zip(self.inputs, args):
            target.copy_(source)
        self.graph.replay()
        # No output may alias graph storage after the next call.
        return tuple(x.clone() for x in self.outputs)
