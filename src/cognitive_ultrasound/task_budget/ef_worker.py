"""Native EchoNet R(2+1)D EF model; never imports JAX/TensorFlow."""

import argparse
import csv
import json
import os
import pickle
import sys
import traceback
import urllib.request
from pathlib import Path

import numpy as np

from ..preparation.common import atomic_json, atomic_npz, read_json
from ..provenance import sha256
from .exchange import write_exchange

WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/"
    "r2plus1d_18_32_2_pretrained.pt"
)


def prepare_assets(cfg):
    import cv2

    file = Path(cfg["ef_weights"])
    file.parent.mkdir(parents=True, exist_ok=True)
    receipt = file.with_suffix(".source.json")
    if not file.exists():
        temporary = file.with_suffix(".partial")
        with (
            urllib.request.urlopen(WEIGHTS_URL, timeout=60) as response,
            temporary.open("wb") as out,
        ):
            while data := response.read(2**20):
                out.write(data)
        temporary.replace(file)
        atomic_json(
            receipt,
            dict(
                url=WEIGHTS_URL,
                sha256=sha256(file),
                pretraining_split="EchoNet original TRAIN (official provenance assumption)",
            ),
        )
    elif receipt.exists() and read_json(receipt)["sha256"] != sha256(file):
        raise ValueError("EF checkpoint changed since acquisition")
    elif not receipt.exists():
        raise ValueError("Existing EF weights need a .source.json with sha256 and provenance")
    stats_file = Path(cfg["ef_stats"])
    if stats_file.exists():
        saved = read_json(stats_file)
        if saved.get("ef_sha256") != sha256(cfg["ef_weights"]) or saved.get(
            "file_list_sha256"
        ) != sha256(cfg["file_list"]):
            raise ValueError(
                "Normalization provenance changed; create an explicit new task asset version"
            )
        return saved
    with open(cfg["file_list"], encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["Split"].upper() == "TRAIN"]
    rng = np.random.default_rng(0)
    chosen = rng.choice(
        len(rows), min(cfg["runtime"]["raw_stats_samples"], len(rows)), replace=False
    )
    total, s1, s2, sources = 0, np.zeros(3), np.zeros(3), []
    for i in chosen:
        file = Path(cfg["raw_videos"]) / (Path(rows[int(i)]["FileName"]).stem + ".avi")
        cap = cv2.VideoCapture(str(file))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or length < 1:
            raise ValueError(f"Missing raw normalization input: {file}")
        # EchoNet get_mean_and_std operates on sampled training clips (default16,period2).
        start = int(rng.integers(max(1, length - 30)))
        for index in np.minimum(start + np.arange(16) * 2, length - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != (112, 112):
                raise ValueError(f"Raw frame unreadable or wrong size: {file}:{index}")
            pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
            total += len(pixels)
            s1 += pixels.sum(0)
            s2 += (pixels * pixels).sum(0)
        cap.release()
        sources.append(dict(file=str(file), start=start, sha256=sha256(file)))
    mean = s1 / total
    std = np.sqrt(s2 / total - mean**2)
    value = dict(
        mean=mean.tolist(),
        std=std.tolist(),
        units="uint8_0_255_RGB",
        source="deterministic original-TRAIN sampled clips; not claimed exact author draw",
        task_frames=32,
        task_period=2,
        sources=sources,
        file_list_sha256=sha256(cfg["file_list"]),
        ef_sha256=sha256(cfg["ef_weights"]),
    )
    atomic_json(stats_file, value)
    return value


class EFModel:
    def __init__(self, cfg, coordinates, model=None):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from torchvision.models.video import r2plus1d_18

        self.torch = torch
        self.device = cfg["task_device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EF CUDA unavailable; no silent CPU fallback")
        torch.set_num_threads(cfg["runtime"]["threads"])
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        if model is None:
            model = r2plus1d_18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
            try:
                payload = torch.load(cfg["ef_weights"], map_location="cpu", weights_only=True)
            except pickle.UnpicklingError:
                # Old official checkpoints may contain NumPy metric scalars.
                # Explicit numeric-only allowlist; never fall back to arbitrary pickle loading.
                with torch.serialization.safe_globals(
                    [
                        np.core.multiarray.scalar,
                        (np.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                        np.dtype,
                        type(np.dtype("float64")),
                        type(np.dtype("float32")),
                        type(np.dtype("int64")),
                        type(np.dtype("int32")),
                    ]
                ):
                    payload = torch.load(cfg["ef_weights"], map_location="cpu", weights_only=True)
            if payload.get("frames", 32) != 32 or payload.get("period", 2) != 2:
                raise ValueError("EF checkpoint temporal metadata incompatible")
            state = payload.get("state_dict", payload)
            model.load_state_dict(
                {k.removeprefix("module."): v for k, v in state.items()}, strict=True
            )
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        execution = cfg["runtime"].get("ef_execution", "eager")
        if execution == "compile":
            import torch._inductor.config as compiler_config

            compiler_config.compile_threads = 1
            self.model = torch.compile(self.model, fullgraph=True, dynamic=False)
        elif execution != "eager":
            raise ValueError("Unknown EF execution mode: " + execution)
        stats = read_json(cfg["ef_stats"])
        if stats["units"] != "uint8_0_255_RGB":
            raise ValueError("Unknown normalization units")
        mean, std = np.array(stats["mean"]), np.array(stats["std"])
        if (
            mean.shape != (3,)
            or std.shape != (3,)
            or not np.isfinite([mean, std]).all()
            or (std <= 0).any()
        ):
            raise ValueError("Invalid fixed task normalization")
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32)[
            None, :, None, None, None
        ]
        self.std = torch.tensor(std, device=self.device, dtype=torch.float32)[
            None, :, None, None, None
        ]
        # Official Zea coordinates are [rho/theta, x, z]; swap output axes to[z,x].
        rho, theta = coordinates[:, :, :]
        self.valid = torch.tensor(
            np.isfinite(rho)
            & np.isfinite(theta)
            & (rho >= 0)
            & (rho <= 111)
            & (theta >= 0)
            & (theta <= 111),
            device=self.device,
        ).T
        # Gather-based bilinear interpolation supports deterministic CUDA backward;
        # grid_sample CUDA backward may be rejected under deterministic algorithms.
        r = np.clip(np.nan_to_num(rho, nan=0), 0, 111).T
        a = np.clip(np.nan_to_num(theta, nan=0), 0, 111).T
        r0, a0 = np.floor(r).astype(int), np.floor(a).astype(int)
        r1, a1 = np.minimum(r0 + 1, 111), np.minimum(a0 + 1, 111)
        wr, wa = r - r0, a - a0
        self.neighbors = [
            (
                torch.tensor((rr * 112 + aa).ravel(), device=self.device),
                torch.tensor(w.ravel(), device=self.device, dtype=torch.float32),
            )
            for rr, aa, w in (
                (r0, a0, (1 - wr) * (1 - wa)),
                (r0, a1, (1 - wr) * wa),
                (r1, a0, wr * (1 - wa)),
                (r1, a1, wr * wa),
            )
        ]
        self.cartesian_shape = r.shape

    def preprocess(self, clips, domain):
        torch = self.torch
        batch, frames = clips.shape[:2]
        if domain == "polar":
            source = clips.reshape(batch * frames, 112 * 112)
            x = sum(
                torch.index_select(source, 1, index) * weight for index, weight in self.neighbors
            )
            x = x.reshape(batch * frames, 1, *self.cartesian_shape)
            x = torch.where(self.valid[None, None], x, -1)
            x = x[:, :, :, 23:-24]  # same CASL EchoNet Cartesian crop
            if x.shape[-2:] != (112, 112):
                raise ValueError(f"Unexpected scan conversion/crop geometry {x.shape}")
            x = (x.clamp(-1, 1) + 1) * 127.5
            x = (
                x.reshape(batch, frames, 1, 112, 112)
                .permute(0, 2, 1, 3, 4)
                .expand(-1, 3, -1, -1, -1)
            )
        elif domain == "cartesian_rgb":
            if clips.shape[2:] != (112, 112, 3):
                raise ValueError("Expected Cartesian RGB uint8-scale input")
            x = clips.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError("Unknown EF domain")
        return (x - self.mean) / self.std

    def evaluate(self, array, gradient=False, domain="polar"):
        torch = self.torch
        clips = torch.tensor(array, dtype=torch.float32, device=self.device, requires_grad=gradient)
        with torch.set_grad_enabled(gradient):
            pred = self.model(self.preprocess(clips, domain)).flatten()
            result = dict(predictions=pred.detach().cpu().numpy())
            if gradient:
                result["gradients"] = torch.autograd.grad(pred.sum(), clips)[0].cpu().numpy()
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError("Nonfinite frozen EF computation")
        return result


def serve(cfg, coordinates):
    evaluator = EFModel(cfg, coordinates)
    import torch

    print(
        json.dumps(
            dict(
                ready=True,
                torch=torch.__version__,
                device=cfg["task_device"],
                weights=sha256(cfg["ef_weights"]),
                stats=sha256(cfg["ef_stats"]),
                frozen=True,
                precision="float32",
                scan_conversion="bilinear",
            )
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["op"] == "close":
                break
            with np.load(request["input"], allow_pickle=False) as data:
                result = evaluator.evaluate(data["clips"], request["gradient"], request["domain"])
            write_exchange(request["output"], result, request.get("ipc_mode", "compressed"))
            print('{"done":true}', flush=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(error=f"{type(error).__name__}: {error}")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "serve"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--coordinates")
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.action == "prepare":
        prepare_assets(cfg)
    else:
        with np.load(args.coordinates, allow_pickle=False) as data:
            serve(cfg, data["coordinates"])


if __name__ == "__main__":
    main()
