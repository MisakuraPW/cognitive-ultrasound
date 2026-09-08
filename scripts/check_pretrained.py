"""Real pretrained CPU compatibility check on synthetic frames, using reduced diffusion steps.

This tests integration and output parity; it is NOT an EchoNet/paper-quality reproduction.
"""

import argparse

import numpy as np

from cognitive_ultrasound.config import ROOT, load
from cognitive_ultrasound.models.diffusion import CASLLoop
from cognitive_ultrasound.provenance import write_json
from cognitive_ultrasound.visualization import save_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/pretrained_synthetic_check")
    args = p.parse_args()
    cfg = load(ROOT / "configs/baseline.yaml")
    cfg.update(num_steps=2, initial_step=1, metrics=["psnr", "ssim"])
    y, x = np.mgrid[:112, :112]
    target = (2 * np.exp(-((x - 56) ** 2 + (y - 60) ** 2) / 500) - 1).astype("float32")[..., None]
    results = []
    for method in ("random", "uniform", "casl"):
        loop = CASLLoop(cfg, method, 7)
        for i in range(2):
            state, timing = loop.step(target)
            assert state["posterior_samples"].shape == (2, 112, 112, 3)
            folder = ROOT / args.output / method / f"frame_{i:04d}"
            save_frame(folder, state)
            np.savez_compressed(folder / "state.npz", **state)
            results.append(
                {"method": method, "frame": i, "actual_lines": state["actual_lines"], **timing}
            )
    write_json(
        ROOT / args.output / "verification.json",
        {"status": "PRETRAINED_SYNTHETIC_REDUCED_STEPS_ONLY", "config": cfg, "rows": results},
    )
    print("Pretrained synthetic integration passed; this is not paper reproduction.")


if __name__ == "__main__":
    main()
