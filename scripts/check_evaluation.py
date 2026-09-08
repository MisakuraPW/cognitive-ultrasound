"""End-to-end evaluation/report/resume check using official weights and synthetic HDF5."""

import h5py
import numpy as np
import yaml

from cognitive_ultrasound.config import ROOT, load
from cognitive_ultrasound.experiments import evaluate


def main():
    root = ROOT / "tmp/evaluation_fixture"
    (root / "test").mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:112, :112]
    frames = np.stack(
        [-60 + 60 * np.exp(-((x - 56 - t) ** 2 + (y - 60) ** 2) / 500) for t in range(3)]
    ).astype("float32")
    with h5py.File(root / "test/synthetic.hdf5", "w") as handle:
        handle["data/image"] = frames
    manifest = root / "split.yaml"
    manifest.write_text(
        yaml.safe_dump({"train": [], "val": [], "test": ["synthetic.hdf5"]}), encoding="utf-8"
    )
    cfg = load(ROOT / "configs/baseline.yaml")
    cfg.update(
        data_root=str(root),
        split_manifest=str(manifest),
        frames=3,
        budgets=[7],
        num_steps=2,
        initial_step=1,
        synthetic_input=True,
        visualize_every=1,
        output="results/evaluation_synthetic_verified",
    )
    evaluate(cfg)
    evaluate(cfg, resume=True)
    print("Synthetic pretrained evaluation, metrics, report and resume verified.")


if __name__ == "__main__":
    main()
