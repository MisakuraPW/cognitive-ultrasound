"""Exercise preparation wrappers and all BF qualification paths on synthetic CPU data."""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import h5py
import numpy as np
import pytest
import yaml

tf = pytest.importorskip("tensorflow")
if tf.keras.backend.backend() != "tensorflow":
    pytest.skip("Fresh TensorFlow process required", allow_module_level=True)

from cognitive_ultrasound.config import ROOT, load  # noqa: E402
from cognitive_ultrasound.preparation import belief  # noqa: E402
from cognitive_ultrasound.preparation.common import read_json  # noqa: E402


def test_preparation_three_stages_and_qualification(tmp_path, monkeypatch):
    cfg = load(ROOT / "configs/preparation.yaml")
    splits = {key: [key + ".hdf5"] for key in ("train", "val", "test")}
    for group in ("train", "val"):
        (tmp_path / "data" / group).mkdir(parents=True)
        with h5py.File(tmp_path / "data" / group / (group + ".hdf5"), "w") as file:
            file["data/image"] = (
                np.random.default_rng(1).uniform(-60, 0, (3, 112, 112)).astype("float32")
            )
    split = tmp_path / "splits.yaml"
    split.write_text(yaml.safe_dump(splits))
    cfg.update(data_root=str(tmp_path / "data"), split_manifest=str(split))
    cfg["frames"]["development"] = 3
    cfg["bf"].update(steps={"codec": 1, "prior": 1, "filter": 1}, chunk_steps=1)
    manifest = {"cohorts": {"train": ["train.hdf5"], "development": ["val.hdf5"]}}
    original = belief.configuration

    def small(*args):
        result = original(*args)
        result.update(width=4, latent_channels=2)
        return result

    monkeypatch.setattr(belief, "configuration", small)
    for stage in ("codec", "prior", "filter"):
        task = dict(stage=stage, cpu=True)
        output = tmp_path / "jobs" / ("bf_" + stage)
        belief.train_stage(task, cfg, manifest, output, tmp_path)
        assert read_json(output / "result.json")["status"] == "completed"
        qualify = tmp_path / "jobs" / ("bf_" + stage + "_qualify")
        belief.qualify(task, cfg, manifest, qualify, tmp_path)
        result = read_json(qualify / "result.json")
        assert result["status"] == "completed"
        assert np.isfinite(list(result["patient_mean_unobserved_mae"].values())).all()
        # Completed qualification resumes without rereading images.
        monkeypatch.setattr(
            belief, "read_frames", lambda *a: pytest.fail("Repeated a completed case")
        )
        belief.qualify(task, cfg, manifest, qualify, tmp_path)
        from cognitive_ultrasound.preparation.common import read_frames

        monkeypatch.setattr(belief, "read_frames", read_frames)
