import json

import h5py
import numpy as np
import pytest
import yaml

from cognitive_ultrasound.config import ROOT, load, validate
from cognitive_ultrasound.data import audit_dataset, inspect_file, read_splits
from cognitive_ultrasound.evaluation.metrics import dice, psnr, uint8_image
from cognitive_ultrasound.evaluation.segmentation import segmentation_failure
from cognitive_ultrasound.experiments import case_seed, synthetic_smoke


def test_pinned_patient_split():
    splits = read_splits(ROOT / "configs/splits/split.yaml")
    assert {k: len(v) for k, v in splits.items()} == {"train": 6985, "val": 500, "test": 500}


def test_reject_patient_leakage(tmp_path):
    p = tmp_path / "split.yaml"
    p.write_text(yaml.safe_dump({"train": ["a.hdf5"], "val": ["a.hdf5"], "test": []}))
    with pytest.raises(ValueError, match="leakage"):
        read_splits(p)


def test_hdf5_range_and_shape(tmp_path):
    p = tmp_path / "a.hdf5"
    with h5py.File(p, "w") as f:
        f["data/image"] = np.full((3, 112, 112), -30, dtype="float32")
    assert inspect_file(p)["frames"] == 3
    with h5py.File(p, "r+") as f:
        f["data/image"][0, 0, 0] = 255
    with pytest.raises(ValueError, match="range"):
        inspect_file(p)


def test_missing_data_not_completed(tmp_path):
    report = tmp_path / "stats.md"
    with pytest.raises(ValueError, match="incomplete"):
        audit_dataset(tmp_path / "absent", ROOT / "configs/splits/split.yaml", report)
    assert not json.loads(report.with_suffix(".json").read_text())["complete"]


def test_psnr_and_dice():
    x = np.zeros((112, 112), dtype=np.uint8)
    assert psnr(x, x) == float("inf")
    assert psnr(x, np.full_like(x, 255)) == pytest.approx(0)
    assert dice(x, x) == 1
    assert dice(x, np.ones_like(x)) == 0
    assert np.array_equal(uint8_image([-1, 0, 1]), [0, 127, 255])


def test_segmentation_failure_consecutive():
    broken = np.zeros((112, 112), dtype=bool)
    broken[20:25, 20:25] = True
    broken[40:45, 40:45] = True
    good = np.ones_like(broken)
    assert segmentation_failure([broken] * 5)
    assert not segmentation_failure([broken] * 4 + [good] + [broken] * 4)


@pytest.mark.parametrize(
    "change",
    [
        {"particles": 1},
        {"budgets": [113]},
        {"budgets": []},
        {"frames": 0},
        {"temporal_window": 2},
        {"split": "train"},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        validate(load(ROOT / "configs/baseline.yaml") | change)


def test_smoke_is_explicitly_synthetic(tmp_path):
    result = synthetic_smoke(tmp_path / "demo")
    assert result["casl_executed"] is False
    assert (tmp_path / "demo/frame_0000/selected_lines.png").is_file()
    assert case_seed(42, "a") == case_seed(42, "a")
    assert case_seed(42, "a") != case_seed(42, "b")
