"""Benchmark protocol checks, with actual HDF5 and subprocess control workload."""

import copy
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cognitive_ultrasound import speedbench as b


@pytest.fixture
def fixture(tmp_path):
    cfg = yaml.safe_load((b.ROOT / "configs/speedbench.yaml").read_text())
    cfg.update(cases=1, frames=5, methods=["official", "interpolation"], timeout_seconds=30)
    cfg["pythons"] = dict(jax=sys.executable, torch=sys.executable)
    cfg["data_root"] = str(tmp_path / "data")
    cfg["functional_test_only"] = True
    cfg["checkpoint"] = str(tmp_path / "missing_official_assets")
    folder = Path(cfg["data_root"]) / "val"
    folder.mkdir(parents=True)
    with h5py.File(folder / "one.hdf5", "w") as h:
        h.create_dataset(
            "data/image",
            data=np.random.default_rng(9).uniform(-60, 0, (5, 112, 112)).astype("float32"),
        )
    cfg["split_manifest"] = str(tmp_path / "split.yaml")
    Path(cfg["split_manifest"]).write_text(
        yaml.safe_dump(dict(train=["train.hdf5"], val=["one.hdf5"], test=["test.hdf5"]))
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "output"
    root.mkdir()
    b.atomic(root / "config.json", cfg)
    b.atomic(root / "manifest.json", dict(cases=["one.hdf5"]))
    return cfg, root, path


def test_new_coarse_schedule_preserves_warm_interval():
    for n in [4, 8, 16]:
        c = b.CoarseProfile("coarse", steps=n, dps=n)
        assert c.variant() == dict(
            initial_step=450, precision="float32", inference_steps=n, guidance_steps=n
        )
        assert c.category == "B"
    assert b.Profile("native8", steps=8, dps=8).variant()["initial_step"] == 492
    assert b.catalog()["fast_belief"]["category"] == "C"
    assert b.catalog()["senior_original"]["kind"] == "unavailable"


def test_float_metrics_reject_bad_masks_and_nonfinite():
    x = np.zeros((112, 112, 1), np.float32)
    m = x.copy()
    m[:, ::8] = 1
    y = x + 0.0001
    score = b.measurements(x, y, m)
    assert score["mse"] > 0  # no uint8 rounding away a small error
    with pytest.raises(FloatingPointError):
        b.measurements(x, y * np.nan, m)
    m[1, 0] = 0.5
    with pytest.raises(ValueError):
        b.measurements(x, y, m)
    assert b.statistics([1, 2, 3])["mean_s"] == 2
    with pytest.raises(ValueError):
        b.statistics([1, np.nan])


def test_real_hdf5_control_case_resume_and_checksum(fixture):
    cfg, root, _ = fixture
    b.worker(cfg, "interpolation", root)
    markers = sorted(root.glob("interpolation/*/*/complete.json"))
    assert len(markers) == 2
    for p in markers:
        r = b.read(p)
        assert r["frames"] == 5 and r["observed_max_error"] == 0
        assert r["micro"] is None
    stamps = [p.stat().st_mtime_ns for p in markers]
    b.worker(cfg, "interpolation", root)
    assert stamps == [p.stat().st_mtime_ns for p in markers]
    (markers[0].parent / "frames.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact"):
        b.worker(cfg, "interpolation", root)


def test_micro_replay_does_not_advance_sequence(fixture, monkeypatch):
    cfg, root, _ = fixture

    class Engine:
        def __init__(self):
            self.adapter = self

        def reset(self, seed):
            self.index = 0

        def snapshot(self):
            return dict(counter=np.array(self.index))

        def capture(self):
            return self.index

        def reinstate(self, state):
            self.index = state

        def force_lines(self, lines):
            self.lines = lines

        def peak(self):
            return 0

        def step(self, target, noise):
            self.index += 1
            mask = np.zeros_like(target)
            mask[:, getattr(self, "lines", np.linspace(0, 111, 14).round().astype(int))] = 1
            return {}, dict(
                prediction=np.where(mask, target, self.index / 100).astype("float32"),
                mask=mask,
                next_action=mask[0, :, 0],
            )

    monkeypatch.setattr(b, "build", lambda *args: Engine())
    b.worker(cfg, "official", root)
    for p in root.glob("official/*/*/complete.json"):
        assert b.read(p)["micro"]["mean_s"] > 0
        with np.load(p.parent / "reference.npz") as z:
            values = [z["prediction"][i][~z["mask"][i].astype(bool)][0] for i in range(5)]
        assert np.allclose(values, np.arange(1, 6) / 100)


def test_config_no_test_and_invalid_method(fixture):
    cfg, _, path = fixture
    assert b.config(path)["modes"] == ["fixed", "closed_loop"]
    cfg["methods"].append("made_up_solver")
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        b.config(path)


def test_real_subprocess_failure_not_success_and_completed_reuse(fixture, monkeypatch):
    cfg, root, path = fixture
    # Fresh coordinator root; existing helper fixture remains untouched.
    output = root.parent / "coordinator"
    cfg["methods"] = ["official", "interpolation"]
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(b, "hardware", lambda: {"cpu": "functional"})
    b.run(path, output)
    state = b.read(output / "status.json")
    assert state["methods"]["official"]["status"] != "completed"  # no GPU/official assets
    assert state["methods"]["interpolation"]["status"] == "completed"
    assert state["status"] == "finished_with_gaps"
    summary = b.read(output / "summary.json")
    assert len(summary["rows"]) == 2 and all(r["complete"] for r in summary["rows"])
    marks = list(output.glob("interpolation/*/*/complete.json"))
    stamps = [p.stat().st_mtime_ns for p in marks]
    b.run(path, output, resume=True)
    assert stamps == [p.stat().st_mtime_ns for p in marks]
    changed = copy.deepcopy(cfg)
    changed["frames"] = 6
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="changed"):
        b.run(path, output)
