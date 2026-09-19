"""CPU tests for data isolation, qualification gates and bounded process supervision."""

import copy
import json
import subprocess
import sys
import tarfile

import h5py
import numpy as np
import pytest

from cognitive_ultrasound.config import ROOT, load
from cognitive_ultrasound.preparation import common
from cognitive_ultrasound.preparation.analysis import pairs, risk_probe, select_variant
from cognitive_ultrasound.preparation.sampling import guidance_indices
from cognitive_ultrasound.preparation.suite import Coordinator, bundle_results, run_lock


@pytest.fixture
def cfg():
    return copy.deepcopy(load(ROOT / "configs/preparation.yaml"))


def test_manifest_patient_isolation_and_data_mutation(tmp_path, monkeypatch, cfg):
    splits = {
        "train": [f"t{i}.hdf5" for i in range(5)],
        "val": [f"v{i}.hdf5" for i in range(6)],
        "test": ["holdout.hdf5"],
    }
    monkeypatch.setattr(common, "read_splits", lambda _: splits)
    split = tmp_path / "split.yaml"
    split.write_text("fixed split")
    cfg.update(
        data_root=str(tmp_path),
        split_manifest=str(split),
        cohorts={"train": 2, "debug": 2, "development": 2, "confirmation": 2},
        frames={k: 4 for k in ("train", "debug", "development", "confirmation")},
    )
    for group in ("train", "val"):
        (tmp_path / group).mkdir()
        for name in splits[group]:
            with h5py.File(tmp_path / group / name, "w") as file:
                file.create_dataset("data/image", data=np.full((5, 112, 112), -30, np.float32))
    first = common.make_manifest(cfg)
    assert first == common.make_manifest(cfg)
    groups = list(first["cohorts"].values())
    assert sum(map(len, groups)) == len(set(sum(groups, [])))
    assert all(not s.startswith("test/") for s in first["files"])
    common.verify_manifest(cfg, first)
    target = tmp_path / next(iter(first["files"]))
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Data changed"):
        common.verify_manifest(cfg, first)


def test_future_pairs_do_not_use_current_truth():
    rows = []
    for i in range(3):
        rows.append(
            dict(
                case="a",
                seed=42,
                frame=i,
                unobserved_mae=float(i),
                **{f: 10 + i for f in common.FEATURES},
            )
        )
    p = pairs(rows)
    assert p[0]["x"] == [10] * 6 and p[0]["y"] == 1
    rows[0]["unobserved_mae"] = 1000
    assert pairs(rows)[0] == p[0]
    assert not pairs([rows[0], rows[2]])


def write_rows(root, job, case, offset=0, scale=1):
    directory = root / "jobs" / job
    for i in range(8):
        row = dict(
            case=case,
            seed=42,
            frame=i,
            unobserved_mae=0.1 + i * 0.01,
            **{f: offset + i * 0.02 for f in common.FEATURES},
        )
        row.update(
            psnr=30, ssim=0.9, timing_valid=i > 1, algorithm_s=4 / scale, adapter_wall_s=5 / scale
        )
        common.atomic_npz(
            directory / "42" / case / f"frame_{i:04d}.npz", row=np.array(json.dumps(row))
        )
    common.atomic_json(directory / "result.json", dict(status="completed"))


def test_acceleration_needs_completed_reference_and_fixed_gate(tmp_path, cfg):
    cfg["candidates"] = ["official25"]
    write_rows(tmp_path, "debug_reference", "a")
    write_rows(tmp_path, "debug_official25", "a", scale=3)
    common.atomic_json(
        tmp_path / "jobs/fixed_official25/result.json", dict(status="completed", passed=False)
    )
    assert select_variant(tmp_path, cfg)["variant"] == "reference"
    common.atomic_json(
        tmp_path / "jobs/fixed_official25/result.json", dict(status="completed", passed=True)
    )
    assert select_variant(tmp_path, cfg)["variant"] == "official25"
    (tmp_path / "jobs/debug_reference/result.json").unlink()
    assert select_variant(tmp_path, cfg)["variant"] == "reference"


def test_risk_fit_does_not_leak_confirmation_labels(tmp_path, cfg):
    cfg["risk"].update(minimum_cases=1, minimum_rows=3)
    for group, name in [
        ("train", "train_case"),
        ("development", "dev_case"),
        ("confirmation", "confirm_case"),
    ]:
        write_rows(tmp_path, "risk_" + group, name)
    common.atomic_json(tmp_path / "selection.json", dict(variant="reference"))
    output = tmp_path / "jobs/risk_probe"
    risk_probe(tmp_path, cfg, output)
    before = common.read_json(output / "model.json")
    for file in common.frame_files(tmp_path / "jobs/risk_confirmation"):
        with np.load(file) as archive:
            row = json.loads(str(archive["row"]))
        row["unobserved_mae"] = 1000
        common.atomic_npz(file, row=np.array(json.dumps(row)))
    risk_probe(tmp_path, cfg, output)
    assert common.read_json(output / "model.json") == before
    assert not common.read_json(output / "result.json")["eligible"]


def test_scheduler_timeout_resume_budget_and_dependency(tmp_path, cfg, monkeypatch):
    cfg["job_hours"] = 0.0001
    coordinator = Coordinator(cfg, tmp_path)
    original = subprocess.Popen

    def fake(command, **kwargs):
        return original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake)
    coordinator.job("slow", "trajectory", "A")
    assert coordinator.state["jobs"]["slow"]["status"] == "timed_out"
    spent = coordinator.elapsed()
    coordinator.job("slow", "trajectory", "A")
    assert coordinator.state["jobs"]["slow"]["status"] == "capped"
    assert coordinator.elapsed() == spent
    coordinator.job("dependent", "trajectory", "B", dependencies=["slow"])
    assert coordinator.state["jobs"]["dependent"]["status"] == "blocked"


def test_scheduler_completed_does_not_repeat_and_archive(tmp_path, cfg, monkeypatch):
    coordinator = Coordinator(cfg, tmp_path)
    original = subprocess.Popen

    def fake(command, **kwargs):
        code = (
            "from pathlib import Path; p=Path("
            + repr(str(tmp_path / "jobs/ok/result.json"))
            + '); p.write_text(\'{"status":"completed"}\')'
        )
        return original([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake)
    coordinator.job("ok", "trajectory", "A")
    assert coordinator.state["jobs"]["ok"]["status"] == "completed"
    spent = coordinator.elapsed()
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected repeated work")
    )
    coordinator.job("ok", "trajectory", "A")
    assert coordinator.elapsed() == spent
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path):
                pass
    archive = bundle_results(tmp_path)
    with tarfile.open(archive) as tar:
        assert any(item.name.endswith("jobs/ok/result.json") for item in tar)
    assert archive.with_suffix(".gz.sha256").exists()


def test_guidance_schedule_has_exact_budget():
    assert len(set(guidance_indices(50, 10))) == 10
    assert list(guidance_indices(50, 1)) == [49]
    assert guidance_indices(50, 5)[-1] == 49
    with pytest.raises(ValueError):
        guidance_indices(10, 11)


def test_missing_optional_interpreter_does_not_abort_pipeline(tmp_path, cfg, monkeypatch):
    coordinator = Coordinator(cfg, tmp_path)

    def missing(*a, **k):
        raise FileNotFoundError("missing independent Python")

    monkeypatch.setattr(subprocess, "Popen", missing)
    coordinator.job("optional", "tbig", "C")
    assert coordinator.state["jobs"]["optional"]["status"] == "failed"
    coordinator.job("following", "trajectory", "B", condition="Expected conditional skip")
    assert coordinator.state["jobs"]["following"]["status"] == "blocked"


def test_trajectory_resume_keeps_finished_frames_and_restores_state(tmp_path, monkeypatch, cfg):
    from cognitive_ultrasound.preparation import casl

    visited = []

    class FakeAdapter:
        fail = True

        def __init__(self, *a):
            self.counter = 0

        def reset(self, *a):
            self.counter = 0

        def arrays(self):
            return {"resume_counter": np.array(self.counter)}

        def restore(self, archive):
            self.counter = int(archive["resume_counter"])

        def step(self, target):
            index = int(target[0, 0, 0])
            if index == 2 and self.fail:
                FakeAdapter.fail = False
                raise RuntimeError("Simulated interrupted frame")
            assert self.counter == index
            self.counter += 1
            visited.append(index)
            return {"psnr": 20.0, "algorithm_s": 1.0}, {"target": target}

    monkeypatch.setattr(casl, "Adapter", FakeAdapter)
    monkeypatch.setattr(
        casl, "read_frames", lambda cfg, split, name, count, start: np.full((1, 112, 112, 1), start)
    )
    cfg["frames"]["debug"] = 4
    task = dict(id="resume", cohort="debug", variant="reference")
    manifest = {"cohorts": {"debug": ["a.hdf5"]}}
    with pytest.raises(RuntimeError, match="Simulated"):
        casl.run_trajectory(task, cfg, manifest, tmp_path)
    first = (tmp_path / "42/a/frame_0000.npz").read_bytes()
    casl.run_trajectory(task, cfg, manifest, tmp_path)
    assert visited == [0, 1, 2, 3]
    assert (tmp_path / "42/a/frame_0000.npz").read_bytes() == first
    assert common.read_json(tmp_path / "result.json")["status"] == "completed"


def test_offline_report_produces_inspectable_figures_and_csv(tmp_path):
    from cognitive_ultrasound.preparation.analysis import report

    write_rows(tmp_path, "debug_reference", "synthetic")
    y, x = np.mgrid[:112, :112]
    target = (np.sin(x / 20) * np.cos(y / 20)).astype("float32")[..., None]
    for file in common.frame_files(tmp_path / "jobs/debug_reference"):
        with np.load(file) as archive:
            row = archive["row"].copy()
        mask = np.zeros_like(target)
        mask[:, ::8, :] = 1
        common.atomic_npz(
            file,
            row=row,
            target=target,
            prediction=target * 0.8,
            mask=mask,
            uncertainty=np.abs(target[..., 0]),
        )
    common.atomic_json(
        tmp_path / "status.json",
        {"jobs": {"debug_reference": {"status": "completed", "elapsed_s": 1}}},
    )
    report(tmp_path)
    assert (tmp_path / "REPORT.md").is_file()
    assert (tmp_path / "frames_summary.csv").is_file()
    from PIL import Image

    image = Image.open(tmp_path / "jobs/debug_reference/examples.png")
    assert image.width > 1000 and image.height > 500
