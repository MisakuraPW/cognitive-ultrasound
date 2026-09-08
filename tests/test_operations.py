import json
import tarfile

import pytest
import yaml

from cognitive_ultrasound.config import ROOT, load, validate
from cognitive_ultrasound.operations import estimate_evaluation, estimate_training, export_results
from cognitive_ultrasound.provenance import sha256, write_json


def test_paper_config_reduces_storage_without_changing_algorithm():
    paper = validate(load(ROOT / "configs/autodl/paper.yaml"))
    baseline = load(ROOT / "configs/autodl/evaluation.yaml")
    for key in baseline.keys() - {
        "output",
        "budgets",
        "segmentation",
        "save_trajectory",
        "visualize_every",
    }:
        assert paper[key] == baseline[key]
    assert paper["budgets"] == [2, 4, 7, 14, 28]
    assert paper["segmentation"] and not paper["save_trajectory"]


def test_export_excludes_bulk_files_and_keeps_metrics(tmp_path):
    source = tmp_path / "outputs"
    for name in (
        "run/frames.csv",
        "run/manifest.json",
        "run/trajectory/state.npz",
        "train/hub/model.weights.h5",
        "train/resume/checkpoint",
        "demo/reconstruction.png",
    ):
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test fixture")
    archive = tmp_path / "exports/results.tar.gz"
    result = export_results(source, archive)
    with tarfile.open(archive) as stream:
        assert set(stream.getnames()) == {
            "outputs/run/frames.csv",
            "outputs/run/manifest.json",
            "outputs/demo/reconstruction.png",
        }
    assert result["sha256"] == sha256(archive)
    assert len(result["excluded"]) == 3
    with pytest.raises(FileExistsError):
        export_results(source, archive)
    full = tmp_path / "full.tar.gz"
    export_results(source, full, True, True)
    with tarfile.open(full) as stream:
        assert len(stream.getnames()) == 6


def test_export_refuses_archive_inside_results(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        export_results(tmp_path, tmp_path / "bad.tar.gz")


def test_evaluation_projection_uses_measured_endpoint_time(tmp_path):
    cfg = load(ROOT / "configs/autodl/paper.yaml")
    cfg.update(methods=["casl"], budgets=[7], frames=100)
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"test": ["a.hdf5", "b.hdf5", "c.hdf5"]}), encoding="utf-8")
    cfg["split_manifest"] = str(split)
    target = tmp_path / "target.yaml"
    target.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pilot = tmp_path / "pilot"
    write_json(pilot / "manifest.json", {"identity": {"config": cfg, "cases": ["p.hdf5"]}})
    write_json(
        pilot / "casl/lines_007/p/complete.json",
        {"frames": 100, "case_wall_s": 200, "warmup_compile_s": 10},
    )
    result = estimate_evaluation(pilot, target)
    assert result["projected_hours"] == pytest.approx(630 / 3600)
    cfg["segmentation"] = False
    target.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="segmentation"):
        estimate_evaluation(pilot, target)


def test_training_projection_drops_compile_epoch_and_warmup(tmp_path):
    cfg = load(ROOT / "configs/autodl/training.yaml")
    cfg.update(epochs=3, steps_per_epoch=100)
    target = tmp_path / "target.yaml"
    target.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pilot = tmp_path / "pilot"
    write_json(
        pilot / "training_manifest.json", {"identity": {"config": cfg, "synthetic_smoke": False}}
    )
    write_json(pilot / "timing/epoch_0001.json", {"epoch_wall_s": 1000, "batch_wall_s": [50] * 10})
    write_json(
        pilot / "timing/epoch_0002.json", {"epoch_wall_s": 27, "batch_wall_s": [3] * 5 + [2] * 5}
    )
    result = estimate_training(pilot, target)
    assert result["measured_step_s"] == 2
    assert result["projected_hours"] == pytest.approx((300 * 2 + 3 * 2) / 3600)


def test_synthetic_training_is_not_a_real_speed_estimate(tmp_path):
    write_json(
        tmp_path / "training_manifest.json", {"identity": {"config": {}, "synthetic_smoke": True}}
    )
    with pytest.raises(ValueError, match="Synthetic"):
        estimate_training(tmp_path, ROOT / "configs/autodl/training.yaml")


def test_export_inventory_matches_sidecar(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    (source / "summary.csv").write_text("psnr\n23\n", encoding="utf-8")
    archive = tmp_path / "run.tar.gz"
    result = export_results(source, archive)
    assert json.loads(archive.with_suffix(".gz.json").read_text(encoding="utf-8")) == result
