import json

import numpy as np
import pytest
import yaml

from cognitive_ultrasound.cloud import inspect_inputs, preflight, validate_paths, video_id
from cognitive_ultrasound.config import ROOT, load, validate


@pytest.fixture
def inventory(tmp_path):
    raw = tmp_path / "raw"
    (raw / "Videos").mkdir(parents=True)
    # Deliberately different from CASL split: report, never silently substitute.
    (raw / "FileList.csv").write_text(
        "FileName,Split\n001.avi,TEST\n002,TRAIN\n003.avi,VAL\n", encoding="utf-8"
    )
    for name in ("001", "002", "003"):
        (raw / "Videos" / f"{name}.avi").write_bytes(b"inventory fixture, not decodable AVI")
    manifest = tmp_path / "split.yaml"
    manifest.write_text(
        yaml.safe_dump({"train": ["001.hdf5"], "val": ["002.hdf5"], "test": ["003.hdf5"]}),
        encoding="utf-8",
    )
    cache = tmp_path / "legacy"
    (cache / "npy").mkdir(parents=True)
    np.save(cache / "npy/001.npy", np.zeros((2, 112, 112), dtype=np.uint8))
    return {
        "raw_root": str(raw),
        "legacy_cache_roots": [str(cache)],
        "protected_project_roots": [str(tmp_path / "EchoRVM")],
        "protected_output_roots": [str(tmp_path / "outputs")],
        "code_root": str(tmp_path / "casl"),
        "polar_root": str(tmp_path / "polar"),
        "output_root": str(tmp_path / "outputs_casl"),
        "backup_root": str(tmp_path / "backup"),
        "split_manifest": str(manifest),
    }


def test_inventory_reads_without_modifying_shared_inputs(inventory, tmp_path):
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = inspect_inputs(inventory)
    assert result["ready_for_conversion_inventory"]
    assert result["casl_vs_original_split_crosswalk"]["train"] == {"TEST": 1}
    assert result["legacy_caches"][0]["samples"][0]["layout"] == "THW"
    assert result["legacy_caches"][0]["samples"][0]["header_valid"]
    assert not result["tracings_present"]  # not required for model-agreement baseline
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_npy_does_not_replace_missing_avi(inventory):
    from pathlib import Path

    (Path(inventory["raw_root"]) / "Videos/001.avi").unlink()
    result = inspect_inputs(inventory)
    assert not result["ready_for_conversion_inventory"]
    assert result["missing_avi"]["train"] == ["001"]
    assert result["legacy_caches"][0]["npy_files"] == 1


def test_duplicate_csv_and_invalid_split_fail_inventory(inventory):
    from pathlib import Path

    with (Path(inventory["raw_root"]) / "FileList.csv").open("a", encoding="utf-8") as stream:
        stream.write("001,TRAIN\n004,\n")
    result = inspect_inputs(inventory)
    assert not result["ready_for_conversion_inventory"]
    assert result["duplicate_csv_ids"] == ["001"]
    assert any("Invalid original Split" in p for p in result["problems"])


@pytest.mark.parametrize(
    "key", ["raw_root", "legacy_cache_roots", "protected_project_roots", "protected_output_roots"]
)
def test_destinations_and_report_cannot_overlap_protected_paths(inventory, key):
    from pathlib import Path

    value = inventory[key]
    protected = Path(value if isinstance(value, str) else value[0])
    with pytest.raises(ValueError, match="protected source"):
        validate_paths(inventory, protected / "report.json")
    with pytest.raises(ValueError, match="protected source"):
        validate_paths({**inventory, "polar_root": str(protected / "casl")})


def test_preflight_writes_only_report(inventory, tmp_path):
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump(inventory), encoding="utf-8")
    report = tmp_path / "outputs_casl/preflight.json"
    result = preflight(config, report, include_system=False)
    assert json.loads(report.read_text(encoding="utf-8")) == result
    assert result["inputs"]["ready_for_conversion_inventory"]


@pytest.mark.parametrize("name", ["../escape", "a\\b", "C:drive", "", ".."])
def test_invalid_video_identifiers_rejected(name):
    with pytest.raises(ValueError, match="Invalid video ID"):
        video_id(name)


def test_autodl_configs_preserve_baseline():
    cfg = validate(load(ROOT / "configs/autodl/evaluation.yaml"))
    base = load(ROOT / "configs/baseline.yaml")
    for key in base.keys() - {"data_root", "checkpoint", "evaluation_checkpoints", "output"}:
        assert cfg[key] == base[key]
    demo = validate(load(ROOT / "configs/autodl/demo.yaml"))
    assert (demo["num_steps"], demo["initial_step"], demo["temporal_window"]) == (500, 450, 3)
    assert (demo["split"], demo["limit_cases"], demo["frames"]) == ("val", 1, 5)
    train = load(ROOT / "configs/autodl/training.yaml")
    base_train = load(ROOT / "configs/training.yaml")
    for key in base_train.keys() - {"train_folder", "val_folder", "output"}:
        assert train[key] == base_train[key]


def test_inheritance_merges_nested_keys_and_rejects_cycles(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("nested: {a: 1, b: 2}\nvalues: [1, 2]\n", encoding="utf-8")
    child.write_text("extends: base.yaml\nnested: {b: 3}\nvalues: [4]\n", encoding="utf-8")
    assert load(child) == {"nested": {"a": 1, "b": 3}, "values": [4]}
    base.write_text("extends: child.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load(child)
