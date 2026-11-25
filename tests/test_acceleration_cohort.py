from pathlib import Path

import pytest
import yaml

from cognitive_ultrasound.acceleration_cohort import lock_cohort, verify_cases

ROOT = Path(__file__).resolve().parents[1]


def test_core_cohort_has_fixed_counts_and_nested_training_prefix(tmp_path):
    config = ROOT / "configs/acceleration_core.yaml"
    split = ROOT / "configs/splits/split.yaml"
    core = lock_cohort(config, split)
    assert core == lock_cohort(config, split)
    assert {k: len(v) for k, v in core["cases"].items()} == {
        "train": 1024, "development": 128, "test": 128,
    }
    expanded = yaml.safe_load(config.read_text(encoding="utf-8"))
    expanded["train"] = None
    full = tmp_path / "larger.yaml"
    full.write_text(yaml.safe_dump(expanded))
    larger = lock_cohort(full, split)
    assert len(larger["cases"]["train"]) == 6985
    assert larger["cases"]["train"][:1024] == core["cases"]["train"]
    for group in ("development", "test"):
        assert larger["cases"][group] == core["cases"][group]
    wrong = {**core["cases"], "test": core["cases"]["train"][:128]}
    with pytest.raises(ValueError, match="test"):
        verify_cases(wrong, yaml.safe_load(split.read_text(encoding="utf-8")))
