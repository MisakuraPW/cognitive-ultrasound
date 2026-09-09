import json
import runpy

import h5py
import numpy as np
import pytest
import yaml

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.conversion import run, verify_h5


def test_invalid_hdf5_frames_and_pixels(tmp_path):
    file = tmp_path / "sample.hdf5"
    with h5py.File(file, "w") as handle:
        handle["data/image_sc"] = np.full((2, 112, 112), -30.0)
        handle["data/image"] = np.full((2, 112, 112), -30.0)
    assert verify_h5(file, True, 2) == 2
    with pytest.raises(ValueError, match="Frame count"):
        verify_h5(file, True, 3)
    with h5py.File(file, "a") as handle:
        handle["data/image"][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid pixels"):
        verify_h5(file, True)


def test_resume_refuses_changed_identity_before_touching_files(tmp_path):
    raw = tmp_path / "raw"
    (raw / "Videos").mkdir(parents=True)
    (raw / "Videos/a.avi").write_bytes(b"placeholder")
    manifest = tmp_path / "split.yaml"
    manifest.write_text(yaml.safe_dump({"train": ["a.hdf5"], "val": [], "test": []}))
    output = tmp_path / "output"
    output.mkdir()
    metadata = output / "conversion_manifest.json"
    metadata.write_text(json.dumps({"identity": {"old": "different source"}}))
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged")
    with pytest.raises(ValueError, match="changed"):
        run(raw, output, manifest, workers=2, resume=True)
    assert sentinel.read_text() == "unchanged"


def test_stop_only_accepts_matching_detached_pipeline():
    scope = runpy.run_path(str(ROOT / "scripts/stop_autodl_pipeline.py"))
    verify = scope["verify_target"]
    command = ["python", str(ROOT / "scripts/autodl_overnight.py"), "--with-training"]
    verify(100, command, ROOT, 100)
    for pid, cmd, cwd, pgid in (
        (1, command, ROOT, 1),
        (100, ["python", "unrelated.py"], ROOT, 100),
        (100, command, ROOT.parent, 100),
        (100, command, ROOT, 50),
    ):
        with pytest.raises(RuntimeError, match="refusing to stop"):
            verify(pid, cmd, cwd, pgid)


def test_local_preflight_memory_and_input_guard(tmp_path, monkeypatch):
    scope = runpy.run_path(str(ROOT / "scripts/local_conversion.py"))
    preflight = scope["preflight"]
    namespace = preflight.__globals__
    raw = tmp_path / "raw"
    (raw / "Videos").mkdir(parents=True)
    (raw / "Videos/a.avi").write_bytes(b"video placeholder")
    monkeypatch.setitem(
        namespace, "read_splits", lambda file: {"train": ["a.hdf5"], "val": [], "test": []}
    )
    monkeypatch.setitem(namespace, "available_memory_gib", lambda: 3.5)
    monkeypatch.setattr(
        namespace["shutil"], "disk_usage", lambda path: type("Disk", (), {"free": 190 * 2**30})()
    )
    preflight(raw, tmp_path / "out", 2)
    with pytest.raises(RuntimeError, match="available RAM"):
        preflight(raw, tmp_path / "out", 4)
    with pytest.raises(ValueError, match="overlaps"):
        preflight(raw, raw / "out", 2)
