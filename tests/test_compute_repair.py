"""Repair boundaries and inheritance must conserve evidence, not bypass validation."""

from pathlib import Path

import pytest

from cognitive_ultrasound.compute_lab.protocol import profiles
from cognitive_ultrasound.compute_lab.scheduling import cost_decision, failure_kind, fastest_torch
from cognitive_ultrasound.compute_lab.suite import BoundaryPause, Suite
from cognitive_ultrasound.preparation.common import atomic_json, read_json


def record(name, seconds, **kwargs):
    p = profiles()[name]
    return dict(
        status="completed",
        profile=p.record(),
        cohort="development",
        closed_loop_s=seconds,
        micro=[dict(closed_loop_s=seconds)],
        **kwargs,
    )


def test_cost_screen_and_no_eager_fallback():
    official = record("official", 1)
    assert not cost_decision(record("torch_eager", 12), official)["proceed"]
    assert fastest_torch([official, record("torch_eager", 0.1, internal_correctness=True)]) is None
    assert fastest_torch([official, record("torch_compile", 2, internal_correctness=True)]) is None
    assert fastest_torch([official, record("torch_graph", 0.1, internal_correctness=False)]) is None
    assert (
        fastest_torch([official, record("torch_graph", 0.1, internal_correctness=True)]) == "graph"
    )


def test_failures_are_not_quality_failures():
    assert failure_kind("BlockingIOError unable to lock file") == "implementation_failure"
    assert failure_kind("AssertionError not close") == "numerical_correctness_failure"
    assert failure_kind("FileNotFoundError weights") == "dependency_or_asset_missing"


def test_boundary_pauses_before_launch(tmp_path):
    atomic_json(tmp_path / "jobs/prior/result.json", dict(status="completed"))
    (tmp_path / "PAUSE_AFTER_JOB").write_text("prior")
    suite = Suite({}, tmp_path)
    with pytest.raises(BoundaryPause):
        suite.job("next", {})
    assert not (tmp_path / "jobs/next").exists()


def test_jax_phase_cannot_enter_torch(tmp_path, monkeypatch):
    suite = Suite({}, tmp_path)
    seen = []
    monkeypatch.setattr(
        suite, "calibration", lambda candidates: {k: dict(status="completed") for k in candidates}
    )
    monkeypatch.setattr(suite, "worthwhile", lambda *a: True)

    def inference(p, *args):
        seen.append(p.backend)
        return dict(status="completed")

    monkeypatch.setattr(suite, "inference", inference)
    with pytest.raises(BoundaryPause):
        suite.run("run", "jax")
    assert set(seen) == {"jax"}


def test_inheritance_rejects_environment_before_copy(tmp_path):
    from cognitive_ultrasound.compute_lab.migration import inherit

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    atomic_json(old / "status.json", dict(status="stopped"))
    atomic_json(old / "identity.json", dict(environment="old"))
    atomic_json(old / "config.json", {})
    with pytest.raises(ValueError, match="incompatible"):
        inherit(old, new, {}, "different_gpu")
    assert not (new / "jobs").exists()


def test_inheritance_rejects_changed_scientific_code(monkeypatch):
    from cognitive_ultrasound.compute_lab import migration

    monkeypatch.setattr(
        migration,
        "source_identity",
        lambda: {"src/cognitive_ultrasound/compute_lab/inference.py": "new"},
    )
    with pytest.raises(ValueError, match="Unreviewed"):
        migration.validate_source({"src/cognitive_ultrasound/compute_lab/inference.py": "old"})


def test_failed_jobs_are_kept_as_evidence_not_inherited(tmp_path, monkeypatch):
    from cognitive_ultrasound.compute_lab import migration

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    atomic_json(old / "status.json", dict(status="stopped"))
    atomic_json(old / "identity.json", dict(environment="gpu", source={}))
    atomic_json(old / "config.json", {})
    atomic_json(old / "jobs/short_torch_compile_b14/result.json", dict(status="failed"))
    atomic_json(old / "jobs/development_official_b14/result.json", record("official", 1))
    monkeypatch.setattr(migration, "validate_source", lambda old: [])
    before = (old / "identity.json").read_bytes()
    migration.inherit(old, new, {}, "gpu")
    assert (new / "jobs/development_official_b14/result.json").exists()
    assert not (new / "jobs/short_torch_compile_b14").exists()
    assert (new / "inherited_evidence/short_torch_compile_b14/result.json").exists()
    assert (old / "identity.json").read_bytes() == before
    assert read_json(new / "inheritance.json")["original_evidence_untouched"]


def test_setup_required_before_background_launch():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/run_compute_lab.sh").read_text()
    assert launcher.index("readiness check") < launcher.index("nohup")
    setup = (root / "scripts/setup_compute_lab.sh").read_text()
    assert (
        setup.index("rm -f .cache/compute-ready.json")
        < setup.index("pip install")
        < setup.index("readiness write")
    )


def test_capture_validator_rejects_stale_outputs():
    import torch

    from cognitive_ultrasound.torch_casl.native import validate_capture

    class Model:
        def frame(self, *args, **kw):
            return tuple(x.square() for x in args)

    model = Model()
    args = tuple(torch.ones(3) * i for i in (1, 2, 3, 4))
    with pytest.raises(AssertionError):
        validate_capture(lambda *x: model.frame(*args), model, args, 50)


def test_capture_validator_accepts_changed_gradient_outputs():
    import torch

    from cognitive_ultrasound.torch_casl.native import validate_capture

    class Model:
        def frame(self, a, b, c, d, **kw):
            return (
                torch.func.grad(lambda x: ((x + b) * c * d).square().sum())(a),
                b + c,
                c * d,
                d.square(),
            )

    model = Model()
    args = tuple(torch.ones(3) * i for i in (1, 2, 3, 4))
    validate_capture(model.frame, model, args, 50)


def test_reviewed_data_hygiene_is_the_only_compatible_io_change(monkeypatch):
    import hashlib

    from cognitive_ultrasound.compute_lab import migration
    from cognitive_ultrasound.provenance import sha256

    paths = [
        "src/cognitive_ultrasound/compute_lab/data.py",
        "src/cognitive_ultrasound/compute_lab/engines.py",
        "src/cognitive_ultrasound/torch_casl/native.py",
    ]
    old = {p: hashlib.sha256(migration.git_bytes(p)).hexdigest() for p in paths}
    current = {p: sha256(migration.ROOT / p) for p in paths}
    monkeypatch.setattr(migration, "source_identity", lambda: current)
    assert "src/cognitive_ultrasound/compute_lab/data.py" in migration.validate_source(old)


def test_hdf5_lock_not_inherited_by_exec_child(tmp_path):
    import os
    import subprocess
    import sys

    import h5py

    from cognitive_ultrasound.compute_lab.data import Writer

    if os.name != "posix":
        pytest.skip("POSIX descriptor/exec regression; exercised on AutoDL")
    file = tmp_path / "trajectory.h5"
    # Reproduce the original C++ helper pattern first, without changing lock policy.
    for secure in (False, True):
        owner = Writer(file) if secure else None
        handle = owner.h5 if owner else h5py.File(file, "w")
        handle["x"] = [1]
        assert os.get_inheritable(handle.id.get_vfd_handle()) is (not secure)
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
            close_fds=False,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout.readline().strip() == "ready"
            owner.close() if owner else handle.close()
            if secure:
                with h5py.File(file) as reopened:
                    assert reopened["x"][0] == 1
            else:
                with pytest.raises(BlockingIOError):
                    h5py.File(file)
        finally:
            child.terminate()
            child.wait(timeout=5)
