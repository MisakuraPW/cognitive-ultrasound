import sys
import tarfile
from dataclasses import asdict

import h5py
import numpy as np
import pytest
import yaml

from cognitive_ultrasound.compute_lab.data import Writer, frames, lock_manifest, throughput
from cognitive_ultrasound.compute_lab.protocol import (
    Profile,
    aggregate,
    numeric,
    profiles,
    quality,
    select,
)
from cognitive_ultrasound.preparation.common import atomic_json, read_json


def rows(n=32, delta=0.0, frame_counts=(2, 5)):
    return [
        dict(
            case=f"p{i}",
            seed=seed,
            frame=f,
            psnr=20 - delta,
            ssim=0.8 - delta / 100,
            mae=0.1 * (1 + delta / 10),
        )
        for i in range(n)
        for seed in (42, 31415, 271828)
        for f in range(frame_counts[i % len(frame_counts)])
    ]


def test_fixed_candidates_and_classification():
    candidates = profiles("graph", ["jax_constants", "jax_prefetch"])
    assert len(candidates) == 20
    assert candidates["jax_25_fp16"].family == "combination"
    assert candidates["torch_eager"].category == "A"
    assert candidates["jax_dps5"].family == "sampling"
    assert all(
        p.record()["particles"] == 2 and p.record()["cold_steps"] == 500
        for p in candidates.values()
    )
    assert not any(p.record()["automatic_adoption"] for p in candidates.values())


def test_numeric_difference_and_nonfinite_never_pass():
    assert numeric([1.0], [1.0 + 1e-5])["passed"]
    assert not numeric([1.0], [1.0 + 1e-5])["bitwise"]
    assert not numeric([1.0], [1.0 + 1e-5], discrete=True)["passed"]
    assert not numeric([np.nan], [1.0])["passed"]
    assert not numeric([1, 2], [1])["passed"]
    assert not numeric(np.array([0.0]), np.array([-0.0]))["bitwise"]


def test_patient_bootstrap_and_worst_patient_guard():
    reference = rows()
    assert quality(reference, rows(delta=0.02), confirmation=True)["passed"]
    assert not quality(reference, rows(delta=0.11), confirmation=True)["passed"]
    bad = rows()
    for r in bad:
        if r["case"] == "p0":
            r["psnr"] -= 0.6
    result = quality(reference, bad, confirmation=True)
    assert result["mean_loss"][0] < 0.1 and not result["passed"]
    assert not quality(reference, bad[:-1])["passed"]
    assert not quality(rows(8), rows(8), confirmation=True)["passed"]
    with pytest.raises(ValueError, match="Duplicate"):
        quality(reference + [reference[0]], bad + [bad[0]])


def test_equal_patient_seed_weight_not_pooled_frames():
    r = rows(2)
    for row in r:
        row["psnr"] = 10 if row["case"] == "p0" else 30
    assert np.mean([v[0] for v in aggregate(r).values()]) == 20
    assert np.mean([x["psnr"] for x in r]) != 20


def test_selection_cannot_use_confirmation_or_failed_quality():
    def record(p, speed, cohort="development", passed=True, equivalent=True):
        return dict(
            profile=p.record(),
            closed_loop_s=speed,
            cohort=cohort,
            quality=dict(passed=passed),
            equivalence_passed=equivalent,
        )

    data = [
        record(Profile("official"), 3),
        record(Profile("bad"), 0.1, passed=False),
        record(Profile("confirmation_leak"), 0.001, "confirmation"),
        record(Profile("drift"), 0.01, equivalent=False),
        record(Profile("good"), 2),
        record(Profile("half", precision="fp16"), 1, equivalent=False),
    ]
    assert select(data) == ["good", "half"]


@pytest.fixture
def h5(tmp_path):
    file = tmp_path / "example.h5"
    x = np.random.default_rng(42).uniform(-60, 0, (9, 112, 112)).astype(np.float32)
    with h5py.File(file, "w") as f:
        f.create_dataset("data/image", data=x)
    return file, (x / 30 + 1)[..., None]


@pytest.mark.parametrize("mode", ["serial", "prefetch", "cache"])
def test_io_order_content_and_owned_outputs(h5, mode):
    file, expected = h5
    actual = list(frames(file, len(expected), mode))
    np.testing.assert_array_equal(actual, expected)
    actual[0][:] = 99
    np.testing.assert_array_equal(actual[1], expected[1])


def test_prefetch_exception_is_not_suppressed(h5):
    file, _ = h5
    with h5py.File(file, "r+") as f:
        f["data/image"][5, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        list(frames(file, 9, "prefetch"))


def test_async_writer_no_drop_reorder_or_buffer_alias(tmp_path):
    file = tmp_path / "out.h5"
    writer = Writer(file, True)
    buffer = np.zeros((8, 8))
    for i in range(30):
        buffer[:] = i
        writer.append(dict(image=buffer))
    buffer[:] = -1
    writer.close()
    with h5py.File(file) as f:
        assert len(f) == 30
        for i in range(30):
            assert (f[str(i)]["image"][()] == i).all()


def test_async_writer_failure_and_nonfinite_rejected(tmp_path):
    writer = Writer(tmp_path / "a.h5", True)
    with pytest.raises(FloatingPointError):
        writer.append(dict(x=np.array([np.inf])))
    writer.h5.create_group("0")
    writer.append(dict(x=np.ones(3)))
    with pytest.raises(ValueError):
        writer.close()


def test_cohorts_locked_before_results_and_test_excluded(tmp_path):
    names = [f"0X{i:016X}.hdf5" for i in range(43)]
    history = tmp_path / "history"
    atomic_json(history / "manifest.json", dict(case=names[0]))
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump(dict(train=["train.hdf5"], val=names, test=["test.hdf5"])))
    root = tmp_path / "data/val"
    root.mkdir(parents=True)
    for name in names:
        with h5py.File(root / name, "w") as h:
            h.create_dataset(
                "data/image", data=np.zeros((3, 112, 112), np.float32), compression="gzip"
            )
    cfg = dict(
        split_manifest=str(split),
        history_roots=[str(history)],
        data_root=str(root.parent),
        cohort_seed=19,
    )
    output = tmp_path / "output"
    manifest = lock_manifest(cfg, output)
    groups = manifest["cohorts"]
    assert list(map(len, groups.values())) == [2, 8, 32]
    assert len(set(sum(groups.values(), []))) == 42
    assert names[0] not in sum(groups.values(), [])
    assert manifest == lock_manifest(cfg, output)
    with h5py.File(root / groups["debug"][0], "r+") as h:
        h["data/image"][0, 0, 0] = -10
    with pytest.raises(ValueError, match="Locked input"):
        lock_manifest(cfg, output)


def test_throughput_hashes(h5, tmp_path):
    file, _ = h5
    val = tmp_path / "val"
    val.mkdir()
    file.rename(val / file.name)
    result = throughput(
        dict(data_root=str(tmp_path)),
        dict(cohorts=dict(debug=[file.name]), files={file.name: dict(frames=9)}),
        tmp_path,
    )
    assert len(result["records"]) == 10 and not result["scientific_configuration_changed"]


def test_result_archive_roundtrip_without_reference_cache(tmp_path):
    from cognitive_ultrasound.compute_lab.report import archive

    source = tmp_path / "run"
    atomic_json(source / "result.json", dict(passed=False))
    atomic_json(source / ".cache/frame.json", dict(raw=True))
    out = tmp_path / "results.tar.gz"
    result = archive(source, out)
    with tarfile.open(out) as tar:
        assert tar.getnames() == ["result.json"]
    assert result["reference_cache_excluded"]
    assert out.with_name(out.name + ".sha256").read_bytes().endswith(b"\n")


def test_resume_does_not_rerun_completed_or_failed_jobs(tmp_path):
    from cognitive_ultrasound.compute_lab.suite import Stopped, Suite

    suite = Suite({}, tmp_path)
    atomic_json(tmp_path / "jobs/done/result.json", dict(status="completed", x=1))
    atomic_json(tmp_path / "jobs/bad/result.json", dict(status="failed"))
    assert suite.job("done", {})["x"] == 1
    assert suite.job("bad", {})["status"] == "failed"
    assert suite.state["failed"] == ["bad"]
    (tmp_path / "STOP").touch()
    with pytest.raises(Stopped):
        suite.job("pending", {})


def test_state_checks_reject_action_or_history_changes():
    from cognitive_ultrasound.compute_lab.inference import state_checks

    a = {
        k: np.ones(3)
        for k in (
            "prediction",
            "uncertainty",
            "next_action",
            "mask",
            "resume_mask",
            "resume_buffer",
            "resume_posterior_samples",
        )
    }
    b = {k: v.copy() for k, v in a.items()}
    b["next_action"][0] = 0
    b["resume_buffer"][1] = 2
    result = state_checks(a, b)
    assert not result["next_action"]["passed"] and not result["resume_buffer"]["passed"]


def test_hardware_or_science_change_invalidates_reuse(tmp_path):
    from cognitive_ultrasound.compute_lab.suite import ensure_identity

    file = tmp_path / "identity.json"
    identity = dict(environment="gpu4090_lib1", science="weights0_split0_fp32", source="code0")
    ensure_identity(file, identity)
    ensure_identity(file, identity)
    for key in identity:
        with pytest.raises(ValueError, match="NEW output"):
            ensure_identity(file, dict(identity, **{key: "different"}))


def test_worker_exception_is_recorded_and_no_precision_fallback(tmp_path):
    import os
    import subprocess

    from cognitive_ultrasound.config import ROOT

    cfg = dict(
        checkpoint="checkpoints/official",
        split_manifest="configs/splits/split.yaml",
        casl_training_config="configs/training.yaml",
        bf_config="configs/belief_filter/pilot.yaml",
    )
    atomic_json(tmp_path / "config.json", cfg)
    atomic_json(
        tmp_path / "task.json",
        dict(
            kind="inference",
            output=str(tmp_path / "worker"),
            profile=asdict(Profile("invalid")),
            cohort="debug",
            budget=14,
        ),
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "cognitive_ultrasound.compute_lab",
            "worker",
            "--config",
            str(tmp_path / "config.json"),
            "--task",
            str(tmp_path / "task.json"),
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        env=env,
    )
    assert p.returncode != 0
    assert read_json(tmp_path / "worker/result.json")["status"] == "failed"
