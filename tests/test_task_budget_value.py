"""Protocol/state/lifecycle checks; explicit synthetic values are not GPU evidence."""

import copy
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.preparation.common import atomic_json, digest
from cognitive_ultrasound.provenance import sha256
from cognitive_ultrasound.task_budget import value_suite, value_workers
from cognitive_ultrasound.task_budget.protocol import clip_indices, task_scores
from cognitive_ultrasound.task_budget.value_protocol import (
    LEVELS,
    anchors,
    candidates,
    common_unobserved,
    coverage,
    hybrid,
    periodic,
    predictability,
    quality,
    reverse,
    uniform,
)
from cognitive_ultrasound.task_budget.value_runtime import Runtime, trajectory
from cognitive_ultrasound.task_budget.value_storage import (
    commit,
    committed,
    compare_trees,
    load_tree,
    save_tree,
    tree_digest,
)


@pytest.fixture
def settings(tmp_path):
    c = value_suite.config(ROOT / "configs/task_budget_value.yaml")
    d = c["value_diagnostics"]
    d.update(functional_fixture=True, skip_fixture_plots=True, checkpoint_frames=16)
    source = tmp_path / "source"
    source.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    c["data_root"] = str(data)
    d["source_batch"] = str(source)
    d["historical_roots"] = [str(tmp_path)]
    names = []
    files = {}
    for i, n in enumerate([64, 68]):
        name = f"fixture{i}.hdf5"
        names.append(name)
        xs = np.linspace(-0.85, 0.75, 112)[None, None, :]
        depth = np.linspace(-0.05, 0.05, 112)[None, :, None]
        time = 0.1 * np.sin(np.arange(n)[:, None, None] / 5)
        frames = np.clip(xs + depth + time, -0.99, 0.99).astype(np.float32)
        p = data / name
        with h5py.File(p, "w") as h:
            h["data/image"] = (frames - 1) * 30
        files[name] = dict(
            relative=name,
            frames=n,
            ef=float(50 + 20 * frames.mean()),
            sha256=sha256(p),
            task_split="VAL",
        )
    original = dict(
        cohorts={"development": names, "train": [], "confirmation": []}, files=files, sources={}
    )
    original["identity"] = digest(original)
    atomic_json(source / "manifest.json", original)
    prior = copy.deepcopy(c)
    prior["functional_fixture"] = True
    prior["feature_scales"] = [1.0] * 12
    atomic_json(source / "config.json", prior)
    atomic_json(source / "status.json", {"status": "completed"})
    root = tmp_path / "run"
    c, m = value_suite.prepare(c, root)
    return c, m, root


@pytest.mark.parametrize("n", [5, 8, 64, 65, 128, 261])
def test_exact_per_video_resource_and_calls(n):
    m = uniform(n, LEVELS["M"])
    for arm, high, s in candidates(n):
        expected = periodic(n, len(high))
        assert Counter(map(tuple, s[1:])) == Counter(map(tuple, expected[1:]))
        assert (
            (s[0] == [10, 4]).all() and (s[-((n - 1) % 4) :] == m[-((n - 1) % 4) :]).all()
            if (n - 1) % 4
            else (s[0] == [10, 4]).all()
        )
        if arm == "same":
            assert int(s.sum()) == int(m.sum())
        assert int(s.sum()) == int(reverse(s).sum())
        assert sum(s[:, 1] > 0) == sum(expected[:, 1] > 0)


@pytest.mark.parametrize("n", [64, 68, 113, 261])
def test_all_start_temporal_coverage(n):
    old = coverage(clip_indices(n, 32, 2, 16), n)
    new = coverage(clip_indices(n, 32, 2, 1), n)
    assert new["complete"] and new["direct_frames"] == n
    assert old["direct_frames"] <= new["direct_frames"]
    assert len(new["appearances"]) == n


def test_short_coverage_not_silently_claimed_complete():
    assert not coverage(clip_indices(63, 32, 2, 1), 63)["complete"]


def test_joint_uncertainty_and_task_sensitivity():
    p = np.array([[[1.0, 1.0, 2.0]], [[1.0, 3.0, 4.0]]])
    g = np.array([[[100.0, 0.0, 2.0]], [[100.0, 0.0, 2.0]]])
    scores = task_scores(p, g)
    assert scores[0] == 0 and scores[1] == 0 and scores[2] > 0
    with pytest.raises(FloatingPointError):
        task_scores(p, np.full_like(g, np.nan))


def test_common_unobserved_and_nonfinite_rejection():
    target = np.zeros((2, 2, 2))
    a = np.ones_like(target)
    b = 2 * a
    m = np.zeros_like(target)
    m[:, :, 0] = 1
    x = common_unobserved(target, [a, b], [m, np.zeros_like(m)])
    assert x["pixels"] == 4 and x["mse"] == [1.0, 4.0]
    assert common_unobserved(target, [a, b], [np.ones_like(m), m])["mse"] == [None, None]
    with pytest.raises(ValueError):
        quality(target, np.full_like(a, np.nan))


def test_hybrid_never_changes_the_shared_cold_and_tail():
    n = 68
    low = np.zeros((n, 2, 2))
    high = np.ones_like(low)
    medium = 0.5 * high
    x = hybrid(low, medium, high, [1, 3])
    q = (n - 1) // 4
    np.testing.assert_array_equal(x[0], medium[0])
    np.testing.assert_array_equal(x[1 + 4 * q :], medium[1 + 4 * q :])


def test_pickle_free_state_integrity_and_aliasing(tmp_path):
    state = {
        "array": np.arange(6, dtype=np.float32),
        "history": [np.ones((2, 2))],
        "none": None,
        "rng": {"key": 123},
    }
    p = tmp_path / "s.npz"
    save_tree(p, state)
    x = load_tree(p)
    compare_trees(state, x)
    assert tree_digest(state) == tree_digest(x)
    x["array"][0] = 99
    assert state["array"][0] == 0
    commit(tmp_path, {"id": 1}, {"ok": True}, ["s.npz"])
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        committed(tmp_path, {"id": 1})
    with pytest.raises(ValueError):
        save_tree(tmp_path / "bad.npz", {"a": np.array([np.nan])})


@pytest.mark.parametrize("point", ["before", "middle"])
def test_actual_closed_loop_restore_and_plus7_branch(settings, point):
    cfg, manifest, root = settings
    name = manifest["cohorts"]["development"][0]
    rt = Runtime(cfg, manifest, root, "P1")
    try:
        a = trajectory(rt, name, 42, uniform(64, LEVELS["M"]))
        at = anchors(64)[1]
        state = load_tree(a[4] / f"{point}_{at:05d}.npz")
        b = trajectory(
            rt, name, 42, uniform(64, LEVELS["M"]), state, (a[0][:at], a[1][:at], a[2][:at])
        )
        np.testing.assert_allclose(a[0], b[0], atol=2e-4, rtol=2e-4)
        np.testing.assert_array_equal(a[1], b[1])
        seq = uniform(64, LEVELS["M"])
        seq[at] = (14, 7) if point == "before" else (7, 14)
        c = trajectory(rt, name, 42, seq, state, (a[0][:at], a[1][:at], a[2][:at]))
        assert c[3]["total_lines"] - a[3]["total_lines"] == 7
        np.testing.assert_array_equal(c[0][:at], a[0][:at])
        assert any(not np.array_equal(x, y) for x, y in zip(c[0][at:], a[0][at:]))
    finally:
        rt.close()


def test_chunk_resume_skips_committed_frames(settings):
    cfg, manifest, root = settings
    name = manifest["cohorts"]["development"][0]
    rt = Runtime(cfg, manifest, root, "P1")
    original = rt.progress

    def stop(**kw):
        if kw.get("frame") == 18:
            (root / "STOP").write_text("stop")
        original(**kw)

    rt.progress = stop
    with pytest.raises(InterruptedError):
        trajectory(rt, name, 42, uniform(64, LEVELS["M"]))
    rt.close()
    (root / "STOP").unlink()
    again = Runtime(cfg, manifest, root, "P1")
    seen = []
    original = again.progress

    def track(**kw):
        if "frame" in kw:
            seen.append(kw["frame"])
        original(**kw)

    again.progress = track
    try:
        a = trajectory(again, name, 42, uniform(64, LEVELS["M"]))
        assert min(seen) == 17 and a[3]["frames"] == 64
        seen.clear()
        trajectory(again, name, 42, uniform(64, LEVELS["M"]))
        assert not seen
    finally:
        again.close()


def test_mixed_middle_first_budget_is_rejected(settings):
    cfg, manifest, root = settings
    rt = Runtime(cfg, manifest, root, "P1")
    name = manifest["cohorts"]["development"][0]
    try:
        a = trajectory(rt, name, 42, uniform(64, LEVELS["M"]))
        at = anchors(64)[1]
        s = load_tree(a[4] / f"middle_{at:05d}.npz")
        schedule = uniform(64, LEVELS["M"])
        schedule[at] = (14, 7)
        with pytest.raises(ValueError, match="first acquisition"):
            trajectory(rt, name, 42, schedule, s, (a[0][:at], a[1][:at], a[2][:at]))
    finally:
        rt.close()


def test_lovo_keeps_all_anchors_of_a_patient_in_one_fold():
    rows = [
        dict(
            case=str(i),
            stage=stage,
            features=[i + j * 0.1, 1.0],
            gain={"mse_gain": i * 0.2 + j * 0.01, "ef_gain": i + j * 0.02},
        )
        for stage in ["first", "second"]
        for i in range(3)
        for j in range(4)
    ]
    results = predictability(rows)
    assert len(results) == 4
    for x in results:
        assert x["videos"] == 3
        for f in x["folds"]:
            assert f["held_out"] not in f["training_cases"]


def test_real_pipeline_all_phases_synthetic_and_complete_resume(settings):
    cfg, manifest, root = settings
    # Exercise the actual coordinator -> CLI worker -> atomic result boundaries.
    state = value_suite.run(cfg, root)
    assert state["status"] == "completed" and state["completed_phases"] == [
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
    ]
    p4 = json.loads((root / "jobs/P4/result.json").read_text())
    assert p4["status"] == "completed"
    assert p4["functional_fixture"] and len(p4["records"]) > 0
    records = json.loads((root / "jobs/P3/result.json").read_text())["records"]
    assert len(records) == 16
    calls = []
    state = value_suite.run(cfg, root, "resume", launch=lambda *a: calls.append(a))
    assert not calls and state["status"] == "completed"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["functional_fixture"]
    assert "CPU合成" in (root / "REPORT_中文.md").read_text(encoding="utf-8")
    receipt = json.loads((root / "bundle_receipt.json").read_text())
    assert receipt["verified"]
    import tarfile

    with tarfile.open(receipt["archive"]) as t:
        assert not any(x.name.endswith(".npz") for x in t)


def test_identity_and_input_change_do_not_silently_resume(settings):
    cfg, manifest, root = settings
    c = copy.deepcopy(cfg)
    c["value_diagnostics"]["ef_margin_pp"] = 0.8
    with pytest.raises(ValueError, match="config changed"):
        value_suite.prepare(c, root)
    p = (
        Path(cfg["data_root"])
        / manifest["files"][manifest["cohorts"]["development"][0]]["relative"]
    )
    with h5py.File(p, "a") as h:
        h["data/image"][0, 0, 0] = -10.0
    with pytest.raises(ValueError, match="input changed"):
        value_suite.prepare(cfg, root)


def test_extension_not_automatic_or_allowed_before_parent_complete(settings):
    cfg, manifest, root = settings
    assert not cfg["value_diagnostics"]["expanded"]
    with pytest.raises(ValueError, match="completed initial"):
        value_suite.extension(cfg, {"cohorts": {}})


def test_corrupted_ef_cache_is_rejected(settings):
    cfg, manifest, root = settings
    rt = Runtime(cfg, manifest, root, "P2")
    images = np.zeros((64, 112, 112), np.float32)
    try:
        rt.readout(images, 50)
        p = next((root / "ef_cache").rglob("*.json"))
        x = json.loads(p.read_text())
        x["prediction"] = 100
        atomic_json(p, x)
        with pytest.raises(ValueError, match="cache corrupted"):
            rt.readout(images, 50)
    finally:
        rt.close()


def test_cli_status_without_assets(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cognitive_ultrasound.task_budget",
            "value-status",
            "--output",
            str(tmp_path / "unused"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert json.loads(result.stdout) == {"status": "not_started", "coordinator_alive": False}


def test_atomic_windows_reader_retry_is_bounded_and_keeps_atomicity(tmp_path, monkeypatch):
    from cognitive_ultrasound.preparation import common

    a = tmp_path / "temporary"
    b = tmp_path / "result"
    a.write_text("new")
    b.write_text("old")
    original = Path.replace
    attempts = []

    def replace(self, target):
        attempts.append(self)
        if len(attempts) < 3:
            raise PermissionError("simulated reader")
        return original(self, target)

    monkeypatch.setattr(common.os, "name", "nt")
    monkeypatch.setattr(Path, "replace", replace)
    common.atomic_replace(a, b)
    assert len(attempts) == 3 and b.read_text() == "new" and not a.exists()


def test_windows_read_retries_only_transient_permission(tmp_path, monkeypatch):
    from cognitive_ultrasound.preparation import common

    p = tmp_path / "progress.json"
    p.write_text('{"frame":7}')
    original = Path.read_text
    attempts = []

    def read(self, *a, **kw):
        attempts.append(self)
        if len(attempts) < 3:
            raise PermissionError("simulated rename")
        return original(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(common.os, "name", "nt")
    assert common.read_json(p) == {"frame": 7} and len(attempts) == 3


def test_runtime_failure_is_not_a_scientific_negative(settings):
    cfg, manifest, root = settings

    def launch(p, c, m, r):
        if p == "P0":
            atomic_json(r / "jobs/P0/result.json", {"status": "completed", "process_wall_s": 0})
        else:
            raise MemoryError("synthetic simulated OOM")

    with pytest.raises(MemoryError):
        value_suite.run(cfg, root, launch=launch)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and status["failures"] == ["P1"]
    assert not (root / "jobs/P4/result.json").exists()


def test_native_abort_cleanup_still_targets_worker_group(monkeypatch):
    class Process:
        pid = 123456

        def poll(self):
            return -6

    called = []
    monkeypatch.setattr(value_suite.os, "name", "posix")
    monkeypatch.setattr(
        value_suite.os, "killpg", lambda pid, sig: called.append((pid, sig)), raising=False
    )
    monkeypatch.setattr(value_suite.signal, "SIGKILL", 9, raising=False)
    value_suite.cleanup_worker(Process())
    assert [pid for pid, sig in called] == [123456, 123456]


def test_exact_extension_filters_history_and_locks_strata(settings, tmp_path):
    import csv

    import yaml

    cfg, manifest, root = settings
    parent = tmp_path / "initial_done"
    parent.mkdir()
    atomic_json(parent / "status.json", {"status": "completed"})
    atomic_json(parent / "identity.json", {"fixture": "parent"})
    cfg["value_diagnostics"]["expansion_parent"] = str(parent)
    names = []
    labels = []
    for group, (count, ef) in enumerate([(8, 25), (6, 50), (6, 70)]):
        for i in range(count):
            name = f"new_{group}_{i}.hdf5"
            names.append(name)
            p = Path(cfg["data_root"]) / "val" / name
            p.parent.mkdir(exist_ok=True)
            with h5py.File(p, "w") as h:
                h["data/image"] = np.zeros((96, 112, 112), np.float32)
            labels.append({"FileName": Path(name).stem, "EF": ef, "Split": "VAL"})
    splits = tmp_path / "splits.yaml"
    splits.write_text(yaml.safe_dump({"train": [], "val": names, "test": []}))
    cfg["split_manifest"] = str(splits)
    f = tmp_path / "labels.csv"
    with f.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=["FileName", "EF", "Split"])
        w.writeheader()
        w.writerows(labels)
    cfg["file_list"] = str(f)
    hist = tmp_path / "history"
    cfg["value_diagnostics"]["historical_roots"] = [str(hist)]
    atomic_json(hist / "old/manifest.json", {"cohorts": {"development": ["val/new_0_0.hdf5"]}})
    a = value_suite.extension(cfg, {"cohorts": manifest["cohorts"]})
    b = value_suite.extension(cfg, {"cohorts": manifest["cohorts"]})
    assert a[0] == b[0] and len(a[0]) == 16 and "new_0_0.hdf5" not in a[0]
    assert [sum(1 for x in a[0] if x.startswith(f"new_{i}_")) for i in range(3)] == [6, 5, 5]


def test_paired_report_does_not_confuse_cost_or_quality_equivalence():
    from cognitive_ultrasound.task_budget.value_report import (
        paired_schedule_statistics,
        schedule_statistics,
    )

    cfg = {"value_diagnostics": {"ef_margin_pp": 0.5, "mse_margin": 0.05, "p90_margin": 0.1}}
    rows = []
    for case in ["a", "b"]:
        for seed in [42, 31415]:
            for alias, error, mse, lines in [
                ("baseline_M", 1.0, 0.1, 14),
                ("saving_periodic", 2.0, 0.2, 10.5),
                ("saving_EF", 1.8, 0.18, 10.5),
            ]:
                rows.append(
                    dict(
                        case=case,
                        seed=seed,
                        aliases=[alias],
                        mean_lines=lines,
                        total_lines=int(lines * 64),
                        perception_calls=80 if lines < 14 else 128,
                        quality={"mse": mse, "frame_mse_p90": mse},
                        ef={"all_starts_v1": {"absolute_error": error}},
                    )
                )
    stats = schedule_statistics(rows, cfg)
    candidate = next(x for x in stats if x["alias"] == "saving_EF")
    assert not candidate["screen_passed"]
    paired = paired_schedule_statistics(rows)
    assert paired[0]["ef_delta"]["mean"] == pytest.approx(-0.2)
    bad = copy.deepcopy(rows)
    bad[-1]["total_lines"] += 1
    with pytest.raises(ValueError, match="costs"):
        paired_schedule_statistics(bad)


def test_cached_sensitivity_uses_matching_cold_and_tail_backgrounds(settings, monkeypatch):
    cfg, manifest, root = settings

    def cached(rt, name, level, seed=42):
        n = manifest["files"][name]["frames"]
        value = {"L": -0.3, "M": 0.1, "H": 0.4}[level]
        return (np.full((n, 112, 112), value, np.float32), None, None, None, None)

    monkeypatch.setattr(value_workers, "fixed", cached)
    rt = Runtime(cfg, manifest, root, "P2")
    try:
        result = value_workers.p2(rt)
        assert len(result["records"]) == 36
        for row in result["records"]:
            assert not row["physical_closed_loop"] and not row["budget_claim_allowed"]
            if row["arm"] in ["repair", "damage"]:
                background = result["backgrounds"][row["case"]][row["arm"]]
                expected = (
                    background["ef"]["all_starts_v1"]["absolute_error"]
                    - row["ef"]["all_starts_v1"]["absolute_error"]
                )
                assert row["ef_gain_from_background"] == pytest.approx(expected)
    finally:
        rt.close()
