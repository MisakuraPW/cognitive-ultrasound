"""Physical causality, exact costs, genuine gradients and pilot-to-full recovery."""

import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest
import yaml

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.preparation.common import atomic_json, digest, read_json
from cognitive_ultrasound.provenance import sha256
from cognitive_ultrasound.task_budget import stage_data, stage_suite, stage_workers
from cognitive_ultrasound.task_budget.stage_policy import Controller, initialize, logits
from cognitive_ultrasound.task_budget.stage_protocol import (
    ARMS,
    checkpoint_choice,
    curriculum,
    dual_update,
    quota_mask,
    training_objective,
)
from cognitive_ultrasound.task_budget.value_protocol import uniform
from cognitive_ultrasound.task_budget.value_runtime import Runtime, trajectory
from cognitive_ultrasound.task_budget.value_storage import committed, load_tree, tree_digest


@pytest.fixture
def settings(tmp_path):
    c = stage_data.config(ROOT / "configs/task_budget_stages.yaml")
    c["cohorts"] = dict(train=3, development=2, confirmation=2)
    c["stage_budget"].update(
        functional_fixture=True, pilot_train=1, pilot_dev=1, hidden=4, history_frames=3
    )
    c["value_diagnostics"]["functional_fixture"] = True
    c["feature_scales"] = [1.0] * 12
    data = tmp_path / "data"
    source = tmp_path / "source"
    source.mkdir()
    splits = dict(train=[], val=[], test=[])
    records, files = [], {}
    for group, split, count in [("TRAIN", "train", 4), ("VAL", "val", 3), ("TEST", "test", 4)]:
        for i in range(count):
            name = f"{split}{i}.hdf5"
            splits[split].append(name)
            ef = 30 + i * 15
            records.append(dict(FileName=Path(name).stem, Split=group, EF=ef))
            p = data / split / name
            p.parent.mkdir(parents=True, exist_ok=True)
            a = (
                np.linspace(-0.8, 0.7, 112)[None, None, :]
                + np.linspace(-0.05, 0.05, 112)[None, :, None]
            )
            a = np.broadcast_to(a, (64, 112, 112)).copy() + 0.03 * np.sin(
                np.arange(64)[:, None, None] / 5 + i
            )
            with h5py.File(p, "w") as h:
                h["data/image"] = ((a - 1) * 30).astype(np.float32)
            files[name] = dict(
                relative=f"{split}/{name}", frames=64, ef=ef, sha256=sha256(p), task_split=group
            )
    labels = tmp_path / "FileList.csv"
    with labels.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["FileName", "Split", "EF"])
        writer.writeheader()
        writer.writerows(records)
    split_file = tmp_path / "split.yaml"
    split_file.write_text(yaml.safe_dump(splits))
    c.update(data_root=str(data), file_list=str(labels), split_manifest=str(split_file))
    c["stage_budget"].update(source_batch=str(source), historical_roots=[str(tmp_path)])
    old = dict(
        cohorts=dict(train=["train0.hdf5"], development=["val0.hdf5"], confirmation=["test0.hdf5"]),
        files={n: files[n] for n in ["train0.hdf5", "val0.hdf5", "test0.hdf5"]},
        sources={k: sha256(c[k]) for k in ["file_list", "split_manifest"]},
    )
    old["identity"] = digest(old)
    atomic_json(source / "manifest.json", old)
    root = tmp_path / "run"
    return c, root


@pytest.mark.parametrize("arm", ["T1", "T2"])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 63, 128, 261])
def test_exact_quota_without_tail_repair(arm, n):
    levels = [4, 7, 10, 14] if arm == "T1" else [0, 2, 4, 7, 14]
    remaining = 7 * n
    rng = np.random.default_rng(17)
    used = []
    for left in range(n, 0, -1):
        legal = quota_mask(levels, remaining, left)
        k = levels[int(rng.choice(np.flatnonzero(legal)))]
        remaining -= k
        used.append(k)
    assert remaining == 0 and sum(used) == 7 * n


def test_pilot_prefix_has_no_reset_and_covers64():
    c = stage_data.config(ROOT / "configs/task_budget_stages.yaml")
    names = [str(i) for i in range(64)]
    stream, ends = curriculum(names, c)
    assert ends == {1: 72, 2: 136, 3: 200, 4: 264}
    assert len(set(x["case"] for x in stream[:16])) == 8
    assert not set(x["case"] for x in stream[16:72]) & set(x["case"] for x in stream[:16])
    assert set(x["case"] for x in stream[:72]) == set(names)
    for i, x in enumerate(stream):
        assert x["visit"] == sum(v["case"] == x["case"] for v in stream[:i])


def test_manifest_lock_and_fresh_confirmation(settings):
    c, root = settings
    c, m = stage_data.prepare(c, root)
    assert len(m["cohorts"]["train"]) == 3
    assert "train0.hdf5" in m["cohorts"]["train"]
    assert "test0.hdf5" not in m["cohorts"]["confirmation"]
    assert m["pilot_train"] == [m["curriculum"][0]["case"]]
    assert stage_data.prepare(read_json(root / "requested_config.json"), root)[1] == m
    m["files"]["train0.hdf5"]["ef"] = 99
    atomic_json(root / "manifest.json", m)
    with pytest.raises(ValueError, match="corrupted"):
        stage_data.prepare(read_json(root / "requested_config.json"), root)


def test_current_truth_cannot_enter_first_policy(settings):
    c, root = settings
    params = initialize(42, c, "T1")
    actors = [Controller(params, c, "T1", 64) for _ in range(2)]
    for actor in actors:
        actor.select(
            0,
            1,
            np.zeros(12),
            np.ones(4, bool),
            np.random.default_rng(9),
            particles=np.zeros((2, 112, 112)),
            prior=np.zeros((2, 112, 112)),
            past=[],
            observed=None,
            mask=None,
            k1=0,
        )
    assert tree_digest(actors[0].trace) == tree_digest(actors[1].trace)
    assert not {"target", "truth", "ef", "future"} & set(actors[0].snapshot())


def test_second_uses_pre_observation_innovation(settings):
    c, root = settings
    actor = Controller(initialize(42, c, "Q2"), c, "Q2", 64)
    mask = np.zeros((112, 112))
    mask[:, 0] = 1
    kwargs = dict(
        particles=np.ones((2, 112, 112)),
        prior=np.zeros((2, 112, 112)),
        past=[],
        observed=np.ones((112, 112)),
        mask=mask,
        k1=7,
    )
    actor.select(1, 1, np.zeros(12), np.ones(5, bool), np.random.default_rng(9), **kwargs)
    # posterior fits observed pixels perfectly, yet prior innovation is positive.
    assert actor.trace[0]["sequence"][-1, 12] > 0


def test_recurrent_history_has_a_real_gradient(settings):
    c, root = settings
    p = initialize(42, c, "T1")
    x = np.zeros((3, 18), np.float32)
    x[0, 0] = 1
    valid = np.ones(3, bool)
    y = logits(p, x, valid)
    z = logits(p, np.zeros_like(x), valid)
    assert not np.array_equal(y, z)
    g = jax.grad(lambda q: logits(q, x, valid)[0])(p)
    assert np.linalg.norm(g["wx"]) > 0 and np.linalg.norm(g["wh"]) > 0


def test_zero_budget_omits_real_second_dps(settings):
    c, root = settings
    c, m = stage_data.prepare(c, root)
    rt = Runtime(c, m, root, "test")
    name = m["pilot_development"][0]
    p = initialize(42, c, "Q2")
    p["bo"] = np.array([100, -100, -100, -100, -100], np.float32)
    controller = Controller(p, c, "Q2", 64)
    _, _, rows, r, _ = trajectory(rt, name, 42, uniform(64, (7, 7)), controller=controller)
    assert all(x["k2"] == 0 and x["reverse_steps_stage2"] == 0 for x in rows[1:])
    assert r["perception_calls"] == 65 and r["total_lines"] == 14 + 63 * 7


def test_learned_chunk_resume_equals_continuous(settings):
    c, root = settings
    c, m = stage_data.prepare(c, root)
    rt = Runtime(c, m, root, "test")
    name = m["pilot_development"][0]
    p = initialize(42, c, "T2")
    real = rt.progress

    def interrupt(**kw):
        if kw.get("frame") == 19:
            raise InterruptedError("simulated crash")
        real(**kw)

    rt.progress = interrupt
    with pytest.raises(InterruptedError):
        trajectory(rt, name, 42, uniform(64, (7, 7)), controller=Controller(p, c, "T2", 64))
    rt.progress = real
    a = Controller(p, c, "T2", 64)
    images, _, rows, _, _ = trajectory(rt, name, 42, uniform(64, (7, 7)), controller=a)
    # Separate trajectory identity solely for the uninterrupted fixture comparison.
    b = Controller(p, c, "T2", 64)
    b.identity["independent_fixture"] = True
    other, _, rows2, _, _ = trajectory(rt, name, 42, uniform(64, (7, 7)), controller=b)
    np.testing.assert_array_equal(images, other)
    assert [[r["k1"], r["k2"]] for r in rows] == [[r["k1"], r["k2"]] for r in rows2]
    assert tree_digest(a.trace) == tree_digest(b.trace)
    assert len(a.trace) == 63 and a.remaining == 0


def test_quality_is_not_an_ef_failure_gate(settings):
    c, root = settings
    x = dict(ef_error=4, mse=0.2, p90=0.3, warm_mean_lines=10)
    ref = dict(full_ef_error=3, mse=0.01, p90=0.02)
    loss, violation = training_objective("T1", x, ref, [1, 1], c)
    assert loss == 1 and violation is None
    loss, violation = training_objective("Q2", x, ref, [1, 1], c)
    assert violation.min() > 0
    assert (dual_update([1, 1], violation, c) > 1).all()


def test_checkpoint_selection_rejects_test_and_ci_is_not_gate(settings):
    c, root = settings
    rows = [
        dict(
            cohort="development", update=1, ef_error=5, mean_lines=14, mse_ratio=0.4, p90_ratio=0.5
        ),
        dict(
            cohort="development", update=2, ef_error=4, mean_lines=14, mse_ratio=0.8, p90_ratio=0.9
        ),
    ]
    assert checkpoint_choice("T1", rows, c)["chosen"]["update"] == 2
    assert not checkpoint_choice("Q2", rows, c)["chosen"]["feasible_point_estimate"]
    rows[0]["cohort"] = "confirmation"
    with pytest.raises(ValueError):
        checkpoint_choice("T1", rows, c)


def test_full_functional_lifecycle_and_resume(settings):
    c, root = settings
    c["cohorts"] = dict(train=2, development=1, confirmation=1)
    launches = []

    def launch(phase, cfg, m, root):
        launches.append(phase)
        stage_workers.worker(phase, cfg, m, root)

    state = stage_suite.run(c, root, launch=launch)
    assert state["status"] == "completed" and not state["failures"]
    assert read_json(root / "pilot_gate.json")["status"] == "passed"
    assert (root / "PILOT_REPORT.md").exists()
    for arm in ARMS:
        last = root / "training" / arm / "updates" / "00009" / "state.npz"
        v = load_tree(last)
        assert v["optimizer"]["step"] == 9
        assert v["update"] == 9
    assert read_json(root / "bundle_receipt.json")["verified"]
    result = stage_suite.run(
        c, root, "resume", launch=lambda *a: pytest.fail("Completed phase repeated")
    )
    assert result["status"] == "completed"
    with pytest.raises(ValueError, match="changed"):
        changed = copy.deepcopy(c)
        changed["stage_budget"]["hidden"] = 5
        stage_suite.run(changed, root, "resume", launch=launch)


def test_cli_probe_is_actual_subprocess(settings):
    c, root = settings
    stage_data.prepare(c, root)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "cognitive_ultrasound.task_budget",
            "stage-worker",
            "--output",
            str(root),
            "--stage-phase",
            "probe",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    batch = digest(read_json(root / "identity.json"))
    assert (
        committed(root / "jobs/probe", dict(batch=batch, phase="probe"))["production_gpu"] is False
    )


def test_optimizer_resume_preserves_reward_duals_and_curriculum(settings):
    import shutil

    c, root = settings
    c, m = stage_data.prepare(c, root)
    other = root.parent / "continuous"
    other.mkdir()
    for name in ["config.json", "requested_config.json", "manifest.json", "identity.json"]:
        shutil.copy2(root / name, other / name)
    a = Runtime(c, m, root, "test")
    b = Runtime(c, m, other, "test")
    stage_workers.train(a, "Q2", 1)
    stage_workers.train(a, "Q2", 2)
    stage_workers.train(b, "Q2", 2)
    x = stage_workers.checkpoint_state(a, "Q2", 2)
    y = stage_workers.checkpoint_state(b, "Q2", 2)
    assert tree_digest(x) == tree_digest(y)


def test_first_stage_cannot_anticipate_changed_current_frame(settings):
    from cognitive_ultrasound.task_budget.data import read_episode
    from cognitive_ultrasound.task_budget.episode import rollout
    from cognitive_ultrasound.task_budget.value_testing import SyntheticPerception, SyntheticTask

    c, root = settings
    c, m = stage_data.prepare(c, root)
    frames = read_episode(c, m, m["pilot_development"][0])
    changed = frames.copy()
    changed[1] = -changed[1]
    params = initialize(42, c, "T1")
    rows = []
    for data in [frames, changed]:
        _, r, _ = rollout(
            c,
            SyntheticPerception(),
            SyntheticTask(c),
            data,
            None,
            "E0",
            17,
            retain_contexts=False,
            controller=Controller(params, c, "T1", 64),
        )
        rows.append(r)
    assert rows[0][1]["k1"] == rows[1][1]["k1"]
    assert rows[0][1]["lines1"] == rows[1][1]["lines1"]


def test_cli_status_without_loading_models(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "cognitive_ultrasound.task_budget",
            "stage-status",
            "--output",
            str(tmp_path / "not_started"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["coordinator_alive"] is False
