"""Functional synthetic tests; never reported as EchoNet scientific/GPU evidence."""

import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cognitive_ultrasound.config import ROOT, load
from cognitive_ultrasound.task_budget.episode import gs_gradient, rollout
from cognitive_ultrasound.task_budget.experiment import balanced_schedule
from cognitive_ultrasound.task_budget.policy import (
    adam,
    adam_state,
    draw,
    initialize,
    restore,
    rl_loss,
    save,
    st_weights,
)
from cognitive_ultrasound.task_budget.protocol import (
    Observation,
    causal_window,
    clip_indices,
    greedy_order,
    mask_bank,
    task_scores,
)
from cognitive_ultrasound.task_budget.report import summarize


@pytest.fixture
def cfg():
    return load(ROOT / "configs/task_budget_ef.yaml")


class ToyPerception:
    """Differentiable toy update; dimensions match the real adapter, no pretrained claim."""

    def __init__(self):
        self.calls = []

    def infer(self, history, masks, previous, key, cold=False):
        self.calls.append((np.asarray(key).copy(), cold))
        # Keep original time channels; allow observed data to inform unobserved pixels.
        h = jnp.asarray(history)
        im = 0.6 * h + 0.15 * jnp.mean(h, axis=(0, 1), keepdims=True)
        return jnp.stack([im - 0.03, im + 0.03])


class ToyTask:
    def score(self, particles, history):
        gradients = np.full_like(particles, 0.3)
        return task_scores(particles, gradients), np.array([50.0, 52.0])

    def video(self, images, gradient=False):
        pred = float(50 + np.mean(images))
        return pred, np.ones_like(images) / images.size, [pred]


def sample_frames(n=4):
    rng = np.random.default_rng(9)
    return rng.uniform(-0.8, 0.8, (n, 112, 112)).astype(np.float32)


def test_unique_exclusion_and_budget_bank():
    scores = np.arange(112, dtype=float)
    first = greedy_order(scores, 14)
    second = greedy_order(scores, 14, first)
    assert not set(first) & set(second)
    bank = mask_bank(first, [0, 4, 7, 14])
    assert np.array_equal(bank.sum(1), [0, 4, 7, 14])
    assert np.all(bank[1] <= bank[2])
    zeros = greedy_order(np.zeros(112), 14)
    assert len(set(zeros)) == 14 and np.ptp(zeros) > 100


def test_oracle_no_hidden_access_and_no_repeat():
    frame = np.zeros((112, 112), np.float32)
    oracle = Observation(frame)
    values, mask = oracle.acquire([4, 7])
    assert mask.sum() == 224 and values.sum() == 0
    with pytest.raises(ValueError):
        oracle.acquire([4])


def test_causal_window_and_video_indices():
    history = [np.full((2, 2), i) for i in range(10)]
    clip = causal_window(history, np.full((2, 2), 10), frames=4, period=2)
    assert clip[:, 0, 0].tolist() == [4, 6, 8, 10]
    ids = clip_indices(100)
    assert ids.max() == 99 and (np.diff(ids, axis=1) == 2).all()
    assert clip_indices(3)[0, -1] == 2


def test_tbig_code_score_not_mean_squared_gradient():
    p = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
    g = np.stack([np.ones((2, 2)), -np.ones((2, 2))])
    assert task_scores(p, g).sum() == 0  # documented official code convention
    with pytest.raises(FloatingPointError):
        task_scores(p, g * np.nan)


def test_st_binary_forward_nonzero_mask_gradient(cfg):
    p = initialize(42, cfg)["first"]
    state = np.ones(12, np.float32)
    bank = mask_bank(np.arange(14), cfg["budgets"]["first"])

    def f(x):
        return st_weights(x, state, np.ones(4, bool), np.zeros(4), 2, 0.7) @ bank

    out = f(p)
    assert np.array_equal(np.asarray(out), bank[2])
    grad = jax.grad(lambda x: jnp.sum(f(x)))(p)
    assert np.linalg.norm(np.asarray(grad["b2"])) > 0


def test_two_stage_history_common_noise_and_zero_skip(cfg):
    p = initialize(42, cfg)
    a = ToyPerception()
    _, rows, ctx = rollout(cfg, a, ToyTask(), sample_frames(), p, "E0", 42, fixed=(4, 0))
    assert rows[0]["k1"] == 10 and rows[0]["k2"] == 4  # shared cold start
    assert rows[1]["perception_calls"] == 1 and len(a.calls) == 5
    # Current frame stage2 overwrites current channel, it does not shift time twice.
    assert np.all(ctx[1]["history"][..., 0] == 0)
    assert np.any(ctx[1]["history"][..., -1] != 0)
    b = ToyPerception()
    _, r2, _ = rollout(cfg, b, ToyTask(), sample_frames(), p, "E0", 42, fixed=(10, 4))
    # Match stage1 diffusion keys despite different second-call counts.
    np.testing.assert_array_equal(a.calls[2][0], b.calls[2][0])
    for r in r2:
        assert len(r["lines1"]) == r["k1"] and len(r["lines2"]) == r["k2"]
        assert not set(r["lines1"]) & set(r["lines2"])


@pytest.mark.parametrize("fixed", [(10, 4), [10, 4], [(10, 4), (4, 0), (7, 2), (14, 4)]])
def test_fixed_json_pair_and_frame_schedule(cfg, fixed):
    _, rows, _ = rollout(
        cfg, ToyPerception(), ToyTask(), sample_frames(), initialize(42, cfg), "E0", 42, fixed=fixed
    )
    expected = [(10, 4)] * 4 if len(np.asarray(fixed).shape) == 1 else fixed
    assert [(r["k1"], r["k2"]) for r in rows] == expected


def test_exact_forward_vjp_retains_primal_and_gradients():
    from cognitive_ultrasound.task_budget.perception import exact_forward_vjp

    kernel = jax.jit(lambda h, m, p, key: jnp.sin(h * m) + p**2)
    wrapped = exact_forward_vjp(kernel)
    inputs = (
        jnp.arange(4, dtype=float),
        jnp.ones(4) * 0.3,
        jnp.ones(4) * 0.2,
        jax.random.PRNGKey(42),
    )
    value, grads = jax.value_and_grad(
        lambda h, m, p: jnp.sum(wrapped(h, m, p, inputs[3])), argnums=(0, 1, 2)
    )(*inputs[:3])
    expected, eg = jax.value_and_grad(
        lambda h, m, p: jnp.sum(kernel(h, m, p, inputs[3])), argnums=(0, 1, 2)
    )(*inputs[:3])
    np.testing.assert_array_equal(np.asarray(wrapped(*inputs)), np.asarray(kernel(*inputs)))
    assert value == expected
    for a, b in zip(grads, eg):
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-4)


def test_first_decision_cannot_read_current_unobserved_pixels(cfg):
    p = initialize(42, cfg)
    frames = sample_frames()
    _, r, c = rollout(cfg, ToyPerception(), ToyTask(), frames, p, "E0", 42)
    altered = frames.copy()
    remaining = set(range(112)) - set(r[1]["lines1"])
    altered[1, :, sorted(remaining)] = 0.99
    _, rr, cc = rollout(cfg, ToyPerception(), ToyTask(), altered, p, "E0", 42)
    np.testing.assert_array_equal(c[1]["state0"], cc[1]["state0"])
    np.testing.assert_array_equal(c[1]["state1"], cc[1]["state1"])
    assert rr[1]["lines2"] == r[1]["lines2"]


@pytest.mark.parametrize("second", [0, 4])
def test_gs_task_gradient_reaches_both_heads_and_replays_hard_frame(cfg, second):
    p = initialize(42, cfg)
    perception = ToyPerception()
    images, _, contexts = rollout(
        cfg, perception, ToyTask(), sample_frames(3), p, "E0", 42, fixed=(10, second)
    )
    _, grad, _ = ToyTask().video(images, True)
    _, gate = gs_gradient(p, contexts, grad, perception, cfg, 1.0, 0.0)
    assert all(x > 0 for x in gate["task_gradient_norms"].values())
    assert gate["hard_replay_max_abs"] < 2e-4


@pytest.mark.parametrize("second", [0, 4])
def test_compiled_gs_changes_inputs_without_stale_history(cfg, second):
    class PureToy:
        def infer(self, history, masks, previous, key, cold=False):
            h = jnp.asarray(history)
            image = .6*h + .15*jnp.mean(h,axis=(0,1),keepdims=True)
            return jnp.stack([image-.03,image+.03])

    perception = PureToy()
    params = initialize(42,cfg)
    fast = copy.deepcopy(cfg)
    fast["runtime"]["gs_execution"] = "jit"
    for shift,temperature in ((0.,1.),(.1,.4)):
        images,_,contexts = rollout(cfg,perception,ToyTask(),sample_frames(3)+shift,
                                    params,"E0",42,fixed=(10,second))
        _,adjoint,_ = ToyTask().video(images,True)
        a,_ = gs_gradient(params,contexts,adjoint,perception,cfg,temperature,2.)
        b,gate = gs_gradient(params,contexts,adjoint,perception,fast,temperature,2.)
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_allclose(x,y,atol=2e-4,rtol=2e-4)
        assert gate["hard_replay_max_abs"] < 2e-4


def test_pilot_is_same_scientific_plan_and_confirmation_stays_last(cfg):
    from cognitive_ultrasound.task_budget.suite import jobs
    from cognitive_ultrasound.task_budget.staging import execution_schedule

    staged=copy.deepcopy(cfg)
    staged["execution"]=dict(pilot_seed=42,pilot_lambda=2.,pilot_cases=4,pilot_fixed=[10,4])
    canonical=jobs(staged)
    assert canonical==jobs(cfg) and len(canonical)==91
    manifest=dict(cohorts=dict(development=[f"case{i}" for i in range(8)]))
    schedule=execution_schedule(canonical,staged,manifest)
    boundary=next(i for i,(s,_) in enumerate(schedule) if s is None)
    before=schedule[:boundary]
    assert all(s.get("cohort")!="confirmation" for s,_ in before)
    assert {s["method"] for s,_ in before if s.get("kind")=="evaluate"}=={"E0","E1","E2"}
    assert all(s["_case_names"]==manifest["cohorts"]["development"][:4] for s,_ in before if s.get("kind")=="evaluate")
    assert [s for s,_ in schedule[boundary+1:]]==canonical


def test_pilot_cannot_add_new_parameter_points(cfg):
    from cognitive_ultrasound.task_budget.staging import pilot_ids

    cfg=copy.deepcopy(cfg)
    cfg["execution"]=dict(pilot_seed=99,pilot_lambda=2.,pilot_cases=4,pilot_fixed=[10,4])
    with pytest.raises(ValueError):pilot_ids(cfg)


def test_rl_mask_illegal_actions_and_resume_adam(cfg, tmp_path):
    p = initialize(42, cfg)
    p["second"]["b2"] = jnp.array([0, 0, 0, 0, 1000.0])
    rng = np.random.default_rng(42)
    legal = np.array([True, True, True, True, False])
    assert all(draw(p["second"], np.zeros(12), legal, rng, True)[0] != 4 for _ in range(20))
    p = initialize(42, cfg)
    _, _, contexts = rollout(cfg, ToyPerception(), ToyTask(), sample_frames(), p, "E2", 42, True)
    grads = jax.grad(rl_loss)(p, contexts, 1.0)
    assert all(np.linalg.norm(np.asarray(grads[h]["b2"])) > 0 for h in ("first", "second"))
    opt = adam_state(p)
    p, opt, _ = adam(p, grads, opt, cfg)
    save(tmp_path / "resume.npz", p, opt, -2.0)
    resumed, ropt, baseline = restore(tmp_path / "resume.npz")
    left, lopt, _ = adam(p, grads, opt, cfg)
    right, rr, _ = adam(resumed, grads, ropt, cfg)
    assert baseline == -2 and rr["step"] == lopt["step"] == 2
    for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(x, y)


def test_resource_match_uses_total_not_outcome(cfg):
    for frames in (56, 64, 137):
        for mean in (7.1, 10.8, 19.2):
            total = round(mean * frames)
            schedule = balanced_schedule(total, frames, cfg["budgets"]["fixed_sweep"], [10, 4])
            assert abs(sum(map(sum, schedule)) - total) / frames < 0.1
            assert schedule[0] == (10, 4)


def test_statistics_equal_video_then_seed():
    records = []
    for case, errors in [("a", [1.0, 3.0]), ("b", [10.0, 10.0])]:
        for e in errors:
            records.append(
                dict(
                    case=case,
                    absolute_error=e,
                    mean_lines=14,
                    seconds=2,
                    mean_psnr=20,
                    mean_ssim=0.8,
                    mean_mae=0.1,
                    prediction_preservation_error=1,
                    perception_calls=20,
                    full_input_absolute_error=2,
                )
            )
    summary = summarize(records)
    assert summary["absolute_error"] == 6
    assert summary["rmse"] == pytest.approx(np.sqrt((1 + 9 + 100 + 100) / 4))


def test_manifest_intersections_and_immutable_config(cfg, tmp_path):
    import csv

    import h5py
    import yaml

    from cognitive_ultrasound.task_budget.data import lock_manifest

    cfg = copy.deepcopy(cfg)
    cfg.update(
        file_list=str(tmp_path / "FileList.csv"),
        split_manifest=str(tmp_path / "splits.yaml"),
        data_root=str(tmp_path / "data"),
        cohorts=dict(train=1, development=1, confirmation=1),
    )
    entries = [
        ("0X1", "TRAIN", "train"),
        ("0X2", "VAL", "val"),
        ("0X3", "TEST", "test"),
        ("0X4", "TEST", "train"),
        ("0X5", "TRAIN", "val"),
    ]
    with open(cfg["file_list"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["FileName", "EF", "Split"])
        w.writeheader()
        w.writerows(dict(FileName=n, EF=55, Split=s) for n, s, _ in entries)
    splits = {s: [n + ".hdf5" for n, _, ss in entries if s == ss] for s in ("train", "val", "test")}
    Path(cfg["split_manifest"]).write_text(yaml.safe_dump(splits))
    for n, _, s in entries:
        file = tmp_path / "data" / s / (n + ".hdf5")
        file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(file, "w") as h:
            h["data/image"] = np.full((64, 112, 112), -30, np.float32)
    manifest = lock_manifest(cfg, tmp_path)
    assert manifest["cohorts"] == dict(
        train=["0X1.hdf5"], development=["0X2.hdf5"], confirmation=["0X3.hdf5"]
    )
    assert lock_manifest(cfg, tmp_path) == manifest
    Path(cfg["file_list"]).write_text("modified")
    with pytest.raises(ValueError):
        lock_manifest(cfg, tmp_path)


def test_frozen_task_input_gradient_and_scan_orientation(cfg, tmp_path):
    import torch

    from cognitive_ultrasound.task_budget.ef_worker import EFModel

    cfg = copy.deepcopy(cfg)
    cfg.update(task_device="cpu", ef_stats=str(tmp_path / "stats.json"))
    Path(cfg["ef_stats"]).write_text(
        json.dumps(dict(mean=[100] * 3, std=[50] * 3, units="uint8_0_255_RGB"))
    )
    # Official Cartesian output geometry[z112,x159], central crop112.
    rho = np.broadcast_to(np.arange(112)[None], (159, 112))
    theta = np.broadcast_to(np.arange(159)[:, None] - 23, (159, 112))
    coordinates = np.stack([rho, theta]).astype(np.float32)

    class SmallEF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            return self.weight * x.mean((1, 2, 3, 4))[:, None] + 55

    model = SmallEF()
    task = EFModel(cfg, coordinates, model=model)
    before = model.weight.detach().clone()
    images = sample_frames(3)[None]
    result = task.evaluate(images, True)
    assert result["gradients"].shape == images.shape and np.linalg.norm(result["gradients"]) > 0
    assert model.weight.requires_grad is False and torch.equal(before, model.weight)
    # Cropping/axis swap recovers the identity coordinate map exactly.
    expected = (torch.tensor(images[:, None]).expand(-1, 3, -1, -1, -1) + 1) * 127.5
    actual = task.preprocess(torch.tensor(images), "polar")
    torch.testing.assert_close(actual, (expected - 100) / 50)


@pytest.mark.parametrize("method", ["E1", "E2"])
def test_training_worker_resumes_same_update_stream(cfg, tmp_path, monkeypatch, method):
    import cognitive_ultrasound.task_budget.experiment as experiment
    from cognitive_ultrasound.preparation.common import read_json

    cfg = copy.deepcopy(cfg)
    cfg["training"].update(updates=2, clip_frames=64)
    cfg["task"].update(frames=2, period=1)

    class Service(ToyTask):
        calls = 0
        seconds = 0

        def close(self):
            pass

    monkeypatch.setattr(experiment, "setup", lambda cfg, out: (ToyPerception(), Service()))
    monkeypatch.setattr(experiment, "read_episode", lambda *a, **k: sample_frames())
    manifest = dict(cohorts=dict(train=["case"]), files=dict(case=dict(frames=4, ef=55)))
    spec = dict(method=method, seed=42, **{"lambda": 2.0})
    left, right = tmp_path / "continuous", tmp_path / "resumed"
    left.mkdir()
    right.mkdir()
    monkeypatch.setattr(experiment, "emit", lambda *a, **k: None)
    experiment.train(spec, cfg, manifest, tmp_path, left)

    def stop_after_first(*a, **kw):
        if kw["update"] == 1:
            (tmp_path / "STOP").write_text("stop")

    monkeypatch.setattr(experiment, "emit", stop_after_first)
    with pytest.raises(InterruptedError):
        experiment.train(spec, cfg, manifest, tmp_path, right)
    assert (right / "updates/00001.npz").exists() and not (right / "result.json").exists()
    (tmp_path / "STOP").unlink()
    monkeypatch.setattr(experiment, "emit", lambda *a, **k: None)
    experiment.train(spec, cfg, manifest, tmp_path, right)
    with np.load(left / "policy.npz") as a, np.load(right / "policy.npz") as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
    left_record = read_json(left / "updates/00002.json")
    right_record = read_json(right / "updates/00002.json")
    assert left_record["frames"] == right_record["frames"] == 4
    assert (
        left_record["case"] == right_record["case"]
        and left_record["action_seed"] == right_record["action_seed"]
    )
    assert (
        left_record["objective"] == right_record["objective"]
        and left_record["gradient_norm"] == right_record["gradient_norm"]
    )


def test_native_ef_checkpoint_service_and_cleanup(cfg, tmp_path):
    import sys

    import torch
    from torchvision.models.video import r2plus1d_18

    from cognitive_ultrasound.task_budget.task import EFService

    cfg = copy.deepcopy(cfg)
    cfg.update(
        task_device="cpu",
        torch_python=sys.executable,
        ef_weights=str(tmp_path / "functional_random_r2.pt"),
        ef_stats=str(tmp_path / "stats.json"),
    )
    network = r2plus1d_18(weights=None)
    network.fc = torch.nn.Linear(network.fc.in_features, 1)
    torch.save(
        dict(
            state_dict={"module." + k: v for k, v in network.state_dict().items()},
            frames=32,
            period=2,
            loss=np.float64(1.0),
        ),
        cfg["ef_weights"],
    )
    Path(cfg["ef_stats"]).write_text(
        json.dumps(dict(mean=[100] * 3, std=[50] * 3, units="uint8_0_255_RGB"))
    )
    rho = np.broadcast_to(np.arange(112)[None], (159, 112))
    theta = np.broadcast_to(np.arange(159)[:, None] - 23, (159, 112))
    service = EFService(cfg, tmp_path, np.stack([rho, theta]).astype(np.float32))
    try:
        result = service.request(sample_frames(32)[None], True)
        assert result["predictions"].shape == (1,)
        assert result["gradients"].shape == (1, 32, 112, 112)
        assert np.linalg.norm(result["gradients"]) > 0
    finally:
        service.close()
    assert service.process.poll() is not None


def test_failed_gs_gate_is_blocked_without_cost_only_or_rl_fallback(cfg, tmp_path, monkeypatch):
    import os

    from cognitive_ultrasound.preparation.common import atomic_json, read_json
    from cognitive_ultrasound.task_budget import report, suite

    actual_plan = suite.jobs(cfg)
    first_confirmation = next(
        i for i, s in enumerate(actual_plan) if s.get("cohort") == "confirmation"
    )
    assert all(i < first_confirmation for i, s in enumerate(actual_plan) if s["kind"] == "train")

    plan = [
        dict(id="probe", kind="probe"),
        dict(id="E1_train", kind="train", method="E1"),
        dict(id="E2_train", kind="train", method="E2"),
    ]

    def prepare(cfg, root):
        atomic_json(root / "config.json", cfg)
        return dict(identity="functional", cohorts={})

    monkeypatch.setattr(suite, "prepare", prepare)
    monkeypatch.setattr(suite, "identity", lambda *a: dict(functional_fixture=True))
    monkeypatch.setattr(suite, "jobs", lambda *a: plan)

    class CompletedProcess:
        pid = os.getpid()

        def __init__(self, args, **kw):
            atomic_json(
                Path(args[-1]).parent / "result.json", dict(status="completed", gs_status="failed")
            )

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(suite.subprocess, "Popen", CompletedProcess)
    monkeypatch.setattr(report, "report", lambda *a: None)
    monkeypatch.setattr(report, "bundle", lambda *a: None)
    suite.run(cfg, tmp_path)
    assert read_json(tmp_path / "jobs/E1_train/result.json")["status"] == "blocked"
    assert read_json(tmp_path / "jobs/E2_train/result.json")["status"] == "completed"
    assert read_json(tmp_path / "status.json")["status"] == "finished_with_gaps"


def test_cold_task_saliency_accumulates_repeated_current_slots(cfg):
    from cognitive_ultrasound.task_budget.task import EFService

    service = object.__new__(EFService)
    service.cfg = cfg
    service.request = lambda clips, gradient: dict(
        predictions=np.array([50.0, 51.0]), gradients=np.ones_like(clips)
    )
    particles = np.stack([np.zeros((112, 112)), np.ones((112, 112))]).astype(np.float32)
    cold, _ = service.score(particles, [])
    warm, _ = service.score(particles, [np.zeros((112, 112))])
    np.testing.assert_allclose(cold, warm * 32**2)
