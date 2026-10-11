"""Existing weights/caches only; CPU replay parity and closed-boundary scheduling."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.preparation.common import atomic_json, atomic_npz
from cognitive_ultrasound.task_budget.value_storage import commit, save_tree

sys.path.insert(0, str(ROOT / "scripts"))
import analyze_task_budget_stages_136 as analysis
import budget136_analysis_core as core
import queue_task_budget_analysis_136 as queue


def test_numpy_replay_matches_actual_jax_gru():
    import jax.numpy as jnp

    from cognitive_ultrasound.task_budget.stage_policy import logits

    rng = np.random.default_rng(5)
    h = 4
    p = dict(
        wx=rng.normal(0, 0.1, (18, 3 * h)).astype(np.float32),
        wh=rng.normal(0, 0.1, (h, 3 * h)).astype(np.float32),
        b=np.zeros(3 * h, np.float32),
        wo=rng.normal(0, 0.1, (h, 5)).astype(np.float32),
        bo=np.zeros(5, np.float32),
    )
    x = rng.normal(0, 0.1, (3, 16, 18)).astype(np.float32)
    v = np.ones((3, 16), bool)
    v[0, :12] = False
    actual = np.stack(
        [np.asarray(logits({k: jnp.asarray(a) for k, a in p.items()}, a, b)) for a, b in zip(x, v)]
    )
    computed, _ = core.forward(p, x, v)
    np.testing.assert_allclose(actual, computed, atol=2e-4, rtol=2e-4)
    changed = x.copy()
    changed[0, :12] = 100
    np.testing.assert_array_equal(core.forward(p, changed, v)[0][0], computed[0])


def test_mask_confidence_does_not_hide_unmasked_uncertainty():
    logits = np.zeros((2, 5), np.float32)
    legal = np.array([[0, 0, 0, 1, 0], [1, 1, 1, 1, 1]], bool)
    raw = core.softmax(logits)
    masked = core.softmax(logits, legal)
    assert core.entropy(masked)[0] == 0 and core.entropy(raw)[0] > 1
    assert core.js(raw, masked)[0] > 0
    with pytest.raises(ValueError):
        core.softmax(logits, np.zeros_like(legal))


def test_pca_equal_video_weight_and_no_cross_model_alignment_claim():
    h = np.array([[0, 0], [0, 0], [0, 0], [10, 0]], float)
    projected, ratio, center, axes = core.weighted_pca(h, ["A", "A", "A", "B"])
    np.testing.assert_allclose(center, [5, 0])
    assert ratio[0] == 1 and projected.shape == (4, 2)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    name = "fixture.hdf5"
    n = 64
    h = 4
    cfg = dict(data_root=str(tmp_path / "data"), stage_budget=dict(functional_fixture=True))
    target = (
        np.broadcast_to(np.linspace(-0.8, 0.7, 112)[None, None, :], (n, 112, 112))
        .copy()
        .astype(np.float32)
    )
    path = Path(cfg["data_root"]) / "val" / name
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        f["data/image"] = (target - 1) * 30
    atomic_json(root / "config.json", cfg)
    atomic_json(root / "identity.json", dict(source={}))
    atomic_json(
        root / "manifest.json",
        dict(
            cohorts=dict(development=[name]),
            files={name: dict(frames=n, relative="val/" + name, sha256=core.sha(path))},
        ),
    )
    atomic_json(root / "status.json", dict(status="stopped", pid=None, created=None))
    atomic_json(root / "scope_closure_136.json", dict(status="completed", target_updates=136))
    archive = tmp_path / "closed.tar.gz"
    archive.write_bytes(b"verified synthetic closure fixture")
    atomic_json(
        root / "scope_136_bundle_receipt.json",
        dict(verified=True, archive=str(archive), sha256=core.sha(archive)),
    )
    params = {}
    paths = {}
    for ai, arm in enumerate(core.ARMS):
        rng = np.random.default_rng(ai + 20)
        a = len(core.LEVELS[arm])
        base = dict(
            wx=rng.normal(0, 0.1, (18, 3 * h)).astype(np.float32),
            wh=rng.normal(0, 0.1, (h, 3 * h)).astype(np.float32),
            b=np.zeros(3 * h, np.float32),
            wo=rng.normal(0, 0.1, (h, a)).astype(np.float32),
            bo=np.zeros(a, np.float32),
        )
        for update in range(1, 137):
            p = {k: v + np.float32(update * 0.00001) for k, v in base.items()}
            params[arm] = p
            d = root / "training" / arm / "updates" / f"{update:05d}"
            r = dict(
                arm=arm,
                update=update,
                case=name,
                section="pilot"
                if update <= 16
                else "full_pass_1"
                if update <= 72
                else "full_pass_2",
                ef_error=5 - update * 0.001,
                mse=0.01,
                mean_lines=14,
                objective=1.0,
                reward=-1.0,
                advantage=0.1,
                gradient_norm=0.02,
                task_gradient_norm=0.01,
                seconds=1,
                dual_after=[1.0, 1.1],
            )
            atomic_json(d / "record.json", r)
            save_tree(d / "state.npz", dict(params=p))
            commit(d, dict(arm=arm, update=update), r, ["record.json", "state.npz"])
    anchors = [8, 23, 38, 53]
    for arm in ["M", *core.ARMS]:
        d = root / "trajectories" / arm
        d.mkdir(parents=True)
        masks = np.zeros_like(target, dtype=np.uint8)
        masks[:, :, :14] = 1
        images = np.where(masks, target, target + 0.02).astype(np.float32)
        rows = [
            dict(
                frame=i,
                k1=7,
                k2=7,
                lines1=list(range(7)),
                lines2=list(range(7, 14)),
                perception_calls=2,
            )
            for i in range(n)
        ]
        atomic_npz(d / "images.npz", images=images, masks=masks)
        atomic_json(d / "rows.json", rows)
        files = ["images.npz", "rows.json"]
        for at in anchors:
            previous = np.stack(
                [
                    np.repeat((images[at] - 0.01)[..., None], 3, axis=-1),
                    np.repeat((images[at] + 0.01)[..., None], 3, axis=-1),
                ]
            )
            save_tree(d / f"after_{at:05d}.npz", dict(previous=previous))
            files.append(f"after_{at:05d}.npz")
        if arm != "M":
            rng = np.random.default_rng(8)
            trace = []
            for i in range(1, n):
                x = rng.normal(0, 0.05, (16, 18)).astype(np.float32)
                v = np.arange(16) >= 16 - min(i, 16)
                legal = np.ones(len(core.LEVELS[arm]), bool)
                if i == n - 1:
                    legal[:] = False
                    legal[np.where(core.LEVELS[arm] == 7)[0][0]] = True
                p = core.softmax(core.forward(params[arm], x[None], v[None])[0], legal[None])[0]
                trace.append(
                    dict(
                        frame=i,
                        sequence=x,
                        valid=v,
                        legal=legal,
                        action=int(np.where(core.LEVELS[arm] == 7)[0][0]),
                        quota_forced=i == n - 1,
                        quota_constrained=i == n - 1,
                        max_probability=float(p.max()),
                        entropy=float(core.entropy(p[None])[0]),
                    )
                )
            save_tree(
                d / "controller.npz",
                dict(identity=dict(arm=arm, parameters=core.tree_hash(params[arm])), trace=trace),
            )
            files.append("controller.npz")
        commit(d, dict(arm=arm), dict(case=name), files)
        paths[arm] = str(d)
    for phase, update in [
        ("pilot_evaluate", 16),
        ("full_pass_1_evaluate", 72),
        ("full_pass_2_evaluate", 136),
    ]:
        cases = [
            dict(
                arm=arm,
                case=name,
                seed=42,
                cohort="development",
                update=update,
                trajectory=paths[arm],
                ef_error=4.0,
                mean_lines=14.0,
                mse=0.01,
                mse_ratio=0.02,
                p90_ratio=0.03,
                ef_vs_fixed=-0.1,
                quota_forced_fraction=1 / 63,
                reference=dict(trajectory=paths["M"]),
            )
            for arm in core.ARMS
        ]
        d = root / "jobs" / phase
        atomic_json(d / "result.json", dict(cases=cases, status="completed"))
        commit(d, dict(phase=phase), dict(cases=cases, status="completed"), ["result.json"])
    return root


def test_source_requires_closed_boundary(source, tmp_path):
    (source / "scope_closure_136.json").unlink()
    with pytest.raises(FileNotFoundError):
        core.Source(source)


def test_core_replay_and_deterministic_samples(source, tmp_path):
    s = core.Source(source)
    out = tmp_path / "analysis"
    out.mkdir()
    data, pcas = core.analyze_models(s, out)
    pictures, gaps = core.samples(s, out)
    assert not gaps and data["T1"]["replay_verified_atol_rtol"] == 2e-4
    assert data["T2"]["latest_only"]["scope"].startswith("counterfactual")
    assert list(pictures) == s.case_names[:3]
    np.testing.assert_array_equal(pictures[s.case_names[0]]["indexes"], [8, 23, 38, 53])
    assert (out / "policy_frames.csv").exists()


def test_full_cpu_report_bundle_and_resume(source, tmp_path):
    before = core.sha(source / "identity.json")
    out = tmp_path / "analysis"
    result = analysis.run(source, out)
    assert result["status"] == "completed" and result["cpu_only"]
    assert "SYNTHETIC FUNCTIONAL ONLY" in (out / "REPORT_中文.html").read_text(encoding="utf-8")
    receipt = analysis.bundle(out)
    assert receipt["verified"]
    (out / "queue_status.json").write_text("mutable scheduler state")
    (out / "analysis.console.log").write_text("mutable operational log")
    assert analysis.run(source, out) == result
    assert analysis.bundle(out) == receipt
    assert core.sha(source / "identity.json") == before
    with pytest.raises(ValueError):
        analysis.run(source, source / "overwrite")


def test_queue_gate_never_starts_during_live_training(source, monkeypatch):
    atomic_json(source / "status.json", dict(status="running", pid=123, created=1))
    monkeypatch.setattr(queue, "alive", lambda *args: True)
    assert queue.ready(source) is False


def test_queue_rejects_failed_closure(source):
    atomic_json(source / ".finish_136/status.json", dict(status="failed"))
    with pytest.raises(RuntimeError):
        queue.ready(source)


def test_queue_gate_accepts_only_verified_closed_export(source):
    atomic_json(source / ".finish_136/status.json", dict(status="completed"))
    assert queue.ready(source)
    receipt = core.read(source / "scope_136_bundle_receipt.json")
    Path(receipt["archive"]).write_bytes(b"bad")
    with pytest.raises(ValueError):
        queue.ready(source)


def test_duplicate_output_owner_rejected(tmp_path):
    lock = tmp_path / ".queue.lock"
    with queue.output_lock(lock):
        with pytest.raises(OSError):
            with queue.output_lock(lock):
                pytest.fail("Second coordinator acquired the same output lock")
    with queue.output_lock(lock):
        pass


def test_orphan_analysis_child_prevents_duplicate_launch(source, tmp_path, monkeypatch):
    out = tmp_path / "analysis"
    atomic_json(out / "queue_status.json", dict(analysis_pid=123, analysis_created=1))
    monkeypatch.setattr(queue, "alive", lambda pid, created: pid == 123 and created == 1)
    with pytest.raises(RuntimeError, match="already owns"):
        queue.queue(source, out)


def test_queue_handoff_waits_then_launches_cpu_only(source, tmp_path, monkeypatch):
    out = tmp_path / "analysis"
    gates = iter([False, False, True])
    waits = []
    monkeypatch.setattr(queue, "ready", lambda _: next(gates))
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: waits.append(seconds))

    class FakeProcess:
        pid = 123
        returncode = 0

        def wait(self, timeout):
            assert timeout == 7200
            atomic_json(out / "complete.json", dict(status="completed"))
            atomic_json(out / "bundle_receipt.json", dict(verified=True))

    class FakePsutilProcess:
        def create_time(self):
            return 1.0

    def launch(command, **kwargs):
        assert len(waits) == 2
        assert command[2].endswith("analyze_task_budget_stages_136.py")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        assert "--source" in command and "--output" in command
        return FakeProcess()

    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    monkeypatch.setattr(queue.psutil, "Process", lambda *args: FakePsutilProcess())
    queue.queue(source, out)
    assert waits == [2, 2]
    assert core.read(out / "queue_status.json")["status"] == "completed"
