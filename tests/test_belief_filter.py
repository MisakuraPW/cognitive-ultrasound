"""CPU functional checks, not scientific accuracy claims or senior-weight tests."""

import copy
import json
import os

import h5py
import numpy as np
import pytest
import yaml

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
tf = pytest.importorskip("tensorflow")
if tf.keras.backend.backend() != "tensorflow":
    pytest.skip("Belief tests require a fresh TensorFlow process", allow_module_level=True)

from cognitive_ultrasound.belief_filter.core import (  # noqa: E402
    ArrayOracle,
    Models,
    initial_memory,
    select_lines,
    step,
)
from cognitive_ultrasound.belief_filter.runner import (  # noqa: E402
    evaluate,
    load_checkpoint,
    loss_for_clip,
    train,
    validate,
)
from cognitive_ultrasound.config import ROOT, load  # noqa: E402


@pytest.fixture
def cfg():
    config = copy.deepcopy(load(ROOT / "configs/belief_filter/pilot.yaml"))
    config.update(
        width=4,
        latent_channels=2,
        clip_frames=2,
        train_cases=1,
        val_cases=1,
        eval_cases=1,
        eval_frames=2,
        checkpoint_every=1,
        validation_every=1,
    )
    config["steps"] = {"codec": 2, "prior": 2, "filter": 2}
    return validate(config)


@pytest.fixture
def models(cfg):
    tf.keras.utils.set_random_seed(42)
    return Models(cfg)


def test_unique_budget_and_ties(cfg):
    lines = select_lines(np.zeros(112), np.zeros(112, bool), 112, cfg)
    assert lines[0] == 55
    assert set(lines) == set(range(112))
    with pytest.raises(ValueError):
        select_lines(np.zeros(112), np.ones(112, bool), 1, cfg)


def test_unobserved_truth_cannot_change_actions_or_reconstruction(models, cfg):
    target = np.random.default_rng(2).uniform(-1, 1, (112, 112, 1)).astype("float32")
    memory = initial_memory(models)
    next_memory, first = step(models, ArrayOracle(target), memory, 0, cfg)
    changed = np.where(first["mask"][0] > 0, target, -target)
    _, second = step(models, ArrayOracle(changed), memory, 0, cfg)
    assert [g["lines"] for g in first["groups"]] == [g["lines"] for g in second["groups"]]
    np.testing.assert_array_equal(first["reconstruction"], second["reconstruction"])
    mask = first["mask"][0].astype(bool)
    np.testing.assert_array_equal(np.asarray(first["reconstruction"])[0][mask], target[mask])
    assert first["mask"][0, 0].sum() == 14
    # A new frame must expose the new observation even if the same line was seen before.
    _, next_frame = step(models, ArrayOracle(-target), next_memory, 1, cfg)
    mask = next_frame["mask"][0].astype(bool)
    assert next_frame["mask"][0, 0].sum() == 14
    np.testing.assert_array_equal(next_frame["observation"][0][mask], (-target)[mask])
    assert np.all(next_frame["observation"][0][~mask] == 0)


def test_filter_gradients_and_frozen_components(models, cfg):
    variables = models.configure_stage("filter")
    before = {k: [v.copy() for v in m.get_weights()] for k, m in models.all().items()}
    clip = np.random.default_rng(3).uniform(-1, 1, (2, 112, 112, 1)).astype("float32")
    with tf.GradientTape() as tape:
        loss = loss_for_clip(models, clip, 0, "filter", cfg)
    gradients = tape.gradient(loss, variables)
    assert np.isfinite(loss)
    assert all(g is not None and np.isfinite(g).all() for g in gradients)
    tf.keras.optimizers.Adam(1e-3).apply_gradients(zip(gradients, variables))
    for name in ("encoder", "decoder", "prior"):
        for old, new in zip(before[name], models.all()[name].get_weights()):
            np.testing.assert_array_equal(old, new)
    assert any(
        not np.array_equal(a, b) for a, b in zip(before["filter"], models.filter.get_weights())
    )


def test_three_stage_resume_and_evaluation(tmp_path, cfg):
    split = {}
    for index, name in enumerate(("train", "val", "test")):
        directory = tmp_path / "data" / name
        directory.mkdir(parents=True)
        filename = f"synthetic_{name}.hdf5"
        split[name] = [filename]
        with h5py.File(directory / filename, "w") as h5:
            h5.create_dataset(
                "data/image",
                data=np.random.default_rng(index).uniform(-60, 0, (3, 112, 112)).astype("float32"),
            )
    manifest = tmp_path / "split.yaml"
    manifest.write_text(yaml.safe_dump(split), encoding="utf-8")
    cfg.update(data_root=str(tmp_path / "data"), split_manifest=str(manifest))
    codec, prior, filtered = [tmp_path / k for k in ("codec", "prior", "filter")]
    with pytest.raises(ValueError, match="trained parent"):
        train(cfg, prior, "prior", cpu=True)
    train(cfg, codec, "codec", cpu=True, stop_after=1)
    assert json.loads((codec / "status.json").read_text())["status"] == "paused"
    with pytest.raises(ValueError, match="completed codec"):
        train(cfg, prior, "prior", codec / "checkpoint.npz", cpu=True)
    train(cfg, codec, "codec", resume=True, cpu=True)
    # Resume must restore optimizer and weights, not silently restart.
    continuous = tmp_path / "continuous"
    train(cfg, continuous, "codec", cpu=True)
    with (
        np.load(codec / "checkpoint.npz") as resumed,
        np.load(continuous / "checkpoint.npz") as full,
    ):
        assert int(resumed["optimizer_0"]) == 2
        for key in full.files:
            if key != "metadata":
                np.testing.assert_allclose(resumed[key], full[key], rtol=1e-5, atol=1e-6)
    train(cfg, codec, "codec", resume=True, cpu=True)
    assert json.loads((codec / "status.json").read_text())["status"] == "completed"
    train(cfg, prior, "prior", codec / "checkpoint.npz", cpu=True)
    train(cfg, filtered, "filter", prior / "checkpoint.npz", cpu=True)
    evaluation = tmp_path / "eval"
    evaluate(cfg, filtered / "checkpoint.npz", evaluation, cpu=True)
    report = json.loads((evaluation / "manifest.json").read_text())
    assert report["status"] == "completed" and not report["full_reproduction"]
    assert report["cases"] == ["synthetic_val.hdf5"]
    frame = evaluation / "synthetic_val" / "frame_0000"
    with np.load(frame / "state.npz") as state:
        assert state["mask"][0].sum() == 14
        assert state["group_reconstructions"].shape == (2, 112, 112, 1)
    assert (frame / "uncertainty.png").is_file()
    # Evaluation budget may change without changing the learned architecture.
    other = copy.deepcopy(cfg)
    other["groups"] = [6, 4]
    load_checkpoint(filtered / "checkpoint.npz", Models(other))
