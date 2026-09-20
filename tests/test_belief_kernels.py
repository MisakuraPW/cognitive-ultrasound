"""Same objective, frozen parameters and optimizer updates under optional graph execution."""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
if tf.keras.backend.backend() != "tensorflow":
    pytest.skip("Fresh TensorFlow process required", allow_module_level=True)

from cognitive_ultrasound.belief_filter.core import Models  # noqa: E402
from cognitive_ultrasound.belief_filter.kernels import make_kernels  # noqa: E402
from cognitive_ultrasound.belief_filter.runner import loss_for_clip  # noqa: E402
from cognitive_ultrasound.config import ROOT, load  # noqa: E402


@pytest.mark.parametrize("stage", ["codec", "prior", "filter"])
def test_graph_matches_eager_objective_and_update(stage):
    cfg = load(ROOT / "configs/belief_filter/pilot.yaml")
    cfg.update(width=4, latent_channels=2, clip_frames=2, training_policy="uniform")
    tf.keras.utils.set_random_seed(42)
    eager, graph = Models(cfg), Models(cfg)
    for name, model in eager.all().items():
        graph.all()[name].set_weights(model.get_weights())
    variables = eager.configure_stage(stage)
    other = graph.configure_stage(stage)
    eo, go = tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4)
    eo.build(variables)
    go.build(other)
    update, evaluate = make_kernels(graph, go, stage, cfg)
    clip = np.random.default_rng(3).uniform(-1, 1, (2, 112, 112, 1)).astype("float32")
    start = tf.constant(17, tf.int32)
    expected = float(loss_for_clip(eager, clip, 17, stage, cfg))
    np.testing.assert_allclose(float(evaluate(clip, start)), expected, rtol=2e-5, atol=1e-6)
    for _ in range(2):
        with tf.GradientTape() as tape:
            loss = loss_for_clip(eager, clip, 17, stage, cfg)
        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        eo.apply_gradients(zip(gradients, variables))
        actual = float(update(clip, start))
        np.testing.assert_allclose(actual, float(loss), rtol=2e-5, atol=1e-6)
    assert int(eo.iterations) == int(go.iterations) == 2
    for name, model in eager.all().items():
        for a, b in zip(model.get_weights(), graph.all()[name].get_weights()):
            np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-6)


def test_graph_never_silently_replaces_greedy_policy():
    cfg = load(ROOT / "configs/belief_filter/pilot.yaml")
    model = Models(cfg)
    with pytest.raises(ValueError, match="uniform"):
        make_kernels(model, tf.keras.optimizers.Adam(), "filter", cfg)


@pytest.mark.parametrize("interval", [2, 4])
def test_reset_interval_graph_matches_eager(interval):
    cfg = load(ROOT / "configs/belief_filter/pilot.yaml")
    cfg.update(
        width=4,
        latent_channels=2,
        clip_frames=4,
        training_policy="uniform",
        training_reset_interval=interval,
    )
    tf.keras.utils.set_random_seed(77)
    models = Models(cfg)
    variables = models.configure_stage("filter")
    optimizer = tf.keras.optimizers.Adam(1e-4)
    optimizer.build(variables)
    _, evaluate = make_kernels(models, optimizer, "filter", cfg)
    clip = np.random.default_rng(8).uniform(-1, 1, (4, 112, 112, 1)).astype("float32")
    expected = float(loss_for_clip(models, clip, 11, "filter", cfg))
    np.testing.assert_allclose(
        float(evaluate(clip, tf.constant(11, tf.int32))), expected, rtol=2e-5, atol=1e-6
    )
