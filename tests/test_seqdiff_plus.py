"""Causal warm initialization and a real predictor update without GPU timing claims."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from cognitive_ultrasound.compute_lab.engines import ViViTEngine


def test_only_newest_warm_channel_changes_after_four_past_frames():
    jnp = pytest.importorskip("jax.numpy")

    @dataclass
    class State:
        posterior_samples: object
        belief_distribution: object

    engine = object.__new__(ViViTEngine)
    seen = []

    def recover(obs, state):
        seen.append(state.posterior_samples)
        return obs, State(state.posterior_samples, obs)

    engine.adapter = SimpleNamespace(agent=SimpleNamespace(recover=recover))
    engine.history = ()
    engine.predict = lambda history: history[:, 0] + history[:, -1]
    samples = jnp.broadcast_to(jnp.arange(3, dtype="float32"), (2, 112, 112, 3))
    state = State(samples, None)
    for index in range(5):
        obs = jnp.full((2, 112, 112, 1), index, dtype="float32")
        engine.recover(obs, state)
    assert all(np.array_equal(old, samples) for old in seen[:4])
    assert np.array_equal(seen[4][..., :2], samples[..., :2])
    assert np.all(np.asarray(seen[4][..., -1]) == 3)
    assert len(engine.history) == 4


def test_vivit_trains_and_loads(tmp_path):
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "jax":
        pytest.skip("Run with KERAS_BACKEND=jax")
    from cognitive_ultrasound.seqdiff_plus import load_predictor, predictor

    model = predictor(width=16)
    history = np.zeros((1, 4, 112, 112, 1), dtype=np.float32)
    history[:, -1] = 0.25
    target = np.zeros((1, 112, 112, 1), dtype=np.float32)
    assert np.array_equal(np.asarray(model(history)), history[:, -1])
    model.compile(optimizer="adam", loss="mae")
    assert np.isfinite(model.train_on_batch(history, target))
    after = np.asarray(model(history))
    assert not np.array_equal(after, history[:, -1])
    path = tmp_path / "test.keras"
    model.save(path)
    np.testing.assert_allclose(load_predictor(path)(history), after, rtol=1e-6, atol=1e-6)
