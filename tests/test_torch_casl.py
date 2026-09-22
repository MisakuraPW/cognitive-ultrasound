"""Real-weight port equivalence and scientific reporting guards, CPU only."""

import numpy as np
import pytest

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.preparation.common import atomic_json, read_json


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from cognitive_ultrasound.torch_casl.reference import export_network, load_model, probes

    checkpoint = ROOT / "checkpoints/official"
    if not (checkpoint / "model.weights.h5").exists():
        pytest.skip("Official EMA checkpoint not available")
    model = load_model(checkpoint, cpu=True)
    directory = tmp_path_factory.mktemp("torch-export")
    export_network(model, checkpoint, directory)
    probes(model, directory)
    return directory, model


@pytest.fixture(scope="module")
def native(exported):
    import torch

    from cognitive_ultrasound.torch_casl.native import FrozenGraph, NativeCASL

    torch.set_num_threads(2)
    return NativeCASL(FrozenGraph(exported[0]))


def test_official_ema_dps_entropy_and_actions(exported, native):
    from cognitive_ultrasound.torch_casl.worker import validate_probes

    result = validate_probes(native, exported[0], "cpu")
    assert result["passed"], result
    assert not any(p.requires_grad for p in native.parameters())


def test_complete_solver_crosses_progress_interval_without_host_conversion(exported, native):
    import jax
    import jax.numpy as jnp

    from cognitive_ultrasound.torch_casl.native import nchw
    from cognitive_ultrasound.torch_casl.reference import matched_frame

    rng = np.random.default_rng(72)
    previous = rng.uniform(-1, 1, (2, 112, 112, 3)).astype("float32")
    mask = np.zeros((1, 112, 112, 3), "float32")
    mask[:, :, ::8] = 1
    measurement = rng.uniform(-1, 1, (2, 112, 112, 3)).astype("float32") * mask
    measurement[:, 0, 0, -1] = 0  # retain official nonzero hard-projection convention
    noise = rng.normal(size=previous.shape).astype("float32")
    args = (measurement, mask, previous, noise)
    expected = jax.jit(matched_frame(exported[1], 14), static_argnames=("steps",))(
        # Eleven steps include step 490, where upstream's default progress
        # recorder attempts NumPy conversion of a JAX tracer. Three missed it.
        *map(jnp.asarray, args), steps=11
    )
    actual = native.frame(*(nchw(a) for a in args), steps=11)
    np.testing.assert_allclose(
        actual[0].permute(0, 2, 3, 1), np.asarray(expected[0]), rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(actual[1], np.asarray(expected[1]), rtol=2e-4, atol=2e-4)
    np.testing.assert_array_equal(actual[2], np.asarray(expected[2]))
    np.testing.assert_allclose(actual[3], np.asarray(expected[3]), atol=2e-5)


def test_compiled_input_gradient_and_two_changed_inputs(exported, native):
    import torch

    from cognitive_ultrasound.torch_casl.native import nchw

    with np.load(exported[0] / "probes.npz", allow_pickle=False) as z:
        args = tuple(nchw(z[k]) for k in ("x", "measurement", "mask", "noise", "signal"))
    compiled = torch.compile(native.dps_step, backend="aot_eager", fullgraph=True)
    for factor in (1.0, 0.7):
        changed = (args[0] * factor, *args[1:], args[3] * 0.9, args[4] * 1.01)
        expected = native.dps_step(*changed)
        actual = compiled(*changed)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_zero_entropy_ties_preserve_upstream_behaviour(native):
    import torch

    selected, entropy = native.select(torch.zeros(2, 3, 112, 112))
    assert selected.sum() == 1  # upstream can repeat line zero; don't silently substitute top-k
    assert selected[0] and torch.count_nonzero(entropy) == 0


def test_no_cuda_graph_cpu_fallback(native):
    import torch

    from cognitive_ultrasound.torch_casl.native import CapturedFrame

    x = torch.zeros(2, 3, 112, 112)
    with pytest.raises(RuntimeError, match="CUDA"):
        CapturedFrame(native, (x, x, x, x), 25)


def test_incomplete_or_wrong_port_cannot_claim_32fps(tmp_path):
    from cognitive_ultrasound.torch_casl.__main__ import report

    atomic_json(tmp_path / "status.json", {"status": "finished_with_gaps"})
    rr = {"steps": 50, "cold": False, "matched_seconds": [1.0]}
    atomic_json(tmp_path / "reference.json", {"completed": True, "rows": [rr]})
    row = dict(steps=50, cold=False, seconds=[0.001], parity={"passed": False}, selected_equal=True)
    atomic_json(tmp_path / "graph.json", {"completed": True, "rows": [row]})
    atomic_json(
        tmp_path / "graph.trajectory.json",
        {
            "completed": True,
            "rows": [
                dict(
                    steps=50,
                    cold=False,
                    host_wall_s=0.001,
                    mae=0.1,
                    reference_mae=0.1,
                    selected_equal=True,
                )
            ],
        },
    )
    report(tmp_path)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "达到（仅本批）" not in text
    assert read_json(tmp_path / "comparison.json")[0]["parity"] is False


def test_bounded_config_and_plan():
    from cognitive_ultrasound.torch_casl.__main__ import configuration

    cfg = configuration(ROOT / "configs/torch_casl.yaml")
    assert cfg["max_minutes"] == 90 and cfg["steps"] == [50, 25]


def test_explicit_noise_matches_official_posterior_rng(exported):
    import jax
    import jax.numpy as jnp
    import keras
    from zea.func import split_seed

    from cognitive_ultrasound.torch_casl.reference import matched_frame

    model = exported[1]
    rng = np.random.default_rng(7)
    previous = rng.uniform(-1, 1, (2, 112, 112, 3)).astype("float32")
    mask = np.zeros((1, 112, 112, 3), "float32")
    mask[:, :, ::8] = 1
    measurement = rng.uniform(-1, 1, previous.shape).astype("float32") * mask
    seeds = split_seed(split_seed(jax.random.PRNGKey(13), 3)[0], 2)
    noise = jnp.stack(
        [keras.random.normal((1, 112, 112, 3), seed=split_seed(k, 2)[0])[0] for k in seeds]
    )

    def sample(y, p, k):
        return model.posterior_sample(
            y[None],
            n_steps=500,
            initial_step=497,
            initial_samples=p[None, None],
            mask=jnp.asarray(mask),
            seed=k,
            omega=10.0,
            track_progress_type=None,
        )[0, 0]

    expected = jax.jit(jax.vmap(sample))(
        jnp.asarray(measurement), jnp.asarray(previous), jnp.stack(seeds)
    )
    actual = jax.jit(matched_frame(model, 14), static_argnames=("steps",))(
        jnp.asarray(measurement), jnp.asarray(mask), jnp.asarray(previous), noise, steps=3
    )[0]
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=2e-4, atol=2e-4)


def test_cuda_capture_changed_input_and_owned_output(exported, native):
    import torch

    from cognitive_ultrasound.torch_casl.native import CapturedFrame, nchw

    if not torch.cuda.is_available():
        pytest.skip("CUDA Graph replay is verified on the server")
    native = native.to(device="cuda", memory_format=torch.channels_last)
    with np.load(exported[0] / "probes.npz", allow_pickle=False) as z:
        x, y, m = [nchw(z[k], "cuda") for k in ("x", "measurement", "mask")]
    args = (y, m, x, x * 0.5)
    graph = CapturedFrame(native, args, steps=3)
    first = graph(*args)
    saved = first[0].clone()
    changed = (y * 0.8, m, x, x * 0.4)
    second = graph(*changed)
    expected = native.frame(*changed, steps=3)
    for a, b in zip(second, expected):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(first[0], saved, rtol=0, atol=0)
    native.to("cpu")


def test_resume_refuses_live_worker_and_charges_interrupted_budget():
    import os

    import psutil

    from cognitive_ultrasound.torch_casl.__main__ import recover_interrupted

    state = dict(
        elapsed_s=10.0,
        jobs={
            "reference": dict(
                status="running",
                pid=os.getpid(),
                created=psutil.Process().create_time(),
                running_s=15.0,
            )
        },
    )
    with pytest.raises(RuntimeError, match="still alive"):
        recover_interrupted(state)
    del state["jobs"]["reference"]["pid"]
    recover_interrupted(state)
    assert state["elapsed_s"] == 25.0
    assert state["jobs"]["reference"]["status"] == "interrupted"
    recover_interrupted(state)
    assert state["elapsed_s"] == 25.0  # never charge the same interrupted checkpoint twice
