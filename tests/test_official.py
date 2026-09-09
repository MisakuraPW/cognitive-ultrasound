import numpy as np
import pytest

from cognitive_ultrasound.official import activate

pytestmark = pytest.mark.official


@pytest.fixture(scope="module", autouse=True)
def backend():
    activate("jax")
    pytest.importorskip("jax")
    pytest.importorskip("keras")


def test_entropy_matches_official_selector():
    from ulsa.entropy import pixelwise_entropy
    from zea.agent.selection import GreedyEntropy

    selector = GreedyEntropy(7, 112, 112, 112)
    p = np.random.default_rng(2).normal(size=(1, 2, 112, 112)).astype("float32")
    np.testing.assert_allclose(
        np.asarray(pixelwise_entropy(p)),
        np.asarray(selector.compute_pixelwise_entropy(p)),
        atol=1e-6,
    )


@pytest.mark.parametrize("name", ["uniform_random", "equispaced", "greedy_entropy"])
def test_sampler_adapter_matches_official(name):
    import jax
    from ulsa.agent import action_selection_wrapper
    from ulsa.selection import selector_from_name

    from cognitive_ultrasound.acquisition import OfficialSampler, PolicyState

    fn = action_selection_wrapper(
        selector_from_name(name, n_actions=7, n_possible_actions=112, img_width=112, img_height=112)
    )
    p = np.random.default_rng(3).normal(size=(2, 112, 112)).astype("float32")
    key = jax.random.PRNGKey(42)
    expected = fn(p, None, key)
    actual = OfficialSampler(fn).select_action(None, PolicyState(p, None, key))
    np.testing.assert_array_equal(actual[0], expected[0])
    assert np.count_nonzero(actual[0]) == 7
    assert actual[1].shape == (112, 112, 1)


def test_uniform_roll_and_random_reproducibility():
    import jax
    from zea.agent.selection import EquispacedLines, UniformRandomLines

    uniform = EquispacedLines(7, 112, 112, 112)
    initial, _ = uniform.initial_sample_stateless()
    following, _ = uniform.sample_stateless(initial)
    np.testing.assert_array_equal(following, np.roll(initial, 1, axis=-1))
    random = UniformRandomLines(7, 112, 112, 112)
    np.testing.assert_array_equal(
        random.sample(seed=jax.random.PRNGKey(1))[0], random.sample(seed=jax.random.PRNGKey(1))[0]
    )


def test_buffer_and_projection_upstream_semantics():
    from ulsa.agent import hard_projection
    from ulsa.buffer import FrameBuffer

    buffer = FrameBuffer((112, 112, 3), buffer_size=3)
    buffer.shift(np.ones((112, 112, 1), dtype="float32"))
    buffer.shift(np.full((112, 112, 1), 2, dtype="float32"))
    np.testing.assert_array_equal(np.asarray(buffer.buffer)[0, 0], [0, 1, 2])
    # Explicitly preserve upstream nonzero-measurement semantics (known zero-value gap).
    projected = np.asarray(hard_projection(np.ones(3), np.array([0.0, -0.5, 0.5])))
    np.testing.assert_array_equal(projected, [1.0, -0.5, 0.5])


def test_preprocessing_matches_official_pipeline():
    from ulsa.pipeline import make_pipeline

    raw = np.random.default_rng(4).uniform(-60, 0, (112, 112)).astype("float32")
    pipeline = make_pipeline(
        data_type="data/image",
        output_range=(-1, 1),
        output_shape=(112, 112, 3),
        action_selection_shape=(112, 112),
    )
    parameters = pipeline.prepare_parameters(dynamic_range=(-60, 0))
    result = np.asarray(pipeline(data=raw, **parameters)["data"])
    np.testing.assert_allclose(result, ((raw + 60.0) / 30.0 - 1.0)[..., None], atol=1e-6)


def test_converter_cli_reads_extracted_dataset_and_preserves_sources(tmp_path):
    """Exercise the real pinned CLI, AVI decoder, split assignment and HDF5 writer."""
    import imageio.v2 as imageio
    import yaml

    from cognitive_ultrasound.data import convert, inspect_file
    from cognitive_ultrasound.provenance import sha256

    raw = tmp_path / "shared/EchoNet-Dynamic"
    videos = raw / "Videos"
    videos.mkdir(parents=True)
    split_dir = tmp_path / "manifests"
    split_dir.mkdir()
    split_file = split_dir / "split.yaml"
    split_file.write_text(
        yaml.safe_dump({split: [f"{split}.hdf5"] for split in ("train", "val", "test")}),
        encoding="utf-8",
    )
    for name in ("train", "val", "test", "rejected"):
        frame = np.full((112, 112, 3), 0 if name == "rejected" else 128, dtype=np.uint8)
        imageio.mimwrite(videos / f"{name}.avi", [frame], fps=10, codec="ffv1")
    before = {p.relative_to(raw): sha256(p) for p in raw.rglob("*") if p.is_file()}
    output = tmp_path / "polar"
    convert(raw, output, split_file)
    for split in ("train", "val", "test"):
        assert inspect_file(output / split / f"{split}.hdf5")["frames"] == 1
    assert (output / "rejected/rejected.hdf5").is_file()
    converted = yaml.safe_load((output / "split.yaml").read_text(encoding="utf-8"))
    assert converted == {name: [f"{name}.hdf5"] for name in ("train", "val", "test", "rejected")}
    assert before == {p.relative_to(raw): sha256(p) for p in raw.rglob("*") if p.is_file()}
