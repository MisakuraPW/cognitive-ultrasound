"""Real CPU objective/resume functional tests; not a GPU speed/quality claim."""

import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml

from cognitive_ultrasound.config import ROOT, load
from cognitive_ultrasound.preparation.common import atomic_json, read_json


@pytest.fixture
def setup(tmp_path):
    spec = load(ROOT / "configs/belief_filter/pilot.yaml")
    spec.update(width=2, latent_channels=2, clip_frames=3, train_cases=1, val_cases=1)
    bf = tmp_path / "bf.yaml"
    bf.write_text(yaml.safe_dump(spec))
    casl_spec = load(ROOT / "configs/training.yaml")
    # Pinned upstream clone/Lambda serialization assumes its 32-dimensional embedding.
    casl_spec["network_kwargs"].update(widths=[2, 4], block_depth=1)
    casl = tmp_path / "casl.yaml"
    casl.write_text(yaml.safe_dump(casl_spec))
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump(dict(train=["train.hdf5"], val=["val.hdf5"], test=[])))
    (tmp_path / "train").mkdir()
    with h5py.File(tmp_path / "train/train.hdf5", "w") as f:
        f.create_dataset(
            "data/image",
            data=np.random.default_rng(1).uniform(-60, 0, (6, 112, 112)).astype(np.float32),
        )
    cfg = dict(
        data_root=str(tmp_path),
        split_manifest=str(split),
        bf_config=str(bf),
        casl_training_config=str(casl),
        training_batch_size=1,
        checkpoint=str(ROOT / "checkpoints/official"),
    )
    atomic_json(tmp_path / "config.json", cfg)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        PYTHONUTF8="1",
        KERAS_BACKEND="tensorflow",
        CUDA_VISIBLE_DEVICES="-1",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="2",
    )
    return tmp_path, cfg, env


def run_stage(setup, stage, mode, until, name, initialize=False, resume=None):
    root, cfg, env = setup
    output = root / name
    output.mkdir()
    atomic_json(
        output / "task.json",
        dict(
            kind="training",
            workload=stage,
            mode=mode,
            until=until,
            initialize=initialize,
            resume_from=str(resume) if resume else None,
            output=str(output),
            cpu_functional_test=True,
        ),
    )
    # Make deterministic plan in a fresh process: coordinator never imports TF.
    if not (root / "training" / (stage + ".batches.json")).exists():
        script = "from pathlib import Path; from cognitive_ultrasound.preparation.common import read_json; from cognitive_ultrasound.compute_lab.training import prepare_batches; import sys; r=Path(sys.argv[1]); prepare_batches(read_json(r/'config.json'),sys.argv[2],r/'training'/(sys.argv[2]+'.batches.json'))"
        p = subprocess.run(
            [sys.executable, "-c", script, str(root), stage],
            env=env,
            capture_output=True,
            timeout=120,
        )
        assert p.returncode == 0, p.stderr.decode("utf-8", errors="replace")
    with (output / "test.log").open("w", encoding="utf-8") as log:
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "cognitive_ultrasound.compute_lab",
                "worker",
                "--config",
                str(root / "config.json"),
                "--task",
                str(output / "task.json"),
                "--output",
                str(root),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=240,
        )
    assert p.returncode == 0, (output / "test.log").read_text(encoding="utf-8")
    return output


def test_real_codec_200_updates_exit_and_resume(setup):
    from cognitive_ultrasound.compute_lab.training import compare

    run_stage(setup, "codec", "eager", 0, "init", initialize=True)
    initial = (setup[0] / "training/codec.initial.npz").read_bytes()
    full = run_stage(setup, "codec", "graph", 200, "continuous")
    first = run_stage(setup, "codec", "graph", 100, "first")
    resumed = run_stage(setup, "codec", "graph", 200, "resumed", resume=first / "checkpoint.npz")
    result = compare(full, resumed)
    assert result["passed"], result
    assert read_json(full / "result.json")["frozen_verified"]
    assert (setup[0] / "training/codec.initial.npz").read_bytes() == initial
    a, b = read_json(full / "result.json"), read_json(resumed / "result.json")
    np.testing.assert_allclose(
        [r["loss"] for r in a["records"][100:]],
        [r["loss"] for r in b["records"]],
        atol=2e-4,
        rtol=2e-4,
    )


@pytest.mark.parametrize("stage", ["prior", "filter", "casl"])
def test_real_other_objectives_gradient_optimizer_and_freezing(setup, stage):
    run_stage(setup, stage, "eager", 0, "init", initialize=True)
    eager = run_stage(setup, stage, "eager", 2, "eager")
    graph = run_stage(setup, stage, "graph", 2, "graph")
    from cognitive_ultrasound.compute_lab.training import compare

    result = compare(eager, graph)
    assert result["passed"], result
    value = read_json(graph / "result.json")
    assert value["frozen_verified"]
    assert "optimizer" in value["state_components"]
    if stage == "casl":
        assert value["ema"] == "included"
