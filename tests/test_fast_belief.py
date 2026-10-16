"""Real Torch/HDF5 functional checks; synthetic pixels are never scientific results."""

import copy
import csv
import json
import shutil
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from cognitive_ultrasound.fast_belief.data import (
    Clips,
    file_lock,
    frames,
    load_config,
    normalize,
    prepare,
    read,
    sha,
    verify,
)
from cognitive_ultrasound.fast_belief.evaluation import final_evaluation, measures
from cognitive_ultrasound.fast_belief.learning import update
from cognitive_ultrasound.fast_belief.model import ArrayOracle, BeliefSystem, LinearResize, select
from cognitive_ultrasound.fast_belief.runner import (
    configure,
    fit,
    frozen_signature,
    load,
    run,
    training_loader,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path):
    configure(True, 1)
    cfg = load_config(REPO / "configs/fast_belief.yaml")
    cfg["model"].update(width=4, latent_channels=2)
    cfg["cohorts"] = dict(train=4, development=1, test=1)
    for s in ["codec", "prior", "filter"]:
        cfg["training"][s] = dict(epochs=1, batch=2, frames=3 if s != "filter" else 4)
    cfg["training"]["filter"].update(burn_in=1, bptt=2)
    cfg["training"].update(checkpoint_updates=1, validate_every=1)
    cfg["runtime"].update(workers=0, threads=1, calibrate_threads=[1], free_gib=0)
    cfg["evaluation"]["component_frames"] = 6
    root = tmp_path / "data"
    split = dict(train=[f"train{i}.hdf5" for i in range(4)], val=["val.hdf5"], test=["test.hdf5"])
    rng = np.random.default_rng(777)
    for s, names in split.items():
        (root / s).mkdir(parents=True)
        for name in names:
            # Temporal correlation gives the actual recurrence something to process.
            base = rng.uniform(-55, -5, (1, 112, 112))
            pixels = np.clip(base + rng.normal(0, 1, (6, 112, 112)), -60, 0).astype("f4")
            with h5py.File(root / s / name, "w") as h:
                h.create_dataset("data/image", data=pixels)
    cfg["data_root"] = str(root)
    cfg["file_list"] = str(tmp_path / "FileList.csv")
    with open(cfg["file_list"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["FileName", "Split", "EF"])
        writer.writeheader()
        for s, names in split.items():
            for n in names:
                writer.writerow(dict(FileName=Path(n).stem, Split=s.upper(), EF=55))
    cfg["split_manifest"] = str(tmp_path / "split.yaml")
    Path(cfg["split_manifest"]).write_text(yaml.safe_dump(split), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, prepare(cfg), config


def test_causal_acquisition_batch_and_frame_reset(setup):
    cfg, _, _ = setup
    model = BeliefSystem(cfg).eval()
    target = torch.rand(2, 1, 112, 112) * 2 - 1
    with torch.no_grad():
        memory = model.initial(2, "cpu")
        next_memory, original = model.frame(ArrayOracle(target), memory, 0)
        assert [a.shape for a in original["actions"]] == [(2, 10), (2, 4)]
        assert torch.equal(original["mask"].sum((1, 2, 3)), torch.tensor([1568.0, 1568.0]))
        assert torch.equal(original["image"] * original["mask"], target * original["mask"])
        # Arbitrarily change all unobserved pixels: both decisions and result must be unchanged.
        changed = torch.where(original["mask"].bool(), target, -target)
        _, replay = model.frame(ArrayOracle(changed), memory, 0)
        assert torch.equal(original["image"], replay["image"])
        for a, b in zip(original["actions"], replay["actions"]):
            assert torch.equal(a, b)
        # Batch processing has no cross-video normalization/dependence.
        single = model.initial(1, "cpu")
        _, independent = model.frame(ArrayOracle(target[:1]), single, 0)
        assert torch.allclose(original["image"][:1], independent["image"], atol=2e-4, rtol=2e-4)
        # The next frame overwrites every newly observed column, never retaining old hard data.
        future = torch.full_like(target, -0.9)
        _, later = model.frame(ArrayOracle(future), next_memory, 1)
        assert torch.equal(later["image"] * later["mask"], future * later["mask"])
        assert torch.equal(later["mask"].sum((1, 2, 3)), torch.tensor([1568.0, 1568.0]))


def test_real_gradients_frozen_parts_and_nonfinite(setup):
    cfg, manifest, _ = setup
    for stage in ["codec", "prior", "filter"]:
        model = BeliefSystem(cfg)
        optimizer = torch.optim.Adam(model.stage(stage), lr=0.001)
        frozen = frozen_signature(model, stage)
        before = {k: p.clone() for k, p in model.named_parameters() if p.requires_grad}
        x, start = next(iter(training_loader(manifest, cfg, stage, 0, 0, 0)))
        value = update(model, optimizer, x, start, stage)
        assert np.isfinite(list(value.values())).all()
        assert frozen_signature(model, stage) == frozen
        assert any(
            not torch.equal(before[k], p) for k, p in model.named_parameters() if k in before
        )
        # Frozen decoder must still propagate an input gradient to learned latent updates.
        if stage == "filter":
            z = torch.randn(1, 2, 28, 28, requires_grad=True)
            grad = torch.autograd.grad(model.codec.decoder(z).square().mean(), z)[0]
            assert torch.isfinite(grad).all() and grad.abs().max() > 0
        x[0, 0, 0, 0, 0] = float("nan")
        with pytest.raises((FloatingPointError, RuntimeError)):
            update(model, optimizer, x, start, stage)


def test_greedy_ties_exclusions_and_metrics():
    configure(True, 1)
    scores = torch.zeros(2, 112)
    used = torch.zeros_like(scores, dtype=torch.bool)
    used[:, 55] = True
    lines = select(scores, used, 14, 0.5, 8)
    assert torch.equal(lines[0], lines[1])
    assert all(len(set(x.tolist())) == 14 and 55 not in x for x in lines)
    with pytest.raises(ValueError):
        select(scores[:, :3], used, 14, 0.5, 8)
    x = torch.rand(1, 1, 112, 112)
    assert measures(x, x)["ssim"] == pytest.approx(1)
    with pytest.raises(FloatingPointError):
        measures(x, x * float("nan"))


def test_manifest_content_and_loader_order(setup):
    cfg, manifest, _ = setup
    groups = [set(v) for v in manifest["cohorts"].values()]
    assert not (groups[0] & groups[1] | groups[0] & groups[2] | groups[1] & groups[2])
    # Spawned Windows worker opens its own HDF5; prefetch must preserve clips and order.
    a = list(training_loader(manifest, cfg, "codec", 2, 0, 0))
    b = list(training_loader(manifest, cfg, "codec", 2, 0, 1))
    for (x, s), (y, t) in zip(a, b):
        assert torch.equal(x, y) and torch.equal(s, t)
    ds = Clips(manifest, manifest["cohorts"]["train"], 3, 7, 0)
    assert np.array_equal(ds[0][0], ds[0][0])
    with pytest.raises(ValueError):
        normalize(np.array([1.0]))
    n = manifest["cohorts"]["train"][0]
    with h5py.File(Path(manifest["root"]) / "train" / n, "r+") as h:
        h["data/image"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        verify(manifest)
    with pytest.raises(ValueError):
        frames(manifest, n)


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("stage", ["codec", "prior", "filter"])
def test_training_resume_exact_weights_adam_rng_and_cursor(setup, tmp_path, stage):
    cfg, manifest, _ = setup
    runtime = dict(chosen_workers=0)
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    full.mkdir()
    interrupted.mkdir()
    for parent in ["codec", "prior"][: ["codec", "prior", "filter"].index(stage)]:
        fit(cfg, manifest, full, parent, torch.device("cpu"), "fixture", runtime)
        shutil.copytree(full / parent, interrupted / parent)
    fit(cfg, manifest, full, stage, torch.device("cpu"), "fixture", runtime)
    fit(cfg, manifest, interrupted, stage, torch.device("cpu"), "fixture", runtime, max_updates=1)
    assert load(interrupted / stage / "last.pt")["state"]["next_index"] == 2
    # Simulate log write beyond the last committed checkpoint.
    with (interrupted / stage / "training.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(update=999)) + "\n")
    fit(cfg, manifest, interrupted, stage, torch.device("cpu"), "fixture", runtime)
    a, b = load(full / stage / "last.pt"), load(interrupted / stage / "last.pt")
    for key in ["model", "optimizer", "rng", "cuda_rng"]:
        assert_nested_equal(a[key], b[key])
    for key in a["state"]:
        if key != "validation_history":
            assert_nested_equal(a["state"][key], b["state"][key])
    assert a["state"]["validation_history"][0]["mse"] == b["state"]["validation_history"][0]["mse"]
    assert len((interrupted / stage / "training.jsonl").read_text().splitlines()) == 2
    assert b["state"]["completed"] and b["state"]["updates"] == 2
    with file_lock(tmp_path / "lock"):
        with pytest.raises(OSError):
            with file_lock(tmp_path / "lock"):
                pass


def test_complete_pipeline_restart_archives_and_evaluation_integrity(setup, tmp_path):
    cfg, manifest, config = setup
    out = tmp_path / "pipeline"
    run(config, out, cpu=True)
    assert read(out / "complete.json")["cpu_functional"]
    receipt = read(out / "export.json")
    for bundle in receipt.values():
        assert sha(bundle["archive"]) == bundle["sha256"]
        with tarfile.open(bundle["archive"]) as t:
            assert len(t.getmembers()) == len(bundle["files"])
            if bundle == receipt["analysis"]:
                assert not any(m.name.endswith(".pt") for m in t)
    assert len(list(out.glob("*/*.pt"))) == 6
    assert not list(out.rglob("*.hdf5"))
    assert not list(out.rglob("*.tmp"))
    rows = read(out / "results.json")
    assert len(rows) == 6
    assert all(r["frames"] == 6 and r["observed_max_error"] == 0 for r in rows)
    stamps = {str(p): p.stat().st_mtime_ns for p in out.rglob("*.pt")}
    run(config, out, cpu=True, resume=True)
    assert stamps == {str(p): p.stat().st_mtime_ns for p in out.rglob("*.pt")}
    # Resume does not blindly accept a damaged per-case CSV.
    case = next((out / "evaluation").rglob("frames.csv"))
    case.write_text("damaged", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        final_evaluation(
            BeliefSystem(cfg),
            manifest,
            out / "evaluation",
            torch.device("cpu"),
            rows[0]["identity"],
            lambda: None,
        )
    changed = copy.deepcopy(cfg)
    changed["training"]["codec"]["epochs"] += 1
    config.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        run(config, out, cpu=True, resume=True)


def test_stop_preserves_checkpoint_and_invalid_config(setup, tmp_path):
    cfg, manifest, config = setup
    out = tmp_path / "stopped"
    out.mkdir()
    (out / "STOP").touch()
    with pytest.raises(InterruptedError):
        fit(cfg, manifest, out, "codec", torch.device("cpu"), "fixture", dict(chosen_workers=0))
    assert load(out / "codec/last.pt")["state"]["updates"] == 0
    cfg["precision"] = "fp16"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config)


@pytest.mark.parametrize("source,target", [(7, 14), (14, 28), (28, 56), (56, 112), (28, 112)])
def test_deterministic_bilinear_values_and_gradients(source, target):
    x = torch.randn(2, 3, source, source, requires_grad=True)
    own = LinearResize(source, target)(x)
    standard = torch.nn.functional.interpolate(
        x, size=(target, target), mode="bilinear", align_corners=False
    )
    assert torch.allclose(own, standard, atol=2e-6, rtol=2e-6)
    noise = torch.randn_like(own)
    a = torch.autograd.grad((own * noise).sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad((standard * noise).sum(), x)[0]
    assert torch.allclose(a, b, atol=2e-5, rtol=2e-5)
