"""Bounded three-stage training and small-subset evaluation; no cloud auto-launch."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from ..config import load, path
from ..data import read_splits
from ..provenance import sha256, write_json


def validate(cfg):
    for key in (
        "latent_channels",
        "width",
        "clip_frames",
        "train_cases",
        "val_cases",
        "eval_cases",
        "eval_frames",
        "checkpoint_every",
        "validation_every",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["clip_frames"] < 2:
        raise ValueError("Prior training needs consecutive frames")
    if not cfg["groups"] or any(type(n) is not int or n < 1 for n in cfg["groups"]):
        raise ValueError("Groups must contain positive integers")
    if sum(cfg["groups"]) > 112:
        raise ValueError("Budget exceeds 112 unique lines")
    for key in ("learning_rate", "delta_limit", "redundancy_sigma", "residual_sigma"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= cfg["value_decay"] < 1:
        raise ValueError("value_decay must be in [0, 1)")
    for group in ("score_weights", "loss_weights"):
        if any(not np.isfinite(v) or v < 0 for v in cfg[group].values()):
            raise ValueError(f"Invalid {group}")
    for n in cfg["steps"].values():
        if type(n) is not int or n < 1:
            raise ValueError("Training steps must be positive")
    return cfg


def runtime(cpu=False):
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf

    if tf.keras.backend.backend() != "tensorflow":
        raise RuntimeError("Run belief filter in a fresh TensorFlow process")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not cpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU unavailable; explicit --cpu permits a local functional test")
    return tf


class Clips:
    def __init__(self, cfg, split, limit):
        names = read_splits(path(cfg["split_manifest"]))[split]
        if not names:
            raise ValueError(f"Empty {split} split")
        rng = np.random.default_rng(cfg["seed"])
        self.names = [
            names[i] for i in sorted(rng.choice(len(names), min(limit, len(names)), False))
        ]
        if split in cfg.get("case_names", {}):
            requested = cfg["case_names"][split]
            if (
                not requested
                or len(set(requested)) != len(requested)
                or not set(requested) <= set(names)
            ):
                raise ValueError(
                    "Explicit cohort must be a nonempty distinct subset of the official split"
                )
            self.names = list(requested)
        self.root = path(cfg["data_root"]) / split
        self.frames = cfg["clip_frames"]
        self.identity, self.lengths = [], []
        for name in self.names:
            file = self.root / name
            with h5py.File(file, "r") as h5:
                ds = h5["data/image"]
                if ds.shape[1:] != (112, 112) or len(ds) < self.frames:
                    raise ValueError(f"Invalid polar sequence: {file}")
                self.lengths.append(len(ds))
            st = file.stat()
            self.identity.append({"name": name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})

    def read(self, index, start, frames):
        with h5py.File(self.root / self.names[index], "r") as h5:
            data = h5["data/image"][start : start + frames].astype(np.float32)
        if not np.isfinite(data).all() or data.min() < -60.001 or data.max() > 0.001:
            raise ValueError("Input must be finite polar dB values in [-60, 0]")
        return (data / 30.0 + 1.0)[..., None]

    def sample(self, seed):
        rng = np.random.default_rng(seed)
        index = int(rng.integers(len(self.names)))
        start = int(rng.integers(self.lengths[index] - self.frames + 1))
        return self.read(index, start, self.frames), start


def save_checkpoint(file, models, optimizer, metadata):
    arrays = {"metadata": np.array(json.dumps(metadata, allow_nan=False))}
    for name, model in models.all().items():
        for index, value in enumerate(model.get_weights()):
            arrays[f"{name}_{index}"] = value
    if optimizer is not None:
        for index, value in enumerate(optimizer.variables):
            arrays[f"optimizer_{index}"] = value.numpy()
    temp = file.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(file)


def load_checkpoint(file, models, optimizer=None):
    with np.load(file, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        keys = ("latent_channels", "width", "delta_limit")
        if any(metadata["architecture"][key] != models.cfg[key] for key in keys):
            raise ValueError("Checkpoint architecture differs")
        for name, model in models.all().items():
            model.set_weights([archive[f"{name}_{i}"] for i in range(len(model.weights))])
        if optimizer is not None:
            for i, value in enumerate(optimizer.variables):
                value.assign(archive[f"optimizer_{i}"])
    return metadata


def loss_for_clip(models, clip, start, stage, cfg):
    import tensorflow as tf

    from .core import ArrayOracle, filter_loss, initial_memory, reconstruction_loss, step

    target = tf.convert_to_tensor(clip)
    if stage == "codec":
        prediction = models.decoder(models.encoder(target, training=True), training=True)
        return reconstruction_loss(target, prediction, cfg["ssim_weight"])
    if stage == "prior":
        latent = tf.stop_gradient(models.encoder(target, training=False))
        terms = []
        for frame in range(1, len(clip)):
            predicted = models.predict(latent[frame - 1 : frame], start + frame)
            terms.append(
                tf.reduce_mean(tf.abs(predicted - latent[frame : frame + 1]))
                + reconstruction_loss(
                    target[frame : frame + 1], models.decoder(predicted), cfg["ssim_weight"]
                )
            )
        return tf.add_n(terms) / len(terms)
    memory, terms = initial_memory(models), []
    for frame, truth in enumerate(clip):
        # No teacher-forced target memory. Truth is used only by oracle and supervised loss.
        memory, state = step(
            models,
            ArrayOracle(truth),
            memory,
            start + frame,
            cfg,
            cfg.get("training_policy", "greedy"),
        )
        value, _ = filter_loss(models, truth[None], state["updates"], cfg)
        terms.append(value)
    return tf.add_n(terms) / len(terms)


def train(cfg, output, stage, initialize=None, resume=False, cpu=False, stop_after=None):
    tf = runtime(cpu)
    from .core import Models

    tf.keras.utils.set_random_seed(cfg["seed"])
    training, validation = (
        Clips(cfg, "train", cfg["train_cases"]),
        Clips(cfg, "val", cfg["val_cases"]),
    )
    models = Models(cfg)
    variables = models.configure_stage(stage)
    optimizer = tf.keras.optimizers.Adam(cfg["learning_rate"])
    optimizer.build(variables)
    identity = {
        "config": cfg,
        "stage": stage,
        "split_sha256": sha256(path(cfg["split_manifest"])),
        "train_files": training.identity,
        "val_files": validation.identity,
    }
    checkpoint = output / "checkpoint.npz"
    start = 0
    parent_hash = None
    if resume:
        saved = load_checkpoint(checkpoint, models, optimizer)
        if saved["identity"] != identity:
            raise ValueError("Resume requires identical data/config/stage")
        start, parent_hash = saved["step"], saved["parent_sha256"]
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use an empty output or --resume; existing results are preserved")
        if stage != "codec":
            if initialize is None:
                raise ValueError("Frozen components require a trained parent checkpoint")
            parent = load_checkpoint(initialize, models)
            expected = {"prior": "codec", "filter": "prior"}[stage]
            if parent["stage"] != expected or parent["status"] != "completed":
                raise ValueError(f"Expected completed {expected} checkpoint")
            if parent["identity"]["split_sha256"] != identity["split_sha256"]:
                raise ValueError("Parent used a different split")
            parent_hash = sha256(initialize)
        elif initialize is not None:
            raise ValueError("Codec stage starts from scratch; --initialize is for later stages")
    output.mkdir(parents=True, exist_ok=True)
    total = cfg["steps"][stage]
    if resume and start >= total:
        print(f"{stage.upper()}_ALREADY_COMPLETED", flush=True)
        return
    end = min(total, start + stop_after) if stop_after is not None else total
    metadata = {
        "architecture": cfg,
        "identity": identity,
        "stage": stage,
        "step": start,
        "status": "running",
        "parent_sha256": parent_hash,
        "variant": "thesis-inspired; no senior or official CASL weights",
        "trainable_parameters": sum(int(np.prod(v.shape)) for v in variables),
        "selection": "seeded video subset; chronological clips; stop-gradient across frames",
    }
    # A killed pilot must also be resumable before its first periodic checkpoint.
    if not resume:
        save_checkpoint(checkpoint, models, optimizer, metadata)
    write_json(output / "status.json", metadata)
    print(
        f"TRAIN {stage}: {start}/{total}; parameters={metadata['trainable_parameters']}", flush=True
    )
    timings = []
    try:
        with (output / "training.jsonl").open("a", encoding="utf-8") as log:
            for index in range(start, end):
                t0 = time.perf_counter()
                clip, frame_start = training.sample(cfg["seed"] + index)
                with tf.GradientTape() as tape:
                    loss = loss_for_clip(models, clip, frame_start, stage, cfg)
                tf.debugging.assert_all_finite(loss, "Nonfinite loss")
                grads = tape.gradient(loss, variables)
                if any(g is None for g in grads):
                    raise RuntimeError("Missing gradient")
                for gradient in grads:
                    tf.debugging.assert_all_finite(gradient, "Nonfinite gradient")
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                optimizer.apply_gradients(zip(grads, variables))
                record = {
                    "step": index + 1,
                    "loss": float(loss),
                    "train_step_s": time.perf_counter() - t0,
                }
                if (index + 1) % cfg["validation_every"] == 0 or index + 1 == end:
                    losses = [
                        float(
                            loss_for_clip(
                                models, validation.read(i, 0, cfg["clip_frames"]), 0, stage, cfg
                            )
                        )
                        for i in range(len(validation.names))
                    ]
                    record["val_loss"] = float(np.mean(losses))
                record["wall_s"] = time.perf_counter() - t0
                timings.append(record["wall_s"])
                if len(timings) >= 10:
                    # Includes validation in the window; excludes checkpoint writes and startup.
                    record["projected_remaining_minutes"] = (
                        float(np.mean(timings[-50:])) * (total - index - 1) / 60
                    )
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                metadata.update(
                    step=index + 1, status="completed" if index + 1 == total else "running"
                )
                if (index + 1) % cfg["checkpoint_every"] == 0 or index + 1 == end:
                    save_checkpoint(checkpoint, models, optimizer, metadata)
                    write_json(output / "status.json", metadata)
        if end < total:
            metadata["status"] = "paused"
            save_checkpoint(checkpoint, models, optimizer, metadata)
        write_json(output / "status.json", metadata)
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "status.json", metadata)
        raise
    print(f"{stage.upper()}_{metadata['status'].upper()}", flush=True)


def evaluate(cfg, checkpoint, output, split="val", policy="greedy", cpu=False, save_frames=3):
    runtime(cpu)
    from PIL import Image

    from ..evaluation.metrics import Metrics, uint8_image
    from .core import ArrayOracle, Models, initial_memory, step

    if split not in ("val", "test"):
        raise ValueError("Evaluate on held-out data")
    models = Models(cfg)
    saved = load_checkpoint(checkpoint, models)
    if saved["stage"] != "filter" or saved["status"] != "completed":
        raise ValueError("Evaluation requires a completed filter-stage checkpoint")
    if saved["identity"]["split_sha256"] != sha256(path(cfg["split_manifest"])):
        raise ValueError("Evaluation split differs from training")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    data = Clips(cfg, split, cfg["eval_cases"])
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "variant": cfg["variant"],
        "config": cfg,
        "checkpoint_sha256": sha256(checkpoint),
        "split": split,
        "policy": policy,
        "cases": data.names,
        "budget": sum(cfg["groups"]),
        "files": data.identity,
        "metric_domain": "polar_uint8",
        "training_steps": saved["step"],
        "timing": "eager synchronized closed loop includes policy/oracle; excludes metrics/I/O",
        "full_reproduction": False,
    }
    write_json(output / "manifest.json", report)
    metrics, summaries = Metrics(["psnr", "ssim"]), []
    try:
        for case, name in enumerate(data.names):
            memory, rows = initial_memory(models), []
            frames = data.read(case, 0, cfg["eval_frames"])
            directory = output / Path(name).stem
            directory.mkdir()
            for index, target in enumerate(frames):
                started = time.perf_counter()
                memory, state = step(models, ArrayOracle(target), memory, index, cfg, policy)
                prediction = np.asarray(state["reconstruction"])[0]
                elapsed = time.perf_counter() - started
                row = {
                    "frame": index,
                    "wall_s": elapsed,
                    "actual_lines": int(state["mask"][0, 0, :, 0].sum()),
                    **metrics(target, prediction),
                }
                rows.append(row)
                if index < save_frames:
                    frame_dir = directory / f"frame_{index:04d}"
                    frame_dir.mkdir()
                    pictures = {
                        "ground_truth": target,
                        "reconstruction": prediction,
                        "sparse_observation": np.where(
                            state["mask"][0] > 0, state["observation"][0], -1
                        ),
                    }
                    for label, pixels in pictures.items():
                        Image.fromarray(uint8_image(pixels).squeeze()).save(
                            frame_dir / f"{label}.png"
                        )
                    Image.fromarray((state["mask"][0, ..., 0] * 255).astype("uint8")).save(
                        frame_dir / "acquired_lines.png"
                    )
                    uncertainty = np.asarray(state["uncertainty"])[0, ..., 0]
                    # Fixed, documented display scales; raw arrays remain in state.npz.
                    Image.fromarray((np.clip(uncertainty / 4.1, 0, 1) * 255).astype("uint8")).save(
                        frame_dir / "uncertainty.png"
                    )
                    Image.fromarray(
                        (np.abs(target - prediction)[..., 0] / 2 * 255).astype("uint8")
                    ).save(frame_dir / "absolute_error.png")
                    np.savez_compressed(
                        frame_dir / "state.npz",
                        target=target,
                        prediction=prediction,
                        mask=state["mask"][0],
                        uncertainty=np.asarray(state["uncertainty"])[0],
                        group_reconstructions=np.stack(
                            [g["reconstruction"] for g in state["groups"]]
                        ),
                        group_scores=np.stack([g["scores"] for g in state["groups"]]),
                    )
                    write_json(frame_dir / "actions.json", [g["lines"] for g in state["groups"]])
            with (directory / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = {
                "case": name,
                "frames": len(rows),
                "psnr": float(np.mean([r["psnr"] for r in rows])),
                "ssim": float(np.mean([r["ssim"] for r in rows])),
                "steady_fps": (len(rows) - 1) / sum(r["wall_s"] for r in rows[1:])
                if len(rows) > 1
                else None,
            }
            summaries.append(summary)
            write_json(directory / "complete.json", summary)
            print(f"EVAL {case + 1}/{len(data.names)} {name}: {summary}", flush=True)
        report.update(
            status="completed",
            results=summaries,
            psnr_patient_mean=float(np.mean([r["psnr"] for r in summaries])),
            ssim_patient_mean=float(np.mean([r["ssim"] for r in summaries])),
        )
        write_json(output / "manifest.json", report)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        write_json(output / "manifest.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--config", default="configs/belief_filter/pilot.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("codec", "prior", "filter"))
    parser.add_argument("--initialize")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--policy", choices=("greedy", "uniform"), default="greedy")
    args = parser.parse_args(argv)
    cfg = validate(load(path(args.config)))
    if args.command == "train":
        if args.stage is None or args.stop_after is not None and args.stop_after < 1:
            parser.error("train requires --stage; --stop-after must be positive")
        train(
            cfg,
            path(args.output),
            args.stage,
            path(args.initialize) if args.initialize else None,
            args.resume,
            args.cpu,
            args.stop_after,
        )
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        evaluate(cfg, path(args.checkpoint), path(args.output), args.split, args.policy, args.cpu)
