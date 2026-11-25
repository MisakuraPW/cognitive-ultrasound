"""Train the ViViT-style transition for the pinned 112x112 CASL model."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import h5py
import keras
import numpy as np
import yaml
from keras import layers, ops

from .preparation.common import atomic_json, read_json
from .preparation.suite import run_lock
from .provenance import sha256
from .speedbench import fingerprint


CONTEXT = 4
SIZE = 112
PATCH = 16


@keras.saving.register_keras_serializable(package="casl")
class Position(layers.Layer):
    def build(self, shape):
        self.position = self.add_weight(
            shape=shape[1:], initializer=keras.initializers.RandomNormal(stddev=0.02),
            name="position",
        )

    def call(self, x):
        return x + self.position


def transformer(tokens, width, memory=None):
    norm = layers.LayerNormalization()(tokens)
    tokens = tokens + layers.MultiHeadAttention(num_heads=8, key_dim=width // 8)(norm, norm)
    if memory is not None:
        norm = layers.LayerNormalization()(tokens)
        tokens = tokens + layers.MultiHeadAttention(num_heads=8, key_dim=width // 8)(norm, memory)
    norm = layers.LayerNormalization()(tokens)
    return tokens + layers.Dense(width)(layers.Dense(4 * width, activation="gelu")(norm))


def predictor(width=128):
    grid = SIZE // PATCH
    history = keras.Input((CONTEXT, SIZE, SIZE, 1))
    packed = layers.Conv3D(width, (2, PATCH, PATCH), strides=(2, PATCH, PATCH))(history)
    memory = Position()(layers.Reshape((2 * grid**2, width))(packed))
    for _ in range(2):
        memory = transformer(memory, width)
    previous = history[:, -1]
    packed = layers.Conv2D(width, PATCH, strides=PATCH)(previous)
    queries = Position()(layers.Reshape((grid**2, width))(packed))
    for _ in range(2):
        queries = transformer(queries, width, memory=memory)
    tiles = layers.Dense(PATCH**2, kernel_initializer="zeros")(layers.LayerNormalization()(queries))
    tiles = layers.Reshape((grid, grid, PATCH, PATCH))(tiles)
    tiles = layers.Permute((1, 3, 2, 4))(tiles)
    residual = layers.Reshape((SIZE, SIZE, 1))(tiles)
    return keras.Model(history, ops.clip(previous + residual, -1, 1), name="vivit_transition")


def load_predictor(path):
    model = keras.models.load_model(path, compile=False, safe_mode=True)
    if tuple(model.input_shape[1:]) != (CONTEXT, SIZE, SIZE, 1) or tuple(model.output_shape[1:]) != (SIZE, SIZE, 1):
        raise ValueError("Unexpected ViViT checkpoint geometry")
    return model


def lock_split(data, split_file, reserved, size, seed, dev_size):
    splits = yaml.safe_load(split_file.read_text(encoding="utf-8"))
    groups = {fold: set(splits[fold]) for fold in ("train", "val", "test")}
    if any(len(groups[fold]) != len(splits[fold]) for fold in groups) or (
        len(set.union(*groups.values())) != sum(map(len, groups.values()))
    ):
        raise ValueError("Invalid CASL split")
    if not set(reserved) <= groups["val"]:
        raise ValueError("Benchmark reservations must be from validation")
    rng = np.random.default_rng(seed)
    if size is None:
        size = len(groups["train"])
    train = rng.permutation(sorted(groups["train"])).tolist()[:size]
    dev = rng.permutation(sorted(groups["val"] - set(reserved))).tolist()[:dev_size]
    if len(train) != size or len(dev) != dev_size:
        raise ValueError("Insufficient locked cases")
    files = {}
    for fold, cases in (("train", train), ("val", dev)):
        for name in cases:
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Unexpected HDF5 path")
            files[f"{fold}/{name}"] = fingerprint(data / fold / name)
    return dict(
        train=train, development=dev, benchmark=reserved, test_used=False,
        files=files, split_sha256=sha256(split_file), seed=seed,
    )


def clips(data, names, fold, rng):
    for name in names:
        with h5py.File(data / fold / name) as handle:
            image = handle["data/image"]
            if image.ndim != 3 or image.shape[1:] != (SIZE, SIZE) or len(image) <= CONTEXT:
                raise ValueError(f"Expected at least five polar frames in {name}")
            start = int(rng.integers(len(image) - CONTEXT))
            frame = image[start : start + CONTEXT + 1].astype(np.float32)
        if not np.isfinite(frame).all() or frame.min() < -60.001 or frame.max() > 0.001:
            raise ValueError(f"Invalid polar image in {name}")
        yield (frame / 30 + 1)[..., None]


def validate(model, batch):
    prediction = np.concatenate([
        np.asarray(model(inputs[:, :-1], training=False))
        for inputs in np.array_split(batch, math.ceil(len(batch) / 8))
    ])
    truth = batch[:, -1]
    error = np.abs(prediction - truth)
    if not np.isfinite(error).all():
        raise FloatingPointError("ViViT prediction is nonfinite")
    return float(error.mean()), float(np.abs(batch[:, -2] - truth).mean())


def save_model(model, output, name):
    path = output / name
    temp = output / "saving.keras"
    model.save(temp)
    temp.replace(path)
    return {"name": name, "sha256": sha256(path)}


def verified_file(output, reference):
    path = output / reference["name"]
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Checkpoint fingerprint changed: {path}")
    return path


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with run_lock(args.output):
        return fit(args)


def fit(args):
    import jax

    if keras.backend.backend() != "jax" or (not args.cpu and jax.default_backend() != "gpu"):
        raise RuntimeError("ViViT training requires Keras/JAX on CUDA")
    jax.config.update("jax_default_matmul_precision", "highest")
    keras.mixed_precision.set_global_policy("float32")
    keras.utils.set_random_seed(args.seed)
    state_file = args.output / "training_state.json"
    manifest_file = args.output / "manifest.json"
    identity = dict(
        cases=args.cases, development=args.development, epochs=args.epochs, batch=args.batch,
        min_epochs=args.min_epochs, patience=args.patience, seed=args.seed, cpu=args.cpu,
        data=str(args.data.resolve()), split_sha256=sha256(args.split), reserved=args.reserved,
        source_sha256=sha256(__file__), keras=keras.__version__, jax=jax.__version__,
    )
    if manifest_file.exists():
        manifest = read_json(manifest_file)
        if manifest["identity"] != identity:
            raise ValueError("ViViT resume requires the same code, data and training protocol")
        for relative, recorded in manifest["files"].items():
            if fingerprint(args.data / relative) != recorded:
                raise ValueError(f"Training input changed: {relative}")
    else:
        manifest = lock_split(args.data, args.split, args.reserved, args.cases, args.seed, args.development)
        manifest["identity"] = identity
        atomic_json(manifest_file, manifest)

    complete = args.output / "complete.json"
    if complete.exists():
        record = read_json(complete)
        if record["checkpoint_sha256"] != sha256(args.output / "best.keras"):
            raise ValueError("Selected ViViT checkpoint changed")
        return

    if state_file.exists():
        state = read_json(state_file)
        model = keras.models.load_model(verified_file(args.output, state["last"]), safe_mode=True)
        if int(np.asarray(model.optimizer.iterations)) != state["updates"]:
            raise ValueError("ViViT optimizer state does not match the update cursor")
        print(json.dumps(dict(event="RESUME", epoch=state["epoch"], updates=state["updates"])), flush=True)
    else:
        model = predictor()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, global_clipnorm=1.0), loss="mae")
        state = dict(epoch=0, offset=0, updates=0, loss_sum=0.0, best=None, best_score=None, stale=0, rows=[])

    dev = np.stack(list(clips(
        args.data, manifest["development"], "val", np.random.default_rng(args.seed + 1),
    )))
    while state["epoch"] < args.epochs and not (
        state["epoch"] >= args.min_epochs and state["stale"] >= args.patience
    ):
        epoch = state["epoch"]
        tick = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, 0]))
        training = rng.permutation(manifest["train"]).tolist()
        start_updates = state["updates"]
        last_progress = tick
        for offset in range(state["offset"], len(training), args.batch):
            clip_rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, offset, 1]))
            batch = np.stack(list(clips(args.data, training[offset : offset + args.batch], "train", clip_rng)))
            model.reset_metrics()
            loss = float(model.train_on_batch(batch[:, :-1], batch[:, -1]))
            if not math.isfinite(loss):
                raise FloatingPointError("ViViT training loss is nonfinite")
            state["offset"] = offset + len(batch)
            state["updates"] += 1
            state["loss_sum"] += loss * len(batch)
            if state["updates"] % args.checkpoint_updates == 0:
                now = time.perf_counter()
                print(json.dumps(dict(
                    event="UPDATE", epoch=epoch + 1, updates=state["updates"], videos=state["offset"],
                    total_videos=len(training), elapsed_s=now - last_progress,
                    interval_updates=min(state["updates"] - start_updates, args.checkpoint_updates),
                )), flush=True)
                persist(model, state, args.output, state_file)
                last_progress = time.perf_counter()
        mae, copy_mae = validate(model, dev)
        row = dict(epoch=epoch + 1, updates=state["updates"], loss=state["loss_sum"] / len(training),
                   mae=mae, copy_mae=copy_mae,
                   elapsed_s=time.perf_counter() - tick)
        print(json.dumps(dict(event="EPOCH", **row)), flush=True)
        improved = state["best_score"] is None or mae < state["best_score"]
        state.update(epoch=epoch + 1, offset=0, loss_sum=0.0, stale=0 if improved else state["stale"] + 1)
        state["rows"].append(row)
        if improved:
            state["best_score"] = mae
        persist(model, state, args.output, state_file, improved=improved)
        atomic_json(args.output / "learning.json", state["rows"])

    selected = verified_file(args.output, state["best"])
    staged = args.output / "selected.keras"
    shutil.copyfile(selected, staged)
    staged.replace(args.output / "best.keras")
    atomic_json(args.output / "learning.json", state["rows"])
    atomic_json(complete, dict(
        checkpoint_sha256=sha256(args.output / "best.keras"),
        best_validation_mae=state["best_score"], cases=len(manifest["train"]),
        epochs=state["epoch"], max_epochs=args.epochs, updates=state["updates"],
        cpu_functional=args.cpu, history="clean polar frames for teacher forcing", pixel_grid=112,
        context=CONTEXT, tubelet=[2, PATCH, PATCH], transformer_depth=[2, 2], heads=8,
        keras=keras.__version__, jax=jax.__version__, device=str(jax.devices()[0]),
    ))


def persist(model, state, output, state_file, improved=False):
    # Publish the new cursor only after its optimizer/model snapshot is complete.
    previous = {state[key]["name"] for key in ("last", "best") if state.get(key)}
    token = f"state-{state['epoch']:02d}-{state['updates']:07d}.keras"
    state["last"] = save_model(model, output, token)
    if improved:
        state["best"] = state["last"]
    atomic_json(state_file, state)
    live = {state[key]["name"] for key in ("last", "best") if state.get(key)}
    for name in previous - live:
        (output / name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserved", nargs="+", required=True)
    parser.add_argument("--cases", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--development", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--checkpoint-updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--cpu", action="store_true", help="Local functional test")
    args = parser.parse_args()
    if min(args.cases, args.epochs, args.batch, args.development, args.min_epochs,
           args.patience, args.checkpoint_updates) < 1 or args.min_epochs > args.epochs:
        parser.error("Invalid training sizes or checkpoint schedule")
    train(args)


if __name__ == "__main__":
    main()
