"""Train the ViViT-style transition for the pinned 112x112 CASL model."""

import argparse
import math
import time
from pathlib import Path

import h5py
import keras
import numpy as np
import yaml
from keras import layers, ops

from .preparation.common import atomic_json
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


def lock_split(data, split_file, reserved, size, seed):
    splits = yaml.safe_load(split_file.read_text(encoding="utf-8"))
    groups = {fold: set(splits[fold]) for fold in ("train", "val", "test")}
    if any(len(groups[fold]) != len(splits[fold]) for fold in groups) or (
        len(set.union(*groups.values())) != sum(map(len, groups.values()))
    ):
        raise ValueError("Invalid CASL split")
    if not set(reserved) <= groups["val"]:
        raise ValueError("Benchmark reservations must be from validation")
    rng = np.random.default_rng(seed)
    train = rng.permutation(sorted(groups["train"])).tolist()[:size]
    dev_size = 2 if size < 32 else 16
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
    prediction = np.asarray(model(batch[:, :-1], training=False))
    truth = batch[:, -1]
    error = np.abs(prediction - truth)
    if not np.isfinite(error).all():
        raise FloatingPointError("ViViT prediction is nonfinite")
    return float(error.mean()), float(np.abs(batch[:, -2] - truth).mean())


def train(args):
    import jax

    if keras.backend.backend() != "jax" or (not args.cpu and jax.default_backend() != "gpu"):
        raise RuntimeError("ViViT training requires Keras/JAX on CUDA")
    jax.config.update("jax_default_matmul_precision", "highest")
    keras.mixed_precision.set_global_policy("float32")
    keras.utils.set_random_seed(args.seed)
    args.output.mkdir(parents=True)
    manifest = lock_split(args.data, args.split, args.reserved, args.cases, args.seed)
    atomic_json(args.output / "manifest.json", manifest)
    model = predictor()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, global_clipnorm=1.0), loss="mae")
    dev = np.stack(list(clips(
        args.data, manifest["development"], "val", np.random.default_rng(args.seed + 1),
    )))
    best, rows, count = math.inf, [], 0
    for epoch in range(args.epochs):
        tick = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        training = rng.permutation(manifest["train"]).tolist()
        model.reset_metrics()
        loss = None
        for offset in range(0, len(training), args.batch):
            batch = np.stack(list(clips(args.data, training[offset : offset + args.batch], "train", rng)))
            loss = float(model.train_on_batch(batch[:, :-1], batch[:, -1]))
            count += 1
            if not math.isfinite(loss):
                raise FloatingPointError("ViViT training loss is nonfinite")
        mae, copy_mae = validate(model, dev)
        row = dict(epoch=epoch + 1, updates=count, loss=loss, mae=mae, copy_mae=copy_mae,
                   elapsed_s=time.perf_counter() - tick)
        print(row, flush=True)
        if mae < best:
            best = mae
            temp = args.output / "saving.keras"
            model.save(temp)
            temp.replace(args.output / "best.keras")
        rows.append(row)
        atomic_json(args.output / "learning.json", rows)
    atomic_json(args.output / "complete.json", dict(
        checkpoint_sha256=sha256(args.output / "best.keras"),
        best_validation_mae=best, cases=args.cases, epochs=args.epochs, updates=count,
        cpu_functional=args.cpu, history="clean polar frames for teacher forcing", pixel_grid=112,
        context=CONTEXT, tubelet=[2, PATCH, PATCH], transformer_depth=[2, 2], heads=8,
        keras=keras.__version__, jax=jax.__version__, device=str(jax.devices()[0]),
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserved", nargs="+", required=True)
    parser.add_argument("--cases", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--cpu", action="store_true", help="Local functional test")
    args = parser.parse_args()
    if min(args.cases, args.epochs, args.batch) < 1:
        parser.error("Positive cases, epochs and batch are required")
    train(args)


if __name__ == "__main__":
    main()
