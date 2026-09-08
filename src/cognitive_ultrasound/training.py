"""Train the upstream model/objective; add logging, AMP scaling and recoverable checkpoints."""

import json
import os
from types import MethodType

from .config import path
from .data import inspect_file, read_splits
from .official import activate
from .provenance import environment, sha256, write_json


def amp_train_step(self, data):
    """The upstream noise MAE step with FP32 loss and explicit Keras loss scaling.

    Upstream train_step does not call scale_loss, so merely setting mixed_float16 is unsafe.
    This optional precision variant must be reported separately from the FP32 paper baseline.
    """
    import keras
    import tensorflow as tf
    from keras import ops

    noises = keras.random.normal(shape=ops.shape(data), dtype="float32")
    times = keras.random.uniform(
        shape=[ops.shape(data)[0], 1, 1, 1], minval=self.min_t, maxval=self.max_t, dtype="float32"
    )
    noise_rates, signal_rates = self.diffusion_schedule(times)
    noisy = signal_rates * tf.cast(data, tf.float32) + noise_rates * noises
    with tf.GradientTape() as tape:
        pred_noise, pred_image = self.denoise(noisy, noise_rates, signal_rates, training=True)
        noise_loss = self.loss(noises, tf.cast(pred_noise, tf.float32))
        image_loss = self.loss(tf.cast(data, tf.float32), tf.cast(pred_image, tf.float32))
        scaled_loss = self.optimizer.scale_loss(noise_loss)
    gradients = tape.gradient(scaled_loss, self.network.trainable_weights)
    self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
    self.noise_loss_tracker.update_state(noise_loss)
    self.image_loss_tracker.update_state(image_loss)
    for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
        ema_weight.assign(self.ema_val * ema_weight + (1 - self.ema_val) * weight)
    return {m.name: m.result() for m in self.metrics}


def amp_denoise(self, noisy_images, noise_rates, signal_rates, training, network=None):
    """Keep diffusion algebra FP32 while the U-Net convolutions use mixed precision."""
    import tensorflow as tf

    pred_noises = self([noisy_images, noise_rates**2], training=training, network=network)
    pred_noises = tf.cast(pred_noises, tf.float32)
    pred_images = (
        tf.cast(noisy_images, tf.float32) - tf.cast(noise_rates, tf.float32) * pred_noises
    ) / tf.cast(signal_rates, tf.float32)
    return pred_noises, pred_images


def train(cfg, resume=False, smoke=False):
    activate("tensorflow")
    import keras
    import tensorflow as tf
    from zea.backend.tensorflow.dataloader import make_dataloader
    from zea.models.diffusion import DiffusionModel

    if cfg["precision"] not in ("float32", "mixed_float16"):
        raise ValueError("Supported training precision: float32 / mixed_float16")
    if cfg["loss"] != "mae" or cfg["temporal_window"] != 3:
        raise ValueError("Baseline training requires upstream MAE and W=3")
    for key in ("batch_size", "epochs", "steps_per_epoch", "validation_steps"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not smoke and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(
            "Training requires a GPU; use --smoke only for a small synthetic CPU test"
        )
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.mixed_precision.set_global_policy(cfg["precision"])
    keras.utils.set_random_seed(cfg["seed"])
    output = path(cfg["output"])
    if os.name == "nt" and not str(output).isascii():
        raise ValueError(
            "TensorFlow on Windows cannot reliably use this non-ASCII path. "
            "For --smoke, set --output to an ASCII temporary directory; use Linux for training."
        )
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "training_manifest.json"
    identity = {"config": cfg, "synthetic_smoke": smoke}
    if not smoke:
        identity["split_sha256"] = sha256(path(cfg["split_manifest"]))
    if metadata.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if not resume or previous["identity"] != identity:
            raise FileExistsError("Existing training run needs identical config and --resume")
    else:
        if any(output.iterdir()):
            raise FileExistsError("Training output is not empty")
        write_json(
            metadata,
            {
                "identity": identity,
                "environment": environment(),
                "resume_semantics": "Epoch-boundary weights, EMA and optimizer restore. RNG/data order restart; not bitwise identical.",
                "status": "running",
            },
        )
    if smoke:
        data = tf.random.stateless_uniform((2, 112, 112, 3), [cfg["seed"], 1], minval=-1, maxval=1)
        train_data = tf.data.Dataset.from_tensor_slices(data).batch(1).repeat()
        val_data = train_data
        steps, val_steps, epochs = 1, 1, 1
    else:
        splits = read_splits(path(cfg["split_manifest"]))
        datasets = []
        for split, key in (("train", "train_folder"), ("val", "val_folder")):
            folder = path(cfg[key])
            files = [folder / name for name in splits[split]]
            if set(p.name for p in folder.glob("*.hdf5")) != set(splits[split]):
                raise ValueError(f"Training {split} folder does not match the patient manifest")
            for file in files:
                inspect_file(file, min_frames=3)
            ds = make_dataloader(
                file_paths=[str(p) for p in files],
                batch_size=cfg["batch_size"],
                key="data/image",
                n_frames=3,
                insert_frame_axis=True,
                image_size=cfg["image_size"],
                image_range=cfg["image_range"],
                normalization_range=[-1, 1],
                shuffle=True,
                cache=False,
                drop_remainder=True,
                overlapping_blocks=True,
                resize_type="resize",
                dataset_repetitions=1,
                assert_image_range=False,
                seed=cfg["seed"],
                wrap_in_keras=False,
            )
            datasets.append(ds.repeat())
        train_data, val_data = datasets
        steps, val_steps, epochs = cfg["steps_per_epoch"], cfg["validation_steps"], cfg["epochs"]
    model = DiffusionModel(
        (112, 112, 3),
        input_range=(-1, 1),
        min_signal_rate=cfg["min_signal_rate"],
        max_signal_rate=cfg["max_signal_rate"],
        network_kwargs=cfg["network_kwargs"],
        ema_val=cfg["ema"],
        guidance="dps",
        operator="inpainting",
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    if cfg["precision"] == "mixed_float16":
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.train_step = MethodType(amp_train_step, model)
        model.denoise = MethodType(amp_denoise, model)
    model.compile(optimizer=optimizer, loss=keras.losses.MeanAbsoluteError(), jit_compile=False)
    model.sample(n_samples=1, n_steps=1, seed=keras.random.SeedGenerator(cfg["seed"]))
    optimizer.build(model.network.trainable_variables)
    epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(
        network=model.network, ema_network=model.ema_network, optimizer=optimizer, epoch=epoch_var
    )
    manager = tf.train.CheckpointManager(checkpoint, str(output / "resume"), max_to_keep=3)
    if resume:
        if not manager.latest_checkpoint:
            raise FileNotFoundError("No complete epoch checkpoint to resume")
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    initial_epoch = int(epoch_var.numpy())

    class Save(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            epoch_var.assign(epoch + 1)
            manager.save(checkpoint_number=epoch + 1)
            self.model.save_weights(output / f"epoch_{epoch + 1:04d}.weights.h5")
            self.model.save_to_preset(str(output / "hub"))
            if cfg["precision"] == "mixed_float16":
                preset_file = output / "hub/config.json"
                preset = json.loads(preset_file.read_text(encoding="utf-8"))
                preset["config"]["dtype"] = "float32"
                write_json(preset_file, preset)

    callbacks = [
        Save(),
        keras.callbacks.TensorBoard(log_dir=str(output / "logs/tensorboard")),
        keras.callbacks.CSVLogger(str(output / "logs/training.csv"), append=resume),
        keras.callbacks.TerminateOnNaN(),
    ]
    (output / "logs").mkdir(exist_ok=True)
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps,
        validation_steps=val_steps,
        callbacks=callbacks,
    )
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record.update(
        status="completed" if int(epoch_var.numpy()) == epochs else "stopped_early",
        epochs_completed=int(epoch_var.numpy()),
        history=history.history or record.get("history", {}),
    )
    write_json(metadata, record)
    return output
