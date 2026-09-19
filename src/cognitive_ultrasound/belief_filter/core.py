"""Equations 4.3--4.19 with explicit, documented engineering assumptions.

TensorFlow only; importing the existing CASL package does not import this module.
The acquisition oracle is the only inference object that can access a full target.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d
from tensorflow import keras


def residual(x, width):
    skip = x if x.shape[-1] == width else keras.layers.Conv2D(width, 1)(x)
    y = keras.layers.Conv2D(width, 3, padding="same", activation="swish")(x)
    y = keras.layers.Conv2D(width, 3, padding="same")(y)
    return keras.layers.Activation("swish")(keras.layers.Add()([skip, y]))


class Models:
    def __init__(self, cfg):
        channels, width = cfg["latent_channels"], cfg["width"]
        image = keras.Input((112, 112, 1))
        x = keras.layers.Conv2D(width, 5, strides=2, padding="same", activation="swish")(image)
        x = keras.layers.Conv2D(width * 2, 3, strides=2, padding="same")(x)
        z = keras.layers.Conv2D(channels, 1)(residual(x, width * 2))
        self.encoder = keras.Model(image, z, name="encoder")
        latent = keras.Input((28, 28, channels))
        x = residual(latent, width * 2)
        for w in (width, width):
            x = keras.layers.UpSampling2D(interpolation="bilinear")(x)
            x = keras.layers.Conv2D(w, 3, padding="same", activation="swish")(x)
        self.decoder = keras.Model(
            latent,
            keras.layers.Conv2D(1, 3, padding="same", activation="tanh")(x),
            name="decoder",
        )

        old, time_map = keras.Input((28, 28, channels)), keras.Input((28, 28, 1))
        x = residual(keras.layers.Concatenate()([old, time_map]), width)
        delta = keras.layers.Conv2D(channels, 3, padding="same", activation="tanh")(x)
        predicted = keras.layers.Add()([old, keras.layers.Rescaling(cfg["delta_limit"])(delta)])
        self.prior = keras.Model([old, time_map], predicted, name="prior")

        # zprior, zdc, zdc-zprior, mask, signed residual, abs residual, history.
        features = keras.Input((28, 28, channels * 3 + 4))
        skip = residual(features, width)
        mid = residual(keras.layers.Conv2D(width * 2, 3, 2, padding="same")(skip), width * 2)
        x = residual(keras.layers.Conv2D(width * 4, 3, 2, padding="same")(mid), width * 4)
        x = keras.layers.Concatenate()([keras.layers.UpSampling2D()(x), mid])
        x = residual(x, width * 2)
        x = keras.layers.Concatenate()([keras.layers.UpSampling2D()(x), skip])
        x = residual(x, width)
        heads = [
            keras.layers.Conv2D(channels, 1, activation="tanh", name="delta")(x),
            keras.layers.Conv2D(channels, 1, activation="sigmoid", name="gate")(x),
            keras.layers.Conv2D(1, 1, name="log_variance")(x),
        ]
        self.filter = keras.Model(features, heads, name="belief_filter")
        self.cfg = cfg

    def all(self):
        return {name: getattr(self, name) for name in ("encoder", "decoder", "prior", "filter")}

    def configure_stage(self, stage):
        active = {"codec": {"encoder", "decoder"}, "prior": {"prior"}, "filter": {"filter"}}[stage]
        for name, model in self.all().items():
            model.trainable = name in active
        return [v for m in self.all().values() for v in m.trainable_variables]

    def predict(self, z, frame):
        # The thesis does not specify time embedding: bounded scalar time is our choice.
        time_map = tf.ones_like(z[..., :1]) * (frame / (frame + 100.0))
        return self.prior([z, time_map], training=False)

    def update(self, z, observation, mask):
        before = self.decoder(z, training=False)
        projected = project(before, observation, mask)
        zdc = self.encoder(projected, training=False)
        error = mask * (observation - before)
        # History is the current-frame acquired mask; cross-frame history is soft state only.
        # Exact 4x downsampling with a gradient, unlike TensorFlow ResizeArea.
        small = [
            tf.nn.avg_pool2d(tf.convert_to_tensor(v), ksize=4, strides=4, padding="VALID")
            for v in (mask, error, tf.abs(error), mask)
        ]
        features = tf.concat([z, zdc, zdc - z, *small], axis=-1)
        delta, gate, logvar = self.filter(features)
        znew = z + gate * (zdc - z) + self.cfg["delta_limit"] * delta
        raw = self.decoder(znew, training=False)
        logvar = tf.clip_by_value(logvar, -6.0, 4.0)
        uncertainty = tf.image.resize(tf.nn.softplus(logvar), (112, 112))
        return znew, raw, project(raw, observation, mask), uncertainty, logvar, projected


def project(prediction, observation, mask):
    return tf.where(mask > 0, observation, prediction)


class ArrayOracle:
    """Simulation boundary: return only pixels explicitly requested by the policy."""

    def __init__(self, target):
        self._target = np.asarray(target, dtype=np.float32)
        if self._target.shape != (112, 112, 1) or not np.isfinite(self._target).all():
            raise ValueError("Oracle expects one finite 112x112x1 frame")

    def acquire(self, lines):
        result = np.zeros_like(self._target)
        result[:, lines, :] = self._target[:, lines, :]
        return result


@dataclass
class Memory:
    latent: object
    image: object
    uncertainty: object
    values: np.ndarray


def initial_memory(models):
    z = tf.zeros((1, 28, 28, models.cfg["latent_channels"]))
    return Memory(z, models.decoder(z), tf.ones((1, 112, 112, 1)), np.zeros(112))


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    spread = np.ptp(values)
    return (values - values.min()) / spread if spread > 1e-12 else np.zeros_like(values)


def line_scores(uncertainty, change, observed_residual, values, mask, cfg, first):
    def columns(tensor):
        return np.asarray(tensor).mean(axis=(0, 1, 3))

    u, c = columns(uncertainty), columns(change)
    # Residual on unmeasured pixels is unavailable. Spread ONLY measured residual
    # sideways as a proxy, an explicit departure where the thesis omits details.
    r = gaussian_filter1d(columns(observed_residual), cfg["residual_sigma"])
    h = gaussian_filter1d(columns(mask), cfg["redundancy_sigma"])
    w = cfg["score_weights"]
    return (
        w["uncertainty"] * normalize(u)
        + w["change"] * normalize(c)
        + w["value"] * normalize(values)
        + (0 if first else w["residual"] * normalize(r) - w["history"] * normalize(h))
    )


def select_lines(scores, acquired, count, cfg):
    scores = np.asarray(scores, dtype=np.float64)
    excluded = np.asarray(acquired, dtype=bool).copy()
    if scores.shape != (112,) or excluded.shape != (112,) or not np.isfinite(scores).all():
        raise ValueError("Invalid line scores/mask")
    if not 1 <= count <= np.count_nonzero(~excluded):
        raise ValueError("Group exceeds remaining distinct lines")
    penalty, selected = np.zeros(112), []
    positions = np.arange(112)
    for _ in range(count):
        candidates = np.where(excluded, -np.inf, scores - cfg["redundancy"] * penalty)
        # Deterministic tie: closest to center, then smallest index.
        best = np.flatnonzero(np.isclose(candidates, candidates.max(), rtol=0, atol=1e-12))
        index = int(best[np.argmin(np.abs(best - 55.5))])
        selected.append(index)
        excluded[index] = True
        penalty += np.exp(-0.5 * ((positions - index) / cfg["redundancy_sigma"]) ** 2)
    return selected


def step(models, oracle, memory, frame, cfg, policy="greedy"):
    """Causal within-frame 10+4 acquisition; no ground truth argument to selection."""
    if policy not in ("greedy", "uniform"):
        raise ValueError("Unknown acquisition policy")
    z = models.predict(memory.latent, frame)
    image = models.decoder(z)
    change = tf.abs(image - memory.image)
    uncertainty = memory.uncertainty
    mask = np.zeros((1, 112, 112, 1), dtype=np.float32)
    observation = np.zeros_like(mask)
    error = np.zeros_like(mask)
    acquired = np.zeros(112, dtype=bool)
    values = memory.values.copy()
    groups, updates = [], []
    fixed = np.linspace(0, 111, sum(cfg["groups"]), dtype=int)
    offset = 0
    for group, count in enumerate(cfg["groups"]):
        scores = line_scores(uncertainty, change, error, values, mask, cfg, first=group == 0)
        lines = (
            select_lines(scores, acquired, count, cfg)
            if policy == "greedy"
            else fixed[offset : offset + count].tolist()
        )
        offset += count
        acquired[lines] = True
        observation += oracle.acquire(lines)[None]
        mask[:, :, lines, :] = 1
        before = models.decoder(z)
        error = mask * np.abs(observation - np.asarray(before))
        measured = error.mean(axis=(0, 1, 3))
        rate = cfg["value_decay"]
        values[lines] = rate * values[lines] + (1 - rate) * measured[lines]
        z, raw, image, uncertainty, logvar, projected = models.update(z, observation, mask)
        updates.append((z, raw, logvar, projected, observation.copy(), mask.copy()))
        groups.append(
            {
                "lines": lines,
                "scores": scores.copy(),
                "reconstruction": np.asarray(image)[0].copy(),
                "uncertainty": np.asarray(uncertainty)[0].copy(),
            }
        )
    next_memory = Memory(
        tf.stop_gradient(z), tf.stop_gradient(image), tf.stop_gradient(uncertainty), values
    )
    state = {
        "reconstruction": image,
        "uncertainty": uncertainty,
        "mask": mask,
        "observation": observation,
        "groups": groups,
        "updates": updates,
    }
    return next_memory, state


def reconstruction_loss(target, prediction, ssim_weight):
    mae = tf.reduce_mean(tf.abs(target - prediction))
    ssim = tf.reduce_mean(tf.image.ssim((target + 1) / 2, (prediction + 1) / 2, max_val=1.0))
    return mae + ssim_weight * (1 - ssim)


def filter_loss(models, target, updates, cfg):
    target_z = tf.stop_gradient(models.encoder(target, training=False))
    terms = {name: [] for name in ("rec", "obs", "lat", "unc", "proj")}
    for z, raw, logvar, projected, observation, mask in updates:
        terms["rec"].append(reconstruction_loss(target, raw, cfg["ssim_weight"]))
        terms["obs"].append(
            tf.reduce_sum(tf.abs(raw - observation) * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
        )
        terms["lat"].append(tf.reduce_mean(tf.abs(z - target_z)))
        logvar = tf.image.resize(logvar, (112, 112))
        terms["unc"].append(tf.reduce_mean(tf.exp(-logvar) * tf.square(raw - target) + logvar))
        terms["proj"].append(tf.reduce_mean(tf.abs(raw - projected)))
    terms = {name: tf.add_n(items) / len(items) for name, items in terms.items()}
    total = tf.add_n([cfg["loss_weights"][name] * value for name, value in terms.items()])
    return total, terms
