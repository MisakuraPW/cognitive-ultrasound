"""Opt-in graph execution for unchanged codec/prior and fixed-mask filter objectives.

Adaptive greedy selection remains eager: its Python policy is not silently replaced.
"""

import numpy as np


def make_kernels(models, optimizer, stage, cfg):
    import tensorflow as tf

    from .core import filter_loss, reconstruction_loss

    if stage == "filter" and cfg.get("training_policy", "greedy") != "uniform":
        raise ValueError("Graph filter training requires the explicitly fixed uniform policy")
    frames = cfg["clip_frames"]
    variables = tuple(v for model in models.all().values() for v in model.trainable_variables)
    masks = []
    indices = np.linspace(0, 111, sum(cfg["groups"]), dtype=int)
    for end in np.cumsum(cfg["groups"]):
        mask = np.zeros((1, 112, 112, 1), np.float32)
        mask[:, :, indices[:end], :] = 1
        masks.append(tf.constant(mask))

    def objective(clip, start):
        start = tf.cast(start, clip.dtype)
        if stage == "codec":
            prediction = models.decoder(models.encoder(clip, training=True), training=True)
            return reconstruction_loss(clip, prediction, cfg["ssim_weight"])
        if stage == "prior":
            latent = tf.stop_gradient(models.encoder(clip, training=False))
            terms = []
            for frame in range(1, frames):
                predicted = models.predict(latent[frame - 1 : frame], start + frame)
                terms.append(
                    tf.reduce_mean(tf.abs(predicted - latent[frame : frame + 1]))
                    + reconstruction_loss(
                        clip[frame : frame + 1], models.decoder(predicted), cfg["ssim_weight"]
                    )
                )
            return tf.add_n(terms) / len(terms)
        memory = tf.zeros((1, 28, 28, cfg["latent_channels"]), dtype=clip.dtype)
        terms = []
        for frame in range(frames):
            if (
                cfg.get("training_reset_interval", 0)
                and frame % cfg["training_reset_interval"] == 0
            ):
                memory = tf.zeros_like(memory)
            truth = clip[frame : frame + 1]
            z = models.predict(memory, start + frame)
            updates = []
            for mask in masks:
                observation = tf.where(mask > 0, truth, tf.zeros_like(truth))
                z, raw, _, _, logvar, projected = models.update(z, observation, mask)
                updates.append((z, raw, logvar, projected, observation, mask))
            value, _ = filter_loss(models, truth, updates, cfg)
            terms.append(value)
            memory = tf.stop_gradient(z)  # preserve original truncated cross-frame gradient
        return tf.add_n(terms) / len(terms)

    def update(clip, start):
        with tf.GradientTape() as tape:
            loss = objective(clip, start)
        tf.debugging.assert_all_finite(loss, "Nonfinite loss")
        gradients = tape.gradient(loss, variables)
        if any(g is None for g in gradients):
            raise RuntimeError("Missing gradient")
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite gradient")
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    signature = [tf.TensorSpec((frames, 112, 112, 1), tf.float32), tf.TensorSpec((), tf.int32)]
    # TensorFlow graphs only: no XLA fast-math, precision or batch-size changes.
    return (
        tf.function(update, input_signature=signature, autograph=False, jit_compile=False),
        tf.function(objective, input_signature=signature, autograph=False, jit_compile=False),
    )
