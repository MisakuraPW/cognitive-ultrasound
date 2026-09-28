"""Same fixed explicit-noise frame as the Torch root-cause diagnosis."""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    from cognitive_ultrasound.official import activate

    activate("jax")
    import jax
    import jax.numpy as jnp

    from cognitive_ultrasound.torch_casl.reference import load_model, matched_frame

    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU required for this comparison")
    cfg = json.loads((args.root / "config.json").read_text())
    manifest = json.loads((args.root / "manifest.json").read_text())
    name = manifest["cohorts"]["debug"][0]
    source = args.root / ".cache/debug/b14/42" / Path(name).with_suffix(".h5").name
    with h5py.File(source) as h:
        before = {k: v[()] for k, v in h["1"].items()}
        current = {k: v[()] for k, v in h["2"].items()}
    inputs = tuple(
        map(
            jnp.asarray,
            (
                np.repeat(current["resume_buffer"][None], 2, axis=0),
                before["resume_mask"][None],
                before["resume_posterior_samples"],
                current["noise"],
            ),
        )
    )
    model = load_model(cfg["checkpoint"])
    kernel = jax.jit(matched_frame(model, 14), static_argnames=("steps",))
    for _ in range(3):
        jax.block_until_ready(kernel(*inputs, steps=50))
    samples = []
    for _ in range(20):
        begin = time.perf_counter()
        values = kernel(*inputs, steps=50)
        jax.block_until_ready(values)
        samples.append(time.perf_counter() - begin)
    args.output.mkdir(parents=True, exist_ok=True)
    converted = [np.asarray(v) for v in values]
    # Align sample axes with Torch output; projected image/action/entropy already match.
    converted[0] = converted[0].transpose(0, 3, 1, 2)
    if not all(np.isfinite(v).all() for v in converted):
        raise FloatingPointError("Nonfinite JAX reference")
    np.savez(
        args.output / "frame_outputs.npz",
        **{f"output_{i}": value for i, value in enumerate(converted)},
    )
    result = dict(
        scope="Same explicit-noise input; not full video or official RNG-in-graph timing",
        case=name,
        frame=2,
        steps=50,
        precision="fp32",
        particles=2,
        budget=14,
        warmups=3,
        repeats=20,
        jax=jax.__version__,
        device=str(jax.devices()[0]),
        mean_s=float(np.mean(samples)),
        p50_s=float(np.median(samples)),
        p95_s=float(np.quantile(samples, 0.95)),
        samples_s=samples,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
