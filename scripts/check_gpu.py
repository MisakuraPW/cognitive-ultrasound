"""Check the preinstalled CUDA stack with actual JAX and TensorFlow GPU convolutions.

No installers, environment changes or downloads. Backends run in separate processes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROBES = {
    "jax": """
import jax
import jax.numpy as jnp
assert jax.default_backend() == 'gpu', jax.devices()
x = jnp.ones((1, 16, 16, 1), dtype=jnp.float32)
w = jnp.ones((3, 3, 1, 2), dtype=jnp.float32)
f = jax.jit(lambda a, b: jax.lax.conv_general_dilated(
    a, b, (1, 1), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC')))
y = f(x, w).block_until_ready()
assert bool(jnp.all(jnp.isfinite(y))) and y.shape == (1, 14, 14, 2)
print('JAX', jax.__version__, 'GPU convolution passed', jax.devices())
""",
    "tensorflow": """
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert gpus, 'TensorFlow GPU unavailable'
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)
with tf.device('/GPU:0'):
    x = tf.ones((1, 16, 16, 1))
    w = tf.Variable(tf.ones((3, 3, 1, 2)))
    with tf.GradientTape() as tape:
        y = tf.nn.conv2d(x, w, strides=1, padding='VALID')
        loss = tf.reduce_sum(y)
    grad = tape.gradient(loss, w)
    assert bool(tf.reduce_all(tf.math.is_finite(grad)).numpy())
assert 'GPU' in y.device and tuple(y.shape) == (1, 14, 14, 2)
print('TensorFlow', tf.__version__, 'GPU convolution/gradient passed', y.device)
print('Build info:', tf.sysconfig.get_build_info())
""",
}


def capture(args, env=None, timeout=180):
    try:
        result = subprocess.run(args, text=True, capture_output=True, env=env, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/root/autodl-tmp/outputs_casl/preparation/gpu.json")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run on the Linux GPU instance; this script does not connect remotely")
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", TF_FORCE_GPU_ALLOW_GROWTH="true")
    result = {
        "python": sys.executable,
        "tools": {tool: shutil.which(tool) for tool in ("nvcc", "ptxas", "nvlink")},
        "nvidia_smi": capture(["nvidia-smi"], timeout=30),
        "nvcc": capture(["nvcc", "--version"], timeout=30),
        "library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "backends": {},
    }
    for backend, code in PROBES.items():
        print(f"Checking {backend} in its own process...", flush=True)
        result["backends"][backend] = capture(
            [sys.executable, "-c", code], env=dict(env, KERAS_BACKEND=backend)
        )
    result["passed"] = all(v["returncode"] == 0 for v in result["backends"].values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
