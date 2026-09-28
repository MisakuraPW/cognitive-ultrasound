"""Separate-framework real GPU probes; runtime identity excludes fluctuating free RAM."""

import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import psutil

from ..hardware import cpu_limit, snapshot
from ..preparation.common import atomic_json, digest


def probe_framework(name):
    result = dict(python=sys.executable, python_version=platform.python_version(), framework=name)
    result["packages"] = {}
    for package in (
        "numpy",
        "keras",
        "jaxlib",
        "jax-cuda12-plugin",
        "tensorflow",
        "torch",
        "h5py",
        "scipy",
        "scikit-image",
    ):
        try:
            result["packages"][package] = version(package)
        except PackageNotFoundError:
            result["packages"][package] = None
    if name == "jax":
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_default_matmul_precision", "highest")
        if jax.default_backend() != "gpu":
            raise RuntimeError("JAX real GPU unavailable")
        f = jax.jit(
            jax.value_and_grad(
                lambda x: jnp.sum(
                    jax.lax.conv_general_dilated(
                        x,
                        jnp.ones((3, 3, 3, 8)),
                        (1, 1),
                        "SAME",
                        dimension_numbers=("NHWC", "HWIO", "NHWC"),
                    )
                    ** 2
                )
            )
        )
        val, grad = f(jnp.ones((2, 32, 32, 3)))
        jax.block_until_ready(grad)
        if not bool(jnp.isfinite(grad).all()):
            raise FloatingPointError("JAX gradient")
        result.update(
            version=jax.__version__, devices=[str(x) for x in jax.devices()], value=float(val)
        )
    elif name == "tensorflow":
        import tensorflow as tf

        tf.config.experimental.enable_tensor_float_32_execution(False)
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("TF GPU unavailable")
        with tf.device("/GPU:0"):
            x = tf.Variable(tf.ones((2, 32, 32, 3)))
            with tf.GradientTape() as tape:
                y = tf.reduce_sum(tf.nn.conv2d(x, tf.ones((3, 3, 3, 8)), 1, "SAME") ** 2)
            g = tape.gradient(y, x)
        tf.debugging.assert_all_finite(g, "TF gradient")
        if "GPU" not in g.device:
            raise RuntimeError("TensorFlow silently placed probe on CPU")
        result.update(
            version=tf.__version__, runtime=tf.sysconfig.get_build_info(), device=g.device
        )
    elif name == "torch":
        import numpy as np
        import torch

        from ..evaluation.metrics import Metrics

        result["metric_probe"] = Metrics(["ssim"])(np.zeros((112, 112)), np.zeros((112, 112)))

        if not torch.cuda.is_available():
            raise RuntimeError("Torch GPU unavailable")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        x = torch.ones((2, 3, 32, 32), device="cuda", requires_grad=True)
        y = (
            torch.nn.functional.conv2d(x, torch.ones((8, 3, 3, 3), device="cuda"), padding=1)
            .square()
            .sum()
        )
        (g,) = torch.autograd.grad(y, x)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(g).all()):
            raise FloatingPointError("Torch gradient")
        result.update(
            version=torch.__version__,
            runtime=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            device=torch.cuda.get_device_name(),
            bf16=torch.cuda.is_bf16_supported(),
        )
    result["passed"] = True
    return result


def probe(cfg, output):
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, python in cfg["pythons"].items():
        target = output / (name + ".json")
        target.unlink(missing_ok=True)  # never accept a previous successful probe after a crash
        command = [
            python,
            "-m",
            "cognitive_ultrasound.compute_lab",
            "framework",
            "--framework",
            name,
            "--output",
            str(target),
        ]
        env = dict(
            os.environ,
            PYTHONUTF8="1",
            KERAS_BACKEND="tensorflow" if name == "tensorflow" else "jax",
            NVIDIA_TF32_OVERRIDE="0",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
        )
        try:
            with (output / (name + ".log")).open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            records[name] = dict(passed=False, error=str(exc))
            continue
        records[name] = (
            json.loads(target.read_text())
            if target.exists()
            else dict(
                passed=False, returncode=completed.returncode, log=str(output / (name + ".log"))
            )
        )
    hardware = snapshot()
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        gpu = smi.stdout.strip()
    except OSError:
        gpu = None
    identity = dict(
        gpu=gpu,
        cpu=platform.processor(),
        quota=cpu_limit(),
        system=platform.platform(),
        frameworks=records,
        cuda_path=os.environ.get("LD_LIBRARY_PATH"),
        tf32=False,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    value = dict(
        identity=identity,
        fingerprint=digest(identity),
        hardware=hardware,
        available_ram=psutil.virtual_memory().available,
        timestamp=time.time(),
        frameworks=records,
        passed=all(x.get("passed", False) for x in records.values()),
    )
    atomic_json(output / "probe.json", value)
    return value
