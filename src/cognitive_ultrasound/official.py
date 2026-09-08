"""Dependency boundary: pin code and choose Keras backend BEFORE any ML imports."""

import os
import sys

from .config import CASL_COMMIT, ROOT, ZEA_COMMIT
from .provenance import command


def activate(backend="jax"):
    if "keras" in sys.modules:
        import keras

        if keras.backend.backend() != backend:
            raise RuntimeError(
                "Training and inference need separate processes (Keras backend differs)"
            )
    os.environ["KERAS_BACKEND"] = backend
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("ZEA_CACHE_DIR", str(ROOT / ".cache/zea"))
    for folder, revision in (
        (ROOT / "vendor/casl", CASL_COMMIT),
        (ROOT / "vendor/casl/zea", ZEA_COMMIT),
    ):
        if command(["git", "rev-parse", "HEAD"], folder) != revision:
            raise RuntimeError(
                f"Missing or wrong upstream revision: {folder}; run scripts/bootstrap.py"
            )
        if command(["git", "diff", "HEAD", "--name-only", "--ignore-submodules"], folder):
            raise RuntimeError(f"Upstream source has local changes: {folder}")
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    # Also inherited by subprocesses such as the official converter.
    entries = [str(ROOT / "vendor/casl"), str(ROOT / "vendor/casl/zea")]
    os.environ["PYTHONPATH"] = os.pathsep.join(entries + [os.environ.get("PYTHONPATH", "")])
