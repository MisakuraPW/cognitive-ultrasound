"""Lossless ephemeral IPC; durable scientific checkpoints still use common.atomic_npz."""

from pathlib import Path

import numpy as np


def write_exchange(file, arrays, mode="compressed"):
    if mode not in ("compressed", "uncompressed", "tmpfs"):
        raise ValueError("Unknown EF IPC mode: " + mode)
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        (np.savez_compressed if mode == "compressed" else np.savez)(stream, **arrays)
        stream.flush()
        # Old compressed path retains its durability operation for fair baseline timing.
        if mode == "compressed":
            import os

            os.fsync(stream.fileno())
    temporary.replace(file)
