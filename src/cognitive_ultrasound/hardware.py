"""Hardware identity without importing a GPU framework or trusting host CPU counts."""

import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil


def cpu_limit():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                count = min(count, max(1, quota // period))
        except (OSError, ValueError):
            pass
    return max(1, count or 1)


def snapshot():
    packages = {}
    for name in ("tensorflow", "keras", "jax", "jaxlib", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        gpu = None
    memory = psutil.virtual_memory().total
    for file in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            memory = min(memory, int(Path(file).read_text()))
        except (OSError, ValueError):
            pass
    return dict(
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_model=platform.processor(),
        cpu_limit=cpu_limit(),
        memory_limit_bytes=memory,
        gpu=gpu,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        packages=packages,
    )


def thread_candidates(count):
    return sorted({1, min(4, count), min(8, count)})
