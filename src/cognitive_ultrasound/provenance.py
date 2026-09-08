import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT


def command(args, cwd=ROOT):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
        return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def sha256(file):
    digest = hashlib.sha256()
    with open(file, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(file, data):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def environment():
    versions = {}
    for name in (
        "numpy",
        "keras",
        "jax",
        "jaxlib",
        "tensorflow",
        "tf2jax",
        "torch",
        "torchvision",
        "h5py",
        "lpips",
        "scikit-image",
        "zea",
        "ulsa",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": versions,
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ]
        ),
        "git": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "casl": command(["git", "rev-parse", "HEAD"], ROOT / "vendor/casl"),
        "zea": command(["git", "rev-parse", "HEAD"], ROOT / "vendor/casl/zea"),
    }


def environment_report(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = environment()
    write_json(output.with_suffix(".json"), info)
    output.write_text(
        "# 实际环境报告\n\n此报告描述执行机器，不代表云端 RTX 4090 环境。\n\n```json\n"
        + json.dumps(info, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )
    return info
