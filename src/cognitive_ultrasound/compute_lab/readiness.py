"""Dependency installation receipt and mandatory synchronous launch preflight."""

import argparse
import importlib.metadata as metadata
import json
import os
import subprocess
import sys

from ..config import ROOT
from ..provenance import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["write", "check"])
    args = p.parse_args()
    requirement = ROOT / "requirements/compute-torch.txt"
    receipt = ROOT / ".cache/compute-ready.json"
    packages = {
        name: metadata.version(name)
        for name in ("torch", "h5py", "scipy", "scikit-image", "PyYAML", "psutil", "matplotlib")
    }
    for line in requirement.read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.strip().split("==")
            if packages[name] != version:
                raise RuntimeError(f"{name}: expected {version}, found {packages[name]}")
    value = dict(python=sys.executable, requirements=sha256(requirement), packages=packages)
    # The Torch env intentionally does not install the full JAX application.
    # Do not inject src/*.egg-info into pip's inventory via worker PYTHONPATH.
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    if args.action == "write":
        receipt.parent.mkdir(exist_ok=True)
        receipt.write_text(json.dumps(value, indent=2))
    elif not receipt.exists() or json.loads(receipt.read_text()) != value:
        raise RuntimeError(
            "Setup missing/stale: run setup_compute_lab.sh successfully before start"
        )


if __name__ == "__main__":
    main()
