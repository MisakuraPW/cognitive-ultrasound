"""Fetch pinned upstreams over HTTPS. No git shell helpers/SSH credentials required."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    (
        "vendor/casl",
        "https://github.com/tue-bmd/casl.git",
        "5f57aba668eb34054b8ced817c666b7d18a51ca2",
    ),
    (
        "vendor/casl/zea",
        "https://github.com/tue-bmd/zea.git",
        "192c0bbd4e89061c38048048673beadcb8723a82",
    ),
]


def main():
    for relative, url, revision in SOURCES:
        folder = ROOT / relative
        cloned = False
        if not (folder / ".git").exists():
            subprocess.run(["git", "clone", "--no-checkout", url, str(folder)], check=True)
            cloned = True
        current = subprocess.check_output(
            ["git", "-C", str(folder), "rev-parse", "HEAD"], text=True
        ).strip()
        if not cloned:
            dirty = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(folder),
                    "status",
                    "--porcelain",
                    "--ignore-submodules",
                    "--untracked-files=no",
                ],
                text=True,
            )
            if dirty:
                raise RuntimeError(f"Refusing to replace local changes: {folder}")
        if current != revision:
            subprocess.run(["git", "-C", str(folder), "fetch", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(folder), "checkout", "--detach", revision], check=True)
    print("Pinned CASL and zea are ready. See README for environment setup.")


if __name__ == "__main__":
    main()
