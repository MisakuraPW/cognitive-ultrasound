"""Create a self-contained source overlay, including uncommitted preparation/BF code."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(file):
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="cognitive-ultrasound-preparation-20260919-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prefix = args.name
    if not prefix or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in prefix
    ):
        parser.error("--name must be a simple directory name")
    files = []
    for directory in ("src", "scripts", "configs", "docs", "tests", "requirements"):
        files.extend(
            f
            for f in (root / directory).rglob("*")
            if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".tmp")
        )
    files.extend(
        f
        for f in root.iterdir()
        if f.is_file()
        and (f.name in ("README.md", "pyproject.toml", ".gitmodules") or "requirement" in f.name)
    )
    files.append(root / "reports/preparation_local_check.md")
    closure_report = root / "reports/preparation_closure_local_check.md"
    if closure_report.exists():
        files.append(closure_report)
    for suffix in ("md", "json"):
        acceleration_report = root / ("reports/hardware_autotune_20260919." + suffix)
        if acceleration_report.exists():
            files.append(acceleration_report)
    hashes = {f.relative_to(root).as_posix(): sha(f) for f in sorted(files)}
    archive = destination / (prefix + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(files):
            tar.add(f, arcname=prefix + "/" + f.relative_to(root).as_posix(), recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            relative = member.name[len(prefix) + 1 :]
            actual = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            if actual != hashes[relative]:
                raise RuntimeError("Source archive mismatch: " + relative)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        sha(archive) + "  " + archive.name + "\n", encoding="ascii", newline="\n"
    )
    (destination / (prefix + ".files.json")).write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(archive=str(archive), bytes=archive.stat().st_size, files=len(files)),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
