"""Package audited CASL data and verify every archived byte before permitting cleanup."""

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cognitive_ultrasound.conversion import atomic_json, conversion_lock  # noqa: E402


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def verify_archive(archive, expected):
    """Read the tar without extracting; compare names, sizes, hashes and manifest."""
    observed = {}
    embedded = None
    with Path(archive).open("rb") as raw:
        reader = HashReader(raw)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or member.name in observed:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                content = tar.extractfile(member)
                if member.name == "CASL-EchoNet-polar/transfer_manifest.json":
                    if embedded is not None:
                        raise ValueError("Duplicate transfer manifest")
                    embedded = json.load(content)
                    continue
                if member.name not in expected:
                    raise ValueError(f"Unlisted member: {member.name}")
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: content.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                observed[member.name] = {"bytes": size, "sha256": digest.hexdigest()}
                if observed[member.name] != expected[member.name]:
                    raise ValueError(f"Archive checksum mismatch: {member.name}")
                if len(observed) % 500 == 0:
                    print(f"Verified archive {len(observed)}/{len(expected)} files", flush=True)
        # Include tar end padding in the full-archive checksum.
        while reader.read(1024 * 1024):
            pass
        archive_sha = reader.digest.hexdigest()
    if observed != expected or embedded != expected:
        raise ValueError("Missing archive files or incorrect embedded manifest")
    return archive_sha


def build(source, archive):
    source, archive = Path(source).resolve(), Path(archive).resolve()
    if source.name != "CASL-EchoNet-polar" or source == archive or source in archive.parents:
        raise ValueError("Archive must be outside the CASL-EchoNet-polar source directory")
    if archive.exists() or archive.with_suffix(archive.suffix + ".partial").exists():
        raise FileExistsError("Refusing to overwrite existing archive or partial archive")
    state = json.loads((ROOT / "results/local_conversion/status.json").read_text())
    metadata = json.loads((source / "conversion_manifest.json").read_text())
    audit_path = ROOT / "results/local_conversion/dataset_statistics.json"
    audit = json.loads(audit_path.read_text())
    if state["status"] != "completed" or metadata["status"] != "completed" or not audit["complete"]:
        raise ValueError("Conversion and full audit must be completed before packaging")
    if Path(state["output"]).resolve() != source:
        raise ValueError("Audited source path differs")
    if json.loads((source / "conversion_failures.json").read_text()):
        raise ValueError("Unresolved conversion failures")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with conversion_lock(source):
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Source contains symlinks")
        files = sorted(
            p for s in ("train", "val", "test", "rejected") for p in (source / s).glob("*.hdf5")
        )
        if len(files) != len(metadata["identity"]["sources"]):
            raise ValueError("Source file count mismatch")
        files += [
            source / name
            for name in (
                "split.yaml",
                "conversion_manifest.json",
                "conversion_compatibility_report.json",
            )
        ]
        before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        if shutil.disk_usage(archive.parent).free < sum(n for n, _ in before.values()) + 2**30:
            raise OSError("Insufficient free space for tar plus safety margin")
        entries = [(p, "CASL-EchoNet-polar/" + p.relative_to(source).as_posix()) for p in files]
        entries += [
            (audit_path, "CASL-EchoNet-polar/dataset_statistics.json"),
            (audit_path.with_suffix(".md"), "CASL-EchoNet-polar/dataset_statistics.md"),
        ]
        partial = archive.with_suffix(archive.suffix + ".partial")
        expected = {}
        with tarfile.open(partial, "w") as tar:
            for i, (path, name) in enumerate(entries, 1):
                info = tar.gettarinfo(str(path), arcname=name)
                with path.open("rb") as stream:
                    reader = HashReader(stream)
                    tar.addfile(info, reader)
                    expected[name] = {"bytes": info.size, "sha256": reader.digest.hexdigest()}
                if i % 500 == 0:
                    print(f"Archived {i}/{len(entries)} files", flush=True)
            data = (json.dumps(expected, indent=2) + "\n").encode()
            info = tarfile.TarInfo("CASL-EchoNet-polar/transfer_manifest.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if any((p.stat().st_size, p.stat().st_mtime_ns) != stat for p, stat in before.items()):
            raise RuntimeError("Source changed during packaging")
        digest = verify_archive(partial, expected)
        partial.replace(archive)
        archive.with_suffix(archive.suffix + ".sha256").write_text(
            f"{digest}  {archive.name}\n", encoding="ascii"
        )
        report = {
            "status": "verified",
            "source": str(source),
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_mtime_ns": archive.stat().st_mtime_ns,
            "archive_sha256": digest,
            "files": len(entries),
            "hdf5_files": len(metadata["identity"]["sources"]),
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "source_deleted": False,
        }
        # Preserve all source hashes outside the source folder before any later deletion.
        atomic_json(archive.with_suffix(".files.json"), expected)
        atomic_json(archive.with_suffix(".verification.json"), report)
        atomic_json(ROOT / "results/local_conversion/package_status.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.archive)
