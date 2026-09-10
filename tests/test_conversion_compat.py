import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cognitive_ultrasound.conversion_compat import compatible_processor, sha256_file, split_digest


class Base:
    def __init__(self, output, splits):
        self.path_out_h5 = output
        self.splits = splits

    def get_split(self, name, sequence):
        raise AssertionError("Rejection mismatch")

    def __call__(self, source):
        pixels = np.arange(12).reshape(3, 4)
        split = self.get_split(source.stem + ".hdf5", pixels)
        folder = self.path_out_h5 / split
        folder.mkdir(parents=True, exist_ok=True)
        with h5py.File(folder / (source.stem + ".hdf5"), "w") as handle:
            handle["pixels"] = pixels
        return pixels


def setup(tmp_path):
    source = tmp_path / "a.avi"
    source.write_bytes(b"verified-original")
    splits = {"train": ["a.hdf5"], "val": [], "test": []}
    registry = {
        "version": "test",
        "splits_sha256": split_digest(splits),
        "cases": {source.name: {"sha256": sha256_file(source), "split": "train", "reason": "test"}},
    }
    return source, splits, registry


def test_hash_bound_override_preserves_pixels_and_records_provenance(tmp_path):
    source, splits, registry = setup(tmp_path)
    proc = compatible_processor(Base, registry)(tmp_path / "out", splits)
    expected = proc(source)
    with h5py.File(tmp_path / "out/train/a.hdf5") as handle:
        np.testing.assert_array_equal(handle["pixels"][:], expected)
        assert json.loads(handle.attrs["casl_split_compatibility"])["sha256"] == sha256_file(source)


@pytest.mark.parametrize("change", ["hash", "split", "manifest", "unknown"])
def test_compatibility_does_not_bypass_unapproved_inputs(tmp_path, change):
    source, splits, registry = setup(tmp_path)
    if change == "hash":
        source.write_bytes(b"changed")
    elif change == "split":
        registry["cases"][source.name]["split"] = "test"
    elif change == "manifest":
        splits["val"] = ["b.hdf5"]
    else:
        registry["cases"] = {}
    with pytest.raises((ValueError, AssertionError)):
        compatible_processor(Base, registry)(tmp_path / "out", splits)(source)
    assert not (tmp_path / "out").exists()


def test_other_assertion_is_not_suppressed(tmp_path):
    source, splits, registry = setup(tmp_path)

    class Other(Base):
        def get_split(self, name, sequence):
            raise AssertionError("other bug")

    with pytest.raises(AssertionError, match="other bug"):
        compatible_processor(Other, registry)(tmp_path / "out", splits)(source)


def test_archive_verification_rejects_changed_bytes(tmp_path):
    import hashlib
    import io
    import runpy
    import tarfile

    verify = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/build_data_tar.py"))[
        "verify_archive"
    ]
    name = "CASL-EchoNet-polar/train/a.hdf5"
    expected = {name: {"bytes": 4, "sha256": hashlib.sha256(b"data").hexdigest()}}
    for contents, success in ((b"data", True), (b"oops", False)):
        archive = tmp_path / "data.tar"
        with tarfile.open(archive, "w") as tar:
            for n, b in [
                (name, contents),
                ("CASL-EchoNet-polar/transfer_manifest.json", json.dumps(expected).encode()),
            ]:
                info = tarfile.TarInfo(n)
                info.size = len(b)
                tar.addfile(info, io.BytesIO(b))
        if success:
            assert verify(archive, expected) == hashlib.sha256(archive.read_bytes()).hexdigest()
        else:
            with pytest.raises(ValueError, match="checksum"):
                verify(archive, expected)
