"""Explicit, hash-bound split compatibility without changing image processing."""

import hashlib
import json
from pathlib import Path

import h5py


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_digest(splits):
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


def compatible_processor(base, registry):
    class Processor(base):
        def __call__(self, avi_file):
            self._compat_entry = None
            self._compat_used = False
            source = Path(avi_file)
            entry = registry["cases"].get(source.name)
            if entry is not None:
                if split_digest(self.splits) != registry["splits_sha256"]:
                    raise ValueError("Compatibility split digest mismatch")
                assigned = next(
                    (s for s, names in self.splits.items() if source.stem + ".hdf5" in names),
                    "rejected",
                )
                if assigned != entry["split"] or sha256_file(source) != entry["sha256"]:
                    raise ValueError("Compatibility source hash or split mismatch")
                self._compat_entry = (source.stem + ".hdf5", entry)
            result = super().__call__(avi_file)
            if self._compat_used:
                file = self.path_out_h5 / entry["split"] / (source.stem + ".hdf5")
                with h5py.File(file, "a") as handle:
                    handle.attrs["casl_split_compatibility"] = json.dumps(
                        {"version": registry["version"], "source": source.name, **entry},
                        sort_keys=True,
                    )
            return result

        def get_split(self, hdf5_file, sequence):
            try:
                return super().get_split(hdf5_file, sequence)
            except AssertionError as error:
                if str(error) != "Rejection mismatch" or self._compat_entry is None:
                    raise
                name, entry = self._compat_entry
                if name != hdf5_file:
                    raise ValueError("Compatibility case mismatch") from error
                self._compat_used = True
                return entry["split"]

    return Processor
