"""Checksummed atomic diagnostic units and pickle-free state snapshots."""

import json
from pathlib import Path

import numpy as np

from ..preparation.common import atomic_json, atomic_npz, digest, read_json
from ..provenance import sha256


def save_tree(file, tree):
    arrays = {}

    def encode(x):
        if isinstance(x, np.ndarray) or (hasattr(x, "shape") and hasattr(x, "dtype")):
            key = f"array{len(arrays)}"
            v = np.asarray(x)
            if v.dtype.kind not in "biuf" or not np.isfinite(v).all():
                raise ValueError("State arrays must be finite and never pickle objects")
            arrays[key] = v
            return {"__array__": key}
        if isinstance(x, dict):
            return {k: encode(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [encode(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x

    meta = encode(tree)
    arrays["metadata"] = np.frombuffer(json.dumps(meta, allow_nan=False).encode(), np.uint8)
    atomic_npz(file, **arrays)


def load_tree(file):
    with np.load(file, allow_pickle=False) as z:

        def decode(x):
            if isinstance(x, dict):
                if set(x) == {"__array__"}:
                    return z[x["__array__"]].copy()
                return {k: decode(v) for k, v in x.items()}
            if isinstance(x, list):
                return [decode(v) for v in x]
            return x

        return decode(json.loads(z["metadata"].tobytes()))


def committed(directory, identity):
    directory = Path(directory)
    p = directory / "complete.json"
    if not p.exists():
        return None
    x = read_json(p)
    if x["identity"] != digest(identity):
        raise ValueError("Diagnostic unit identity changed")
    if x.get("result_digest") != digest(x["result"]):
        raise ValueError("Committed result payload corrupted")
    for name, hash_ in x["files"].items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory.resolve()) or sha256(file) != hash_:
            raise ValueError("Committed diagnostic file was changed/corrupted")
    return x["result"]


def commit(directory, identity, result, files=()):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        if not (directory / name).resolve().is_relative_to(directory.resolve()):
            raise ValueError("Outside unit")
    atomic_json(
        directory / "complete.json",
        dict(
            identity=digest(identity),
            result=result,
            result_digest=digest(result),
            files={name: sha256(directory / name) for name in files},
        ),
    )
    return result


def tree_digest(tree):
    import hashlib

    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(k.encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(str(len(x)).encode())
            for v in x:
                visit(v)
        else:
            h.update(json.dumps(x, allow_nan=False, sort_keys=True).encode())

    visit(tree)
    return h.hexdigest()


def compare_trees(a, b):
    """Only deterministic state; timing metadata is intentionally omitted by caller."""
    if isinstance(a, dict):
        if set(a) != set(b):
            raise AssertionError("Different state fields")
        for k in a:
            compare_trees(a[k], b[k])
    elif isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError("Different history length")
        for x, y in zip(a, b):
            compare_trees(x, y)
    elif isinstance(a, np.ndarray):
        if a.dtype.kind in "biu":
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, atol=2e-4, rtol=2e-4)
    else:
        if a != b:
            raise AssertionError(f"State mismatch: {a!r} vs {b!r}")
