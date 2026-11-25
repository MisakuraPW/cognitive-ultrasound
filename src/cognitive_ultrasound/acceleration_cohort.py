"""Stable nested case order for the two acceleration baselines."""

import hashlib
from pathlib import Path

import yaml

from .preparation.common import digest
from .provenance import sha256

FOLDS = dict(train="train", development="val", test="test")


def lock_cohort(config_file, split_file):
    config = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    raw = yaml.safe_load(Path(split_file).read_text(encoding="utf-8"))
    cases = {}
    for role, split in FOLDS.items():
        names = raw[split]
        count = config[role] if config[role] is not None else len(names)
        if not 0 < count <= len(names):
            raise ValueError(f"Invalid cohort size for {role}")
        prefix = f"{config['seed']}:{split}:"
        ordered = sorted(names, key=lambda name: hashlib.sha256((prefix + name).encode()).digest())
        cases[role] = ordered[:count]
    verify_cases(cases, raw)
    result = dict(name=config["name"], seed=config["seed"], cases=cases,
                  frames=config["frames"], split_sha256=sha256(split_file))
    return dict(**result, identity=digest(result))


def verify_cases(cases, splits):
    used = set()
    for role, split in FOLDS.items():
        names = cases[role]
        picked = set(names)
        if not names or len(picked) != len(names) or used & picked or not picked <= set(splits[split]):
            raise ValueError(f"Invalid locked {role} cases")
        if any(n != Path(n).name or "/" in n or "\\" in n or not n.endswith(".hdf5") for n in names):
            raise ValueError("Invalid locked HDF5 filename")
        used |= picked
