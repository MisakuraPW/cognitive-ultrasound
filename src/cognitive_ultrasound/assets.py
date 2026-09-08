from pathlib import Path

from .config import LPIPS_REVISION, SEGMENTATION_REVISION
from .provenance import sha256, write_json

ASSETS = {
    "lpips": (
        "zeahub/lpips",
        LPIPS_REVISION,
        ["config.json", "lin/lin.weights.h5", "vgg/vgg.weights.h5"],
    ),
    "segmentation": (
        "zeahub/echonet-dynamic",
        SEGMENTATION_REVISION,
        [
            "config.json",
            "saved_model.pb",
            "fingerprint.pb",
            "variables/variables.data-00000-of-00001",
            "variables/variables.index",
        ],
    ),
}


def fetch_evaluation_assets(folder, names=("lpips", "segmentation")):
    from huggingface_hub import snapshot_download

    for name in names:
        repo, revision, files = ASSETS[name]
        destination = Path(folder) / name
        snapshot_download(repo, revision=revision, local_dir=destination, allow_patterns=files)
        write_json(
            destination / "provenance.json",
            {
                "repo": repo,
                "revision": revision,
                "files": {f: sha256(destination / f) for f in files},
            },
        )


def asset_hashes(folder, names):
    hashes = {}
    for name in names:
        hashes[name] = {f: sha256(Path(folder) / name / f) for f in ASSETS[name][2]}
    return hashes
