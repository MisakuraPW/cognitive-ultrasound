"""Bind school paths and lock the common benchmark before training."""

import argparse
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import yaml

from cognitive_ultrasound.provenance import sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def completed_checkpoint(parent, task, relative):
    stage, name = relative.split("/", 1)
    pattern = f"{task}-*/{stage}/complete.json"
    receipts = sorted(parent.glob(pattern), key=lambda p: int(p.parent.parent.name.rsplit("-", 1)[1]))
    for receipt in reversed(receipts):
        record = json.loads(receipt.read_text())
        checkpoint = receipt.parent / name
        expected = record.get("selected_checkpoint_sha256", record.get("checkpoint_sha256"))
        if record["cpu_functional"]:
            continue
        if sha256(checkpoint) != expected:
            raise ValueError(f"Checkpoint integrity failed: {checkpoint}")
        return checkpoint
    raise FileNotFoundError(f"Missing verified GPU training output for {task}")


def configure(task, output, data, jax_python, torch_python):
    bf = yaml.safe_load(Path("configs/fast_belief.yaml").read_text(encoding="utf-8"))
    bench = yaml.safe_load(Path("configs/speedbench.yaml").read_text(encoding="utf-8"))
    smoke = task.endswith("-smoke")
    split = Path(bf["split_manifest"]).resolve()
    validation = sorted(yaml.safe_load(split.read_text(encoding="utf-8"))["val"])
    count = 2 if smoke else 8
    names = np.random.default_rng(bench["seed"]).permutation(validation).tolist()[:count]
    cores = len(os.sched_getaffinity(0))
    threads = min(4, max(1, cores - 2))
    bf.update(
        data_root=data, file_list=None, split_manifest=str(split),
        evaluation_split="val", reserved_validation=names,
    )
    bf["runtime"].update(threads=threads, workers=min(2, cores - 1), calibrate_threads=[threads])
    if smoke:
        bf["cohorts"] = dict(train=8, development=2, test=2)
        for stage in ("codec", "prior", "filter"):
            bf["training"][stage].update(epochs=1, batch=4)
        bf["evaluation"]["component_frames"] = 12
    bench.update(
        data_root=data, split_manifest=str(split), checkpoint=str(Path("checkpoints/official").resolve()),
        torch_export=None, belief_checkpoint=str(output / "belief/filter/best.pt"),
        pythons=dict(jax=jax_python, torch=torch_python), case_names=names, cases=count,
        frames=16 if smoke else 64, threads=threads,
        methods=["official", "jax_coarse4", "interpolation", "fast_belief"],
    )
    if task == "accel-compare":
        belief = completed_checkpoint(output.parent, "bf-train", "belief/filter/best.pt")
        vivit = completed_checkpoint(output.parent, "seqdiff-train", "vivit/best.keras")
        bench.update(
            belief_checkpoint=str(belief), vivit_checkpoint=str(vivit),
            methods=["official", "jax_coarse4", "seqdiff_plus", "interpolation", "fast_belief"],
        )
    elif task.startswith("bf-"):
        write(output / "training.json", bf)
    else:
        bench.update(
            belief_checkpoint=None, vivit_checkpoint=str(output / "vivit/best.keras"),
            methods=["official", "jax_coarse4", "seqdiff_plus", "interpolation"],
        )
        write(output / "training.json", dict(
            cases=8 if smoke else None, epochs=1 if smoke else 30, batch=4 if smoke else 8,
            development=2 if smoke else 64, min_epochs=1 if smoke else 10,
            patience=6, checkpoint_updates=100,
        ))
    write(output / "benchmark.json", bench)
    print(f"{task} benchmark={names} frames={bench['frames']}", flush=True)


def train_vivit(output):
    from cognitive_ultrasound.seqdiff_plus import train

    cfg = json.loads((output / "benchmark.json").read_text())
    train_cfg = json.loads((output / "training.json").read_text())
    train(argparse.Namespace(
        **train_cfg, data=Path(cfg["data_root"]), split=Path(cfg["split_manifest"]),
        reserved=cfg["case_names"], output=output / "vivit", seed=20261125, cpu=False,
    ))


def summarize(output, bench):
    status = json.loads((bench / "status.json").read_text())
    print(json.dumps(status["methods"], ensure_ascii=False))
    for marker in sorted(bench.glob("*/*/*/complete.json")):
        record = json.loads(marker.read_text())
        fields = ("method", "case", "mode", "warm", "quality", "late_unobserved_mae")
        print(json.dumps({key: record[key] for key in fields}))
    if status["status"] != "completed":
        raise RuntimeError("One or more benchmark methods did not complete")
    files = [
        output / "benchmark.json",
        *(bench / name for name in ("manifest.json", "identity.json", "status.json", "summary.csv", "speed_quality.png")),
        *sorted(bench.glob("*/*/*/complete.json")),
        *sorted(bench.glob("*/*/*/frames.json")),
    ]
    archive = output / "paired_results.tar.gz"
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as package:
        for member in files:
            package.add(member, arcname=member.relative_to(output))
    with tarfile.open(temporary, "r:gz") as package:
        for member in files:
            archived = package.extractfile(member.relative_to(output).as_posix())
            if archived.read() != member.read_bytes():
                raise ValueError(f"Archive validation failed: {member}")
    temporary.replace(archive)
    print(f"archive={archive} bytes={archive.stat().st_size} sha256={sha256(archive)}")
    print(f"SEU_ACCELERATION_OK {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_subparsers(dest="action", required=True)
    setup = action.add_parser("configure")
    setup.add_argument("task", choices=["bf-smoke", "bf-train", "seqdiff-smoke", "seqdiff-train", "accel-compare"])
    setup.add_argument("output", type=Path)
    setup.add_argument("data")
    setup.add_argument("jax_python")
    setup.add_argument("torch_python")
    result = action.add_parser("summarize")
    result.add_argument("output", type=Path)
    result.add_argument("--benchmark", type=Path)
    vivit = action.add_parser("train-vivit")
    vivit.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "configure":
        configure(args.task, args.output, args.data, args.jax_python, args.torch_python)
    elif args.action == "train-vivit":
        train_vivit(args.output)
    else:
        summarize(args.output, args.benchmark or args.output / "benchmark")


if __name__ == "__main__":
    main()
