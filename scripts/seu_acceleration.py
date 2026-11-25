"""Bind school paths and lock the common benchmark before training."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def configure(task, output, data, jax_python, torch_python):
    bf = yaml.safe_load(Path("configs/fast_belief.yaml").read_text(encoding="utf-8"))
    bench = yaml.safe_load(Path("configs/speedbench.yaml").read_text(encoding="utf-8"))
    smoke = task == "bf-smoke"
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
    write(output / "training.json", bf)
    write(output / "benchmark.json", bench)
    print(f"BF cohort={bf['cohorts']} benchmark={names} frames={bench['frames']}", flush=True)


def summarize(output):
    status = json.loads((output / "benchmark/status.json").read_text())
    print(json.dumps(status["methods"], ensure_ascii=False))
    for marker in sorted((output / "benchmark").glob("*/*/*/complete.json")):
        record = json.loads(marker.read_text())
        fields = ("method", "case", "mode", "warm", "quality", "late_unobserved_mae")
        print(json.dumps({key: record[key] for key in fields}))
    if status["status"] != "completed":
        raise RuntimeError("One or more benchmark methods did not complete")
    print(f"SEU_ACCELERATION_OK {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_subparsers(dest="action", required=True)
    setup = action.add_parser("configure")
    setup.add_argument("task", choices=["bf-smoke", "bf-train"])
    setup.add_argument("output", type=Path)
    setup.add_argument("data")
    setup.add_argument("jax_python")
    setup.add_argument("torch_python")
    result = action.add_parser("summarize")
    result.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "configure":
        configure(args.task, args.output, args.data, args.jax_python, args.torch_python)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
