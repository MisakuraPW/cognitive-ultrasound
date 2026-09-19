"""One task per process: never mix TensorFlow and JAX training runtimes."""

from .common import read_json


def execute(root, task):
    cfg, manifest = read_json(root / "config.json"), read_json(root / "manifest.json")
    output = root / "jobs" / task["id"]
    output.mkdir(parents=True, exist_ok=True)
    kind = task["kind"]
    if kind == "reuse":
        import shutil

        from .common import atomic_json

        source = root / "jobs" / task["source"]
        if read_json(source / "result.json")["status"] != "completed":
            raise ValueError("Cannot reuse incomplete trajectories")
        for file in source.glob("*/*/*"):
            if file.is_file() and (file.suffix == ".npz" or file.name == "complete.json"):
                target = output / file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
        atomic_json(
            output / "result.json",
            dict(**read_json(source / "result.json"), reused_from=task["source"]),
        )
    elif kind == "trajectory":
        from .casl import run_trajectory

        run_trajectory(task, cfg, manifest, output)
    elif kind in ("fixed_history", "branches", "closed_loop"):
        from . import casl

        getattr(casl, kind)(task, cfg, manifest, output, root)
    elif kind in ("bf_train", "bf_qualify"):
        from .belief import qualify, train_stage

        (train_stage if kind == "bf_train" else qualify)(task, cfg, manifest, output, root)
    elif kind == "tbig":
        from .tbig import run

        run(task, cfg, output)
    elif kind == "risk_probe":
        from .analysis import risk_probe

        risk_probe(root, cfg, output)
    elif kind == "parity":
        from .analysis import parity

        parity(root, output)
    else:
        raise ValueError(f"Unknown task kind {kind}")
