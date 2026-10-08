"""Narrow audited reuse of committed E2 updates after the fixed-pair/GS-only repair."""

import argparse
import ast
import copy
import hashlib
import shutil
import subprocess
from pathlib import Path

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.preparation.suite import run_lock
from cognitive_ultrasound.provenance import sha256
from cognitive_ultrasound.task_budget.data import configuration, lock_manifest
from cognitive_ultrasound.task_budget.suite import identity, process_alive

BASE = "a8f4d26"
EPISODE = "src/cognitive_ultrasound/task_budget/episode.py"
PERCEPTION = "src/cognitive_ultrasound/task_budget/perception.py"


def base_source(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def tree(text, name):
    return next(n for n in ast.parse(text).body if getattr(n, "name", None) == name)


def same(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def audit_e2_source(old_identity):
    for path, digest in old_identity["source"].items():
        if hashlib.sha256(base_source(path)).hexdigest() != digest:
            raise ValueError(f"Unreviewed source snapshot: {path}")
        if path not in (EPISODE, PERCEPTION) and sha256(ROOT / path) != digest:
            raise ValueError(f"E2 dependency changed: {path}")
    old = tree(base_source(EPISODE).decode(), "rollout")
    new = tree((ROOT / EPISODE).read_text(), "rollout")
    removed = [
        n
        for n in new.body
        if not isinstance(n, ast.For)
        and any(isinstance(x, ast.Name) and x.id == "fixed_array" for x in ast.walk(n))
    ]
    if len(removed) != 3:
        raise ValueError("Unexpected fixed-array repair shape")
    new.body = [n for n in new.body if n not in removed]
    old_assignment = next(
        n
        for n in ast.walk(old)
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "fixed_now" for x in n.targets)
    )

    class RestoreFixedAssignment(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(x, ast.Name) and x.id == "fixed_now" for x in node.targets):
                return copy.deepcopy(old_assignment)
            return self.generic_visit(node)

    new = RestoreFixedAssignment().visit(new)
    if not same(old, new):
        raise ValueError("Rollout changed beyond JSON-pair normalization")
    # The E2 training call must still use the same fixed default tuple.
    if not same(
        tree(base_source("src/cognitive_ultrasound/task_budget/experiment.py").decode(), "train"),
        tree((ROOT / "src/cognitive_ultrasound/task_budget/experiment.py").read_text(), "train"),
    ):
        raise ValueError("Training algorithm changed")
    old_class = tree(base_source(PERCEPTION).decode(), "CASLPerception")
    new_class = tree((ROOT / PERCEPTION).read_text(), "CASLPerception")
    new_class.body = [n for n in new_class.body if getattr(n, "name", None) != "infer_for_replay"]
    init = next(n for n in new_class.body if getattr(n, "name", None) == "__init__")
    additions = [
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Attribute) and x.attr == "replay_warm" for x in n.targets)
    ]
    if len(additions) != 1:
        raise ValueError("Unexpected replay initialization repair")
    init.body.remove(additions[0])
    if not same(old_class, new_class):
        raise ValueError("Normal CASL perception changed")
    return dict(
        reviewed_base=BASE,
        normal_perception_ast_identical=True,
        training_ast_identical=True,
        rollout_change="pair-vs-schedule normalization; E2 uses unchanged default tuple",
        gs_repair="new replay-only VJP; normal infer path unchanged",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--config", default="configs/task_budget_ef.yaml")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or output.exists():
        raise ValueError("Use a new absent output directory")
    diagnostic = read_json(args.diagnostic)
    if not diagnostic["same_primal_custom_vjp"]["passed"] or not diagnostic["e2_forward"]["passed"]:
        raise ValueError("Numerical diagnostic not passed")
    cfg = configuration(args.config)
    with run_lock(source):
        status = read_json(source / "status.json")
        if process_alive(status.get("pid"), status.get("created")) or process_alive(
            status.get("worker_pid"), status.get("worker_created")
        ):
            raise ValueError("Original run must be stopped")
        old = read_json(source / "identity.json")
        audit = audit_e2_source(old)
        if cfg != read_json(source / "config.json"):
            raise ValueError("Scientific configuration changed")
        manifest = lock_manifest(cfg, source)
        current = identity(cfg, manifest)
        for key in (
            "config",
            "manifest",
            "checkpoint",
            "ef_weights",
            "ef_stats",
            "hardware",
            "python",
            "task_environment",
            "upstream",
            "runtime_environment",
        ):
            if current[key] != old[key]:
                raise ValueError(f"Environment/science identity changed: {key}")
        job = source / "jobs/E2_l0_s42_train"
        checkpoints = sorted((job / "updates").glob("*.npz"))
        if (
            not checkpoints
            or sha256(checkpoints[-1]) != diagnostic["e2_forward"]["checkpoint_sha256"]
        ):
            raise ValueError("Latest E2 checkpoint differs from verified checkpoint")
        output.mkdir(parents=True)
        copies = {}

        def preserve(file, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            if sha256(target) != sha256(file):
                raise ValueError(f"Copy failed: {target}")
            copies[str(target.relative_to(output))] = sha256(target)

        for name in ("manifest.json", "config.json"):
            preserve(source / name, output / name)
        for checkpoint in checkpoints:
            preserve(checkpoint, output / "jobs/E2_l0_s42_train/updates" / checkpoint.name)
            preserve(
                checkpoint.with_suffix(".json"),
                output / "jobs/E2_l0_s42_train/updates" / checkpoint.with_suffix(".json").name,
            )
        for name in ("identity.json", "status.json"):
            preserve(source / name, output / "inherited_evidence" / name)
        preserve(job / "console.log", output / "inherited_evidence/E2_partial.console.log")
        preserve(Path(args.diagnostic), output / "inherited_evidence/repair_diagnostic.json")
        atomic_json(
            output / "inheritance.json",
            dict(
                source=str(source),
                source_identity_sha256=sha256(source / "identity.json"),
                audit=audit,
                committed_e2_updates=len(checkpoints),
                copied_files=copies,
                original_results_preserved=True,
                probe_not_reused=True,
                inference_results_not_reused=True,
                scope="Only reviewed partial E2 training; new identity generated on startup",
            ),
        )
        print(f"Preserved {len(checkpoints)} committed E2 updates in {output}", flush=True)


if __name__ == "__main__":
    main()
