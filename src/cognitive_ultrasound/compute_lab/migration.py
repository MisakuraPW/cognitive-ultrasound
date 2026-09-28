"""Explicit v2 inheritance. Old identity/evidence are never rewritten."""

import ast
import hashlib
import shutil
import subprocess
from pathlib import Path

from ..config import ROOT
from ..preparation.common import atomic_json, read_json, source_identity
from ..preparation.suite import run_lock
from ..provenance import sha256

BASE = "24aad69"
# Changes outside this reviewed repair scope cannot reuse v2 evidence.
ALLOWED = {
    "src/cognitive_ultrasound/compute_lab/suite.py",
    "src/cognitive_ultrasound/compute_lab/engines.py",
    "src/cognitive_ultrasound/compute_lab/__main__.py",
    "src/cognitive_ultrasound/compute_lab/report.py",
    "src/cognitive_ultrasound/compute_lab/migration.py",
    "src/cognitive_ultrasound/compute_lab/scheduling.py",
    "src/cognitive_ultrasound/compute_lab/readiness.py",
    "src/cognitive_ultrasound/compute_lab/data.py",
    "src/cognitive_ultrasound/torch_casl/native.py",
    "scripts/setup_compute_lab.sh",
    "scripts/run_compute_lab.sh",
    "scripts/diagnose_compute_repair.py",
}


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def symbol(text, name):
    return ast.dump(
        next(x for x in ast.parse(text).body if getattr(x, "name", None) == name),
        include_attributes=False,
    )


def validate_source(old):
    current = source_identity()
    changed = {p for p in set(old) | set(current) if old.get(p) != current.get(p)}
    if changed - ALLOWED:
        raise ValueError(f"Unreviewed implementation changes: {sorted(changed - ALLOWED)}")
    for p, digest in old.items():
        if hashlib.sha256(git_bytes(p)).hexdigest() != digest:
            raise ValueError(f"Source is not the reviewed {BASE} batch: {p}")
    for p, name in [
        ("src/cognitive_ultrasound/compute_lab/engines.py", "JaxEngine"),
        ("src/cognitive_ultrasound/torch_casl/native.py", "NativeCASL"),
        ("src/cognitive_ultrasound/torch_casl/native.py", "FrozenGraph"),
    ]:
        if symbol(git_bytes(p).decode(), name) != symbol(
            (ROOT / p).read_text(encoding="utf-8"), name
        ):
            raise ValueError(f"Reused computational dependency changed: {name}")
    data_path = "src/cognitive_ultrasound/compute_lab/data.py"
    before = ast.parse(git_bytes(data_path).decode())
    after = ast.parse((ROOT / data_path).read_text(encoding="utf-8"))
    # Only the close-on-exec descriptor hygiene is compatible with prior JAX data
    # results. Do not waive checks for any changes to reading/writing algorithms.
    after.body = [
        node
        for node in after.body
        if not (
            isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "os"
        )
    ]
    writer = next(
        node for node in after.body if isinstance(node, ast.ClassDef) and node.name == "Writer"
    )
    init = next(
        node
        for node in writer.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    expected = ast.parse(
        "if os.name == 'posix' and self.h5.driver == 'sec2':\n    os.set_inheritable(self.h5.id.get_vfd_handle(), False)"
    ).body[0]
    matches = [
        node
        for node in init.body
        if ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)
    ]
    if len(matches) != 1:
        raise ValueError("Expected reviewed descriptor hygiene only")
    init.body.remove(matches[0])
    if ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
        raise ValueError("Data computation changed beyond descriptor hygiene")
    return sorted(changed)


def inherit(source, root, cfg, environment):
    from .suite import owned_alive

    source = Path(source).resolve()
    if source == root.resolve():
        raise ValueError("Inheritance requires a NEW directory")
    receipt = root / "inheritance.json"
    if receipt.exists():
        saved = read_json(receipt)
        if saved["source"] != str(source):
            raise ValueError("Inheritance source changed")
        return
    if (root / "identity.json").exists() or (root / "jobs").exists():
        raise ValueError("Destination already initialized; use a fresh output")
    with run_lock(source):
        state = read_json(source / "status.json")
        if owned_alive(state.get("pid"), state.get("created")) or owned_alive(
            state.get("worker_pid"), state.get("worker_created")
        ):
            raise ValueError("Source coordinator/worker must be stopped before inheritance")
        identity = read_json(source / "identity.json")
        if read_json(source / "config.json") != cfg or identity["environment"] != environment:
            raise ValueError("Scientific config or environment incompatible; no inheritance")
        changed = validate_source(identity["source"])
        records = {}

        def copy(file):
            relative = file.relative_to(source)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = sha256(file)
            shutil.copy2(file, target)
            if sha256(target) != digest:
                raise ValueError(f"Copy verification failed: {relative}")
            records[str(relative)] = digest

        jobs = []
        for f in sorted((source / "jobs").glob("*/result.json")):
            value = read_json(f)
            name = f.parent.name
            eligible = (
                value.get("profile", {}).get("backend") == "jax"
                or name == "export"
                or name in ("short_torch_eager_b14", "development_torch_eager_b14")
            )
            if value["status"] == "completed" and eligible:
                for file in f.parent.rglob("*"):
                    if file.is_file() and file.suffix not in (".partial", ".tmp"):
                        copy(file)
                jobs.append(name)
        for folder in ("export", ".cache", "closure_finished"):
            for file in (source / folder).rglob("*"):
                if (
                    file.is_file()
                    and file.suffix not in (".partial", ".tmp")
                    and file.name not in ("run.lock", ".run.lock")
                ):
                    copy(file)
        for name in (
            "manifest.json",
            "throughput.json",
            "closure_completion.json",
            "closure_audit.json",
            "closure_bundle_receipt.json",
            "closure-v4.full-results.tar.gz",
            "closure-v4.full-results.tar.gz.sha256",
        ):
            if (source / name).exists():
                copy(source / name)
        # Failed Torch jobs remain historical evidence; they are NOT terminal jobs in new batch.
        evidence = root / "inherited_evidence"
        evidence.mkdir(exist_ok=True)
        for name in ("identity.json", "status.json", "config.json"):
            shutil.copy2(source / name, evidence / name)
        for name in ("short_torch_compile_b14", "short_torch_graph_b14"):
            if (source / "jobs" / name).exists():
                shutil.copytree(source / "jobs" / name, evidence / name)
        atomic_json(
            receipt,
            dict(
                source=str(source),
                source_identity_sha256=sha256(source / "identity.json"),
                reviewed_base=BASE,
                changed_paths=changed,
                jobs=jobs,
                files=records,
                compatibility="JAX class, NativeCASL, FrozenGraph AST unchanged; data, inference, weights and environment unchanged. Eager retained as historical cost evidence; repaired compile/graph rerun.",
                original_evidence_untouched=True,
            ),
        )
