"""Audit and finish the ORIGINAL finite closure design in a separate copy only."""

import shutil
from pathlib import Path

from ..preparation.common import atomic_json, read_json
from ..provenance import sha256
from .report import closure_audit


def finish(suite):
    cfg, root = suite.cfg, suite.root
    audit = closure_audit(cfg, root)
    if audit["status"] != "audited":
        raise FileNotFoundError("Full closure results required before new cohort lock")
    source = Path(cfg["closure_root"])
    state = read_json(source / "status.json")
    missing = [
        name
        for name, job in state["jobs"].items()
        if name != "tbig"
        and (job["status"] != "completed" or not (source / "jobs" / name / "result.json").exists())
    ]
    destination = root / "closure_finished"
    receipt = destination / "copy_receipt.json"
    if not receipt.exists():
        # Independent files, not hard links: repairs must not modify the old evidence.
        shutil.copytree(
            source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("run.lock")
        )
        atomic_json(receipt, dict(source=str(source), status_sha256=sha256(source / "status.json")))
    original_status = read_json(destination / "status.json")
    for name in missing:
        taskfile = destination / "jobs" / name / "task.json"
        if not taskfile.exists():
            # No original executable specification: don't invent a scientific task.
            atomic_json(
                root / ("closure_missing_" + name + ".json"),
                dict(status="blocked", reason="Original task specification absent", job=name),
            )
            continue
        task = read_json(taskfile)
        value = suite.job(
            "closure_repair_" + name,
            dict(kind="closure_repair", old_task=task, closure_root=str(destination)),
            "tensorflow" if task["kind"].startswith("closure_bf") else "jax",
        )
        original_status["jobs"][name]["status"] = value["status"]
    atomic_json(destination / "status.json", original_status)
    from ..preparation.closure import render_report

    render_report(destination)
    atomic_json(
        root / "closure_completion.json",
        dict(
            original_missing=missing,
            source_retained=True,
            original_design_unchanged=True,
            report=str(destination / "REPORT.md"),
            status="completed"
            if all(
                v["status"] == "completed"
                for k, v in original_status["jobs"].items()
                if k != "tbig"
            )
            else "finished_with_gaps",
        ),
    )


def repair(task, output):
    from ..preparation.worker import execute

    root = Path(task["closure_root"])
    execute(root, task["old_task"])
    result = read_json(root / "jobs" / task["old_task"]["id"] / "result.json")
    atomic_json(
        output / "result.json",
        dict(
            status=result["status"],
            original_task=task["old_task"],
            result_path=str(root / "jobs" / task["old_task"]["id"]),
        ),
    )
