"""Rebuild a completed run's report using an explicit, audited rendering patch only."""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.preparation.suite import bundle_results, run_lock
from cognitive_ultrasound.provenance import sha256


def inventory(root):
    return {
        str(f.relative_to(root)): (f.stat().st_size, f.stat().st_mtime_ns)
        for f in (root / "jobs").rglob("*")
        if f.is_file() and f.suffix in (".json", ".npz")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-file", required=True)
    args = parser.parse_args()
    root, code = Path(args.output).resolve(), Path(args.analysis_file).resolve()
    audit = code.parent
    with run_lock(root):
        ledger = read_json(root / "status.json")
        if not ledger.get("jobs") or any(
            v["status"] != "completed" for v in ledger["jobs"].values()
        ):
            raise RuntimeError("Repair only: all scientific tasks must already be completed")
        for job in ledger["jobs"]:
            if read_json(root / "jobs" / job / "result.json")["status"] != "completed":
                raise RuntimeError("Task completion marker missing: " + job)
        saved = audit / "status.before.json"
        if saved.exists():
            raise FileExistsError("Use a new repair audit directory")
        atomic_json(saved, ledger)
        before = inventory(root)
        identity_hash = sha256(root / "identity.json")
        spec = importlib.util.spec_from_file_location(
            "cognitive_ultrasound.preparation.analysis", code
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        record = dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            renderer=str(code),
            renderer_sha256=sha256(code),
            experiment_identity_sha256=identity_hash,
            inference_rerun=False,
        )
        try:
            module.report(root)
            assert before == inventory(root)
            assert sha256(root / "identity.json") == identity_hash
            ledger.pop("error", None)
            ledger.update(status="completed", report_repair=str(audit.relative_to(root)))
            atomic_json(root / "status.json", ledger)
            record.update(status="completed", scientific_artifacts_unchanged=True)
            atomic_json(audit / "result.json", record)
            archive = bundle_results(root)
            if archive is None:
                raise RuntimeError("Report rebuilt but result archive was not rebuilt")
            print(
                json.dumps(
                    dict(status="completed", archive=str(archive), bytes=archive.stat().st_size)
                ),
                flush=True,
            )
        except BaseException as error:
            ledger.update(status="failed", error=f"Report repair: {type(error).__name__}: {error}")
            atomic_json(root / "status.json", ledger)
            record.update(status="failed", error=str(error))
            atomic_json(audit / "result.json", record)
            raise


if __name__ == "__main__":
    main()
