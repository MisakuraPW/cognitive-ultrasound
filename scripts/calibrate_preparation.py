"""Benchmark acceleration against existing BF checkpoints without updating them."""

import argparse
from pathlib import Path

from cognitive_ultrasound.belief_filter.preflight import calibrate
from cognitive_ultrasound.preparation.belief import configuration
from cognitive_ultrasound.preparation.common import atomic_json, read_json, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.from_run).resolve(), Path(args.output).resolve()
    if output == root or root in output.parents:
        parser.error("Calibration output must be separate from the existing experiment")
    cfg, manifest = read_json(root / "config.json"), read_json(root / "manifest.json")
    verify_manifest(cfg, manifest)
    model_cfg = configuration(cfg, manifest)
    model_cfg["execution"] = "auto"
    reports = {}
    for stage in ("codec", "prior", "filter"):
        reports[stage] = calibrate(
            model_cfg,
            stage,
            output / stage,
            root / "jobs" / ("bf_" + stage) / "training/checkpoint.npz",
            remaining_steps=model_cfg["steps"][stage],
        )
        atomic_json(output / "summary.json", reports)
    print("HARDWARE_CALIBRATION_COMPLETED; original checkpoints unchanged", flush=True)


if __name__ == "__main__":
    main()
