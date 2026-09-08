"""Run on the Linux instance before installing ML dependencies. Requires only PyYAML.

NPY header inspection additionally uses NumPy when available. Writes only the requested report.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cognitive_ultrasound.cloud import preflight  # noqa: E402
from cognitive_ultrasound.config import ROOT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/autodl/paths.yaml"))
    p.add_argument("--output", default="/root/autodl-tmp/outputs_casl/preparation/preflight.json")
    args = p.parse_args()
    if sys.platform != "linux":
        p.error(
            "This entry point is for the Linux AutoDL instance; no remote inspection has occurred"
        )
    result = preflight(args.config, args.output)
    inputs = result["inputs"]
    print(
        json.dumps(
            {
                k: inputs[k]
                for k in (
                    "ready_for_conversion_inventory",
                    "problems",
                    "original_split_counts",
                    "casl_split_counts",
                    "casl_vs_original_split_crosswalk",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Report: {args.output}")
    return 0 if inputs["ready_for_conversion_inventory"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
