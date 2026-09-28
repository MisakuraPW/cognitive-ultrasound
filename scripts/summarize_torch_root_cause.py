"""Summarize diagnostic kernel traces without double counting CPU operator events."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def kernels(trace):
    events = json.loads(trace.read_text())["traceEvents"]
    gpu = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    buckets = {}
    for e in gpu:
        name = e["name"]
        if "indexing_backward" in name:
            category = "deterministic_index_accumulation"
        elif "RadixSort" in name or "radix_sort" in name:
            category = "index_sorting"
        elif "nhwcToNchw" in name or "nchwToNhwc" in name:
            category = "layout_conversion"
        else:
            category = "other"
        b = buckets.setdefault(category, {"count": 0, "milliseconds": 0.0})
        b["count"] += 1
        b["milliseconds"] += e["dur"] / 1000
    return dict(
        buckets=buckets,
        total_kernel_ms=sum(e["dur"] for e in gpu) / 1000,
        timing_warning="Instrumented trace durations; use uninstrumented timings for FPS",
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    records = {}
    for p in sorted(args.root.glob("*/result.json")):
        result = json.loads(p.read_text())
        trace = p.parent / "frame_graph.trace.json"
        records[p.parent.name] = dict(result=result, frame_profile=kernels(trace))
    a = args.root / "bilinear_original/frame_outputs.npz"
    b = args.root / "bilinear_separable/frame_outputs.npz"
    comparison = {}
    if a.exists() and b.exists():
        with np.load(a) as original, np.load(b) as changed:
            for k in original.files:
                x, y = original[k], changed[k]
                delta = np.abs(x.astype(float) - y.astype(float))
                comparison[k] = dict(
                    max_abs=float(delta.max()),
                    mean_abs=float(delta.mean()),
                    finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),
                    equivalent=bool(np.allclose(x, y, atol=2e-4, rtol=2e-4)),
                    bitwise_equal=bool(np.array_equal(x, y)),
                )
    out = dict(
        scope="Fixed debug input diagnosis, not a cohort quality or adoption result",
        variants=records,
        interpolation_comparison=comparison,
    )
    (args.root / "summary.json").write_text(json.dumps(out, indent=2))
    for name, record in records.items():
        print(name, record["result"]["timings"]["frame_graph_50"], record["frame_profile"])
    print("REPLACEMENT", comparison)


if __name__ == "__main__":
    main()
