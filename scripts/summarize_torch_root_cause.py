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
        if not trace.exists():
            continue
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
    jax = args.root / "jax_fixed/result.json"
    if jax.exists():
        out["jax_fixed"] = json.loads(jax.read_text())
    out["fixed_frame_comparisons"] = {}
    for reference in ("bilinear_original", "jax_fixed"):
        ref = args.root / reference / "frame_outputs.npz"
        if not ref.exists():
            continue
        with np.load(ref) as expected:
            for name in ("bilinear_original", "bilinear_separable", "separable_contiguous"):
                target = args.root / name / "frame_outputs.npz"
                if not target.exists():
                    continue
                with np.load(target) as actual:
                    checks = {}
                    for key in actual.files:
                        x, y = actual[key], expected[key]
                        if x.shape != y.shape:
                            raise ValueError(f"Output shape mismatch: {name}, {reference}, {key}")
                        delta = np.abs(x.astype(float) - y.astype(float))
                        checks[key] = dict(
                            max_abs=float(delta.max()),
                            mean_abs=float(delta.mean()),
                            tolerance_passed=bool(np.allclose(x, y, atol=2e-4, rtol=2e-4)),
                            exactly_equal=bool(np.array_equal(x, y)),
                        )
                    out["fixed_frame_comparisons"][name + "_vs_" + reference] = checks
    (args.root / "summary.json").write_text(json.dumps(out, indent=2))
    for name, record in records.items():
        print(name, record["result"]["timings"]["frame_graph_50"], record["frame_profile"])
    print("REPLACEMENT", comparison)


if __name__ == "__main__":
    main()
