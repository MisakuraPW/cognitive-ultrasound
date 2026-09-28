"""One fixed warm frame: controlled determinism A/B and separate kernel profiling.

No cohort sweep, no training, no scientific-baseline adoption.
"""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deterministic", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from cognitive_ultrasound.torch_casl.native import FrozenGraph, NativeCASL, CapturedFrame, nchw

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(bool(args.deterministic))
    manifest = json.loads((root / "manifest.json").read_text())
    name = manifest["cohorts"]["debug"][0]
    source = root / ".cache/debug/b14/42" / Path(name).with_suffix(".h5").name
    with h5py.File(source) as h:
        before = {k: v[()] for k, v in h["1"].items()}
        current = {k: v[()] for k, v in h["2"].items()}
    network = FrozenGraph(root / "export").to(device="cuda", memory_format=torch.channels_last)
    model = NativeCASL(network, 14).to("cuda")
    frame_args = (
        nchw(current["resume_buffer"][None], "cuda").expand(2, -1, -1, -1),
        nchw(before["resume_mask"][None], "cuda"),
        nchw(before["resume_posterior_samples"], "cuda"),
        nchw(current["noise"], "cuda"),
    )
    measurement, mask, previous, z = frame_args
    t = torch.ones((2, 1, 1, 1), device="cuda") * model.max_t
    dt = model.max_t / 500
    n, s = model.rates(t - 449 * dt)
    x = s * previous + n * z
    n, s = model.rates(t - 450 * dt)
    nn, ns = model.rates(t - 451 * dt)
    step_args = (x, measurement, mask, n, s, nn, ns)
    timings = {}

    def measure(label, fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - begin)
        timings[label] = dict(
            mean_s=float(np.mean(samples)),
            p50_s=float(np.median(samples)),
            p95_s=float(np.quantile(samples, 0.95)),
        )
        print("TIMING", label, timings[label], flush=True)

    measure("network_eager", lambda: network(x, n.square()))
    compiled_network = torch.compile(network, fullgraph=True)
    measure("network_compiled", lambda: compiled_network(x, n.square()))
    measure("dps_eager", lambda: model.dps_step(*step_args))
    model.step_fn = torch.compile(model.dps_step, fullgraph=True)
    model.select = torch.compile(model.select, fullgraph=True)
    measure("dps_compiled", lambda: model.step_fn(*step_args))
    measure("frame_compiled_50", lambda: model.frame(*frame_args, steps=50))
    captured = CapturedFrame(model, frame_args, 50)
    measure("frame_graph_50", lambda: captured(*frame_args))
    reference = model.frame(*frame_args, steps=50)
    repeated = model.frame(*frame_args, steps=50)
    graph = captured(*frame_args)
    differences = {
        name: [float((a.float() - b.float()).abs().max()) for a, b in zip(values, reference)]
        for name, values in [("uncaptured_repeat", repeated), ("graph_vs_uncaptured", graph)]
    }

    def profile(label, fn):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(output / (label + ".trace.json")))
        events = []
        for event in prof.key_averages():
            events.append(
                dict(
                    name=event.key,
                    count=event.count,
                    cpu_us=event.self_cpu_time_total,
                    device_us=event.self_device_time_total,
                )
            )
        events.sort(key=lambda e: e["device_us"], reverse=True)
        (output / (label + ".operators.json")).write_text(json.dumps(events, indent=2))
        print("TOP_KERNELS", label, json.dumps(events[:12]), flush=True)
        return events

    ops = profile("dps_compiled", lambda: model.step_fn(*step_args))
    profile("frame_graph", lambda: captured(*frame_args))
    result = dict(
        deterministic=bool(args.deterministic),
        precision="fp32",
        steps=50,
        particles=2,
        budget=14,
        case=name,
        frame=2,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        timings=timings,
        differences=differences,
        profile_separate_from_timing=True,
        upsampling=[n["config"] for n in network.nodes if n["kind"] == "UpSampling2D"],
        top_dps=ops[:15],
    )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
