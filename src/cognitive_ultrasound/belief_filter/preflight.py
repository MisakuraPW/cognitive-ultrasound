"""Bounded, isolated GPU calibration; never advances production optimizer state.

Only execution mode and CPU thread count vary. All scientific settings are fixed.
Each new stage/run measures again; a changed device/runtime invalidates its receipt.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..config import ROOT
from ..hardware import snapshot, thread_candidates
from ..preparation.common import atomic_json, digest, read_json
from ..provenance import sha256


def choose(records, remaining_steps):
    options = []
    for record in records:
        for mode, measurement in record.get("modes", {}).items():
            if measurement.get("passed") and measurement.get("headroom_ok"):
                cost = measurement["cold_s"] + max(0, remaining_steps - 1) * measurement["median_s"]
                options.append(dict(execution=mode, threads=record["threads"], projected_s=cost))
    if not options:
        raise RuntimeError(
            "No configuration passed GPU smoke/parity/headroom checks; see preflight"
        )
    best = min(options, key=lambda x: x["projected_s"])
    eager = [x for x in options if x["execution"] == "eager"]
    # Avoid selecting graph for a marginal or noisy timing advantage.
    baseline = min(eager, key=lambda x: x["projected_s"]) if eager else None
    if (
        best["execution"] == "graph"
        and baseline
        and best["projected_s"] > baseline["projected_s"] * 0.9
    ):
        best = baseline
    return best


def calibrate(
    cfg, stage, output, checkpoint=None, cpu=False, remaining_steps=None, fresh_optimizer=False
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    machine = snapshot()
    sources = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    identity = dict(
        hardware=machine,
        config=cfg,
        stage=stage,
        sources=sources,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        fresh_optimizer=fresh_optimizer,
    )
    key = digest(identity)
    receipt_id = key[:16] + "-" + str(time.time_ns())
    directory = output / receipt_id
    directory.mkdir(exist_ok=True)
    if cpu:
        receipt = dict(
            receipt_id=receipt_id,
            identity=identity,
            fingerprint=key,
            chosen=dict(execution="eager", threads=1),
            reason="Explicit CPU functional run; GPU acceleration unverified",
            candidates=[],
        )
        atomic_json(output / "selection.json", receipt)
        return receipt
    remaining = cfg["steps"][stage]
    if checkpoint:
        with np.load(checkpoint, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
        if meta["stage"] == stage and not fresh_optimizer:
            remaining = max(1, remaining - meta["step"])
    records = []
    # Fresh short measurements even with a matching receipt: current contention may differ.
    started = time.monotonic()
    for threads in thread_candidates(machine["cpu_limit"]):
        if time.monotonic() - started > 480:
            break
        request = directory / f"request-{threads}.json"
        result = directory / f"result-{threads}.json"
        atomic_json(
            request,
            dict(
                config=cfg,
                stage=stage,
                checkpoint=str(checkpoint) if checkpoint else None,
                fresh_optimizer=fresh_optimizer,
                threads=threads,
                output=str(result),
            ),
        )
        env = dict(
            os.environ,
            PYTHONPATH=str(ROOT / "src"),
            KERAS_BACKEND="tensorflow",
            OMP_NUM_THREADS=str(threads),
            TF_NUM_INTRAOP_THREADS=str(threads),
            TF_NUM_INTEROP_THREADS="1",
            PYTHONUNBUFFERED="1",
        )
        print(f"PREFLIGHT {stage}: eager/graph, threads={threads}", flush=True)
        # Timeout/OOM are confined to this disposable process; no production checkpoints written.
        with (directory / f"probe-{threads}.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "cognitive_ultrasound.belief_filter.preflight",
                        "--probe",
                        str(request),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
                record = (
                    read_json(result)
                    if completed.returncode == 0
                    else dict(error=f"exit {completed.returncode}")
                )
            except subprocess.TimeoutExpired:
                record = dict(error="180 second probe timeout")
        records.append(dict(threads=threads, **record))
    if remaining_steps is not None:
        remaining = remaining_steps
    try:
        selected = choose(records, remaining)
    except RuntimeError:
        atomic_json(
            output / "selection.json", dict(status="failed", identity=identity, candidates=records)
        )
        raise
    receipt = dict(
        receipt_id=receipt_id,
        identity=identity,
        fingerprint=key,
        chosen=selected,
        candidates=records,
        remaining_steps=remaining,
        calibration_wall_s=time.monotonic() - started,
        scope="Fastest passing tested candidate, not a global optimum; FP32 and objective unchanged",
        limits="Two parity updates and five timed clips; not proof of full-training equivalence",
    )
    atomic_json(directory / "selection.json", receipt)
    atomic_json(output / "selection.json", receipt)
    print("PREFLIGHT_SELECTED " + json.dumps(selected), flush=True)
    return receipt


def probe(request):
    from .core import Models
    from .kernels import make_kernels
    from .runner import Clips, load_checkpoint, loss_for_clip, runtime

    tf = runtime()
    cfg, stage = request["config"], request["stage"]
    tf.keras.utils.set_random_seed(cfg["seed"])
    data = Clips(cfg, "train", cfg["train_cases"])
    clips = [data.sample(cfg["seed"] + i) for i in range(7)]
    models, optimizers, functions = {}, {}, {}
    for mode in ("eager", "graph"):
        if (
            mode == "graph"
            and stage == "filter"
            and cfg.get("training_policy", "greedy") != "uniform"
        ):
            continue
        model = Models(cfg)
        variables = model.configure_stage(stage)
        optimizer = tf.keras.optimizers.Adam(cfg["learning_rate"])
        optimizer.build(variables)
        if request.get("checkpoint"):
            with np.load(request["checkpoint"], allow_pickle=False) as archive:
                saved = json.loads(str(archive["metadata"]))
            load_checkpoint(
                Path(request["checkpoint"]),
                model,
                optimizer
                if saved["stage"] == stage and not request.get("fresh_optimizer")
                else None,
            )
        if mode == "graph":
            for name, component in models["eager"].all().items():
                model.all()[name].set_weights(component.get_weights())
            for a, b in zip(optimizers["eager"].variables, optimizer.variables):
                b.assign(a)
            function, _ = make_kernels(model, optimizer, stage, cfg)
        else:

            def function(clip, start, model=model, optimizer=optimizer, variables=variables):
                with tf.GradientTape() as tape:
                    loss = loss_for_clip(model, clip, int(start), stage, cfg)
                tf.debugging.assert_all_finite(loss, "Nonfinite loss")
                gradients = tape.gradient(loss, variables)
                if any(g is None for g in gradients):
                    raise RuntimeError("Missing gradient")
                for g in gradients:
                    tf.debugging.assert_all_finite(g, "Nonfinite gradient")
                gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, variables))
                return loss

        models[mode], optimizers[mode], functions[mode] = model, optimizer, function
    initial = {
        mode: (
            {name: m.get_weights() for name, m in model.all().items()},
            [v.numpy() for v in optimizers[mode].variables],
        )
        for mode, model in models.items()
    }
    losses, cold = {}, {}
    for mode, function in functions.items():
        values = []
        for i in range(2):
            started = time.perf_counter()
            clip, start = clips[i]
            values.append(float(function(clip, tf.constant(start, tf.int32))))
            optimizers[mode].variables[-1].numpy()  # synchronize completed update, not just loss
            if i == 0:
                cold[mode] = time.perf_counter() - started
        losses[mode] = values
    parity = True
    parity_error = None
    if "graph" in models:
        try:
            np.testing.assert_allclose(losses["graph"], losses["eager"], rtol=2e-4, atol=2e-6)
            for name, model in models["eager"].all().items():
                for a, b in zip(model.get_weights(), models["graph"].all()[name].get_weights()):
                    np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-6)
            for a, b in zip(optimizers["eager"].variables, optimizers["graph"].variables):
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-4, atol=2e-6)
        except AssertionError as error:
            parity, parity_error = False, str(error)[-1500:]
    modes = {}
    for mode, function in functions.items():
        # Restore identical initial state before timed clips; production state is never touched.
        weights, state = initial[mode]
        for name, model in models[mode].all().items():
            model.set_weights(weights[name])
        for variable, value in zip(optimizers[mode].variables, state):
            variable.assign(value)
        tf.config.experimental.reset_memory_stats("GPU:0")
        samples = []
        for i in range(2, 7):
            started = time.perf_counter()
            clip, start = data.sample(cfg["seed"] + i)  # include input read, like production
            value = float(function(clip, tf.constant(start, tf.int32)))
            optimizers[mode].variables[-1].numpy()
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite timed loss")
            samples.append(time.perf_counter() - started)
        peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]
        # Both probe models coexist: conservative allocator headroom estimate.
        memory = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        capacities = [tuple(map(float, line.split(","))) for line in memory]
        free, total = min(capacities, key=lambda pair: pair[0] / pair[1])
        modes[mode] = dict(
            passed=mode == "eager" or parity,
            cold_s=cold[mode],
            median_s=float(np.median(samples)),
            samples_s=samples,
            peak_allocator_bytes=peak,
            headroom_ok=free >= max(1024, total * 0.1),
            free_mib=free,
        )
    atomic_json(
        request["output"],
        dict(
            tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
            tensorflow_build=tf.sysconfig.get_build_info(),
            modes=modes,
            graph_parity_passed=parity,
            graph_parity_error=parity_error,
            loss_pairs=losses,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()
    probe(read_json(args.probe))


if __name__ == "__main__":
    main()
