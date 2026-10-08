"""Finite EF engineering calibration. Never resumes the research coordinator."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cognitive_ultrasound.preparation.common import atomic_json, atomic_npz, read_json
from cognitive_ultrasound.provenance import sha256


def compare(a, b):
    if set(a) != set(b):
        return dict(passed=False, reason="array keys differ")
    deltas = {}
    passed = True
    bitwise = True
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        valid = x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        ok = valid and np.allclose(x, y, atol=2e-4, rtol=2e-4)
        passed = passed and bool(ok)
        bitwise = bitwise and valid and x.dtype == y.dtype and x.tobytes() == y.tobytes()
        deltas[key] = float(np.max(np.abs(x-y))) if valid and x.size else None
    return dict(passed=passed, bitwise=bitwise, max_abs=deltas, atol=2e-4, rtol=2e-4)


def npz(file):
    with np.load(file, allow_pickle=False) as d:
        return {k: d[k].copy() for k in d.files}


def ef_trial(cfg, output, input_file, coordinates_file):
    from cognitive_ultrasound.task_budget.task import EFService
    from cognitive_ultrasound.task_budget.protocol import greedy_order, task_scores

    clips = npz(input_file)["clips"]
    coords = npz(coordinates_file)["coordinates"]
    startup = time.perf_counter()
    task = EFService(cfg, output, coords)
    try:
        first = task.request(clips, gradient=True)
        first_s = time.perf_counter()-startup
        for _ in range(3):
            task.request(clips, gradient=True)
        times = []
        for _ in range(20):
            start = time.perf_counter()
            value = task.request(clips, gradient=True)
            times.append(time.perf_counter()-start)
            if not compare(first, value)["passed"]:
                raise AssertionError("Changing output on repeated fixed EF input")
        # Separate signature: full-video EF prediction and its image adjoint.
        one = task.request(clips[:1], gradient=True)
        inference = task.request(clips[:1], gradient=False)
        score = task_scores(clips[:, -1], value["gradients"][:, -1])
        arrays = dict(predictions=value["predictions"], gradients=value["gradients"],
                      single_prediction=one["predictions"], single_gradient=one["gradients"],
                      inference=inference["predictions"], score=score,
                      lines=np.asarray(greedy_order(score, 28)))
        atomic_npz(output/"arrays.npz", **arrays)
        return dict(status="completed", first_request_and_startup_s=first_s,
                    mean_s=float(np.mean(times)), p50_s=float(np.median(times)),
                    p95_s=float(np.quantile(times, .95)), repeats=20, warmups=3,
                    runtime=cfg["runtime"], timing_scope="Synchronous EF input gradient plus IPC; not whole CASL")
    finally:
        task.close()


def rollout_trial(cfg, manifest, output):
    from cognitive_ultrasound.task_budget.experiment import evaluate
    root = output
    atomic_json(root/"config.json", cfg)
    small = copy.deepcopy(manifest)
    small["cohorts"]["development"] = manifest["cohorts"]["development"][:2]
    atomic_json(root/"manifest.json", small)
    spec = dict(id="paired64", kind="evaluate", method="E0", fixed=[10, 4],
                seed=42, cohort="development", _calibration_frames=64, _calibration_save_full=True)
    job = root/"jobs"/spec["id"]
    job.mkdir(parents=True, exist_ok=True)
    atomic_json(job/"task.json", spec)
    start = time.perf_counter()
    evaluate(spec, cfg, small, root, job)
    value = read_json(job/"result.json")
    arrays = {}
    discrete = {}
    for name in small["cohorts"]["development"]:
        d = job/Path(name).stem
        rows = read_json(d/"trajectory.json")
        for key in ("k1", "k2", "lines1", "lines2"):
            if key in rows[0]:
                discrete[name+":"+key] = [r[key] for r in rows]
        for key, arr in npz(d/"snapshots.npz").items():
            arrays[name+":"+key] = arr
        arrays[name+":all_images"] = npz(d/"calibration_images.npz")["images"]
        for key in ("state_before", "state_intermediate", "causal_task_particles_before", "causal_task_particles_intermediate"):
            arrays[name+":"+key] = np.array([r[key] for r in rows])
        arrays[name+":ef"] = np.array([read_json(d/"complete.json")["ef_prediction"]])
    atomic_npz(output/"arrays.npz", **arrays)
    atomic_json(output/"discrete.json", discrete)
    return dict(status="completed", process_work_s=time.perf_counter()-start,
                sum_case_s=sum(x["seconds"] for x in value["records"]), frames=128,
                runtime=cfg["runtime"], case_seconds=[x["seconds"] for x in value["records"]],
                timing_scope="two fixed cases, first64 frames, includes model load/JIT/full-input references and output")


def gs_trial(cfg, manifest, output):
    import jax
    from cognitive_ultrasound.task_budget.experiment import setup
    from cognitive_ultrasound.task_budget.data import read_episode
    from cognitive_ultrasound.task_budget.episode import rollout, gs_gradient
    from cognitive_ultrasound.task_budget.policy import initialize

    perception, task = setup(cfg, output)
    params = initialize(42, cfg)
    try:
        frames = read_episode(cfg, manifest, manifest["cohorts"]["development"][0], count=8)
        images, rows, contexts = rollout(cfg, perception, task, frames, params, "E2", 42, training=True)
        _, adjoint, _ = task.video(images, gradient=True)
        # One representative warm frame for each actually encountered K2 branch.
        chosen = []
        seen = set()
        for i, c in enumerate(contexts):
            if not c["cold"] and bool(c["k2"]) not in seen:
                chosen.append((c, adjoint[i]));seen.add(bool(c["k2"]))
        if False not in seen:
            zero_images, _, zero_contexts = rollout(cfg, perception, task, frames[:3], params, "E0", 42, fixed=(7,0))
            _, zero_adjoint, _ = task.video(zero_images, gradient=True)
            chosen.append((zero_contexts[-1], zero_adjoint[-1]));seen.add(False)
        results = {}
        baseline = None
        for mode in ("eager", "jit"):
            configured = copy.deepcopy(cfg)
            configured["runtime"]["gs_execution"] = mode
            begin = time.perf_counter()
            gradient, gate = gs_gradient(params, [c for c, _ in chosen], np.stack([g for _, g in chosen]),
                                         perception, configured, 1.0, 2.0)
            first_s = time.perf_counter()-begin
            first_arrays = {f"{h}.{k}":np.asarray(v) for h, tree in gradient.items() for k,v in tree.items()}
            if baseline is None:baseline = first_arrays
            times = []
            for _ in range(3):
                gs_gradient(params, [c for c, _ in chosen], np.stack([g for _, g in chosen]),
                            perception, configured, 1.0, 2.0)
            for _ in range(5):
                begin = time.perf_counter()
                gradient, gate = gs_gradient(params, [c for c, _ in chosen], np.stack([g for _, g in chosen]),
                                             perception, configured, .7, 2.0)
                jax.block_until_ready(gradient)
                times.append(time.perf_counter()-begin)
            arrays = {f"{h}.{k}":np.asarray(v) for h, tree in gradient.items() for k,v in tree.items()}
            atomic_npz(output/f"{mode}.npz", **arrays)
            results[mode] = dict(first_s=first_s, mean_s=float(np.mean(times)),
                                 p50_s=float(np.median(times)), p95_s=float(np.quantile(times,.95)),
                                 warmups=3, repeats=5, context_count=len(chosen), gate=gate,
                                 sampled_branches=sorted(seen),
                                 timing_scope="warm local DPS VJP;5 repeats due backward cost; not a full training update")
            atomic_json(output/"partial.json", results)
        results["verification"] = compare(npz(output/"eager.npz"), npz(output/"jit.npz"))
        # Explicitly change masks/history/temperature through the already captured signature.
        later = [(c, adjoint[i]) for i,c in enumerate(contexts) if not c["cold"]][-2:]
        a = copy.deepcopy(cfg);a["runtime"]["gs_execution"]="eager"
        b = copy.deepcopy(cfg);b["runtime"]["gs_execution"]="jit"
        x, _ = gs_gradient(params,[c for c,_ in later],np.stack([g for _,g in later]),perception,a,.4,8.0)
        y, _ = gs_gradient(params,[c for c,_ in later],np.stack([g for _,g in later]),perception,b,.4,8.0)
        results["changed_inputs_verification"] = compare(
            {f"{h}.{k}":np.asarray(v) for h,t in x.items() for k,v in t.items()},
            {f"{h}.{k}":np.asarray(v) for h,t in y.items() for k,v in t.items()})
        return dict(status="completed", **results)
    finally:
        task.close()


def training_trial(cfg, manifest, output, source, method):
    import shutil
    from cognitive_ultrasound.task_budget.experiment import train

    start_update = 0
    if method == "E2":
        old = source/"jobs/E2_l0_s42_train/updates"
        checkpoints = sorted(old.glob("*.npz"))
        if checkpoints:
            (output/"updates").mkdir(exist_ok=True)
            for file in (checkpoints[-1], checkpoints[-1].with_suffix(".json")):
                shutil.copy2(file, output/"updates"/file.name)
            start_update = int(npz(checkpoints[-1])["step"])
    spec = dict(id="isolated_training",kind="train",method=method,seed=42,**{"lambda":0.0})
    started=time.perf_counter()
    train(spec,cfg,manifest,output,output,end_update=start_update+2)
    records=[read_json(p) for p in sorted((output/"updates").glob("*.json")) if int(p.stem)>start_update]
    return dict(status="completed",method=method,start_update=start_update,production_advanced=False,
                measured_updates=records,process_work_s=time.perf_counter()-started,
                scope="two real updates in isolated clone; original50-update temperature and RNG schedule retained")


def child(args, name, cfg, action, timeout):
    output = args.output/name
    output.mkdir(parents=True, exist_ok=True)
    if (output/"result.json").exists():
        return read_json(output/"result.json")
    atomic_json(output/"trial_config.json", cfg)
    env = dict(os.environ, OMP_NUM_THREADS=str(cfg["runtime"]["threads"]))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--source", str(args.source),
               "--output", str(output), "--action", action, "--trial-config", str(output/"trial_config.json")]
    start = time.perf_counter()
    from cognitive_ultrasound.preparation.suite import stop_process
    with (output/"console.log").open("a",encoding="utf-8") as log:
        p = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process(p)
            atomic_json(output/"result.json",dict(status="failed",reason="bounded candidate timeout",timeout_s=timeout))
    result = read_json(output/"result.json") if (output/"result.json").exists() else dict(status="failed",returncode=p.returncode)
    result["process_wall_s"] = time.perf_counter()-start
    atomic_json(output/"result.json",result)
    print(json.dumps(dict(stage=name,status=result["status"],seconds=result["process_wall_s"])),flush=True)
    return result


def coordinate(args):
    cfg = read_json(args.source/"config.json")
    manifest = read_json(args.source/"manifest.json")
    state = read_json(args.source/"status.json")
    from cognitive_ultrasound.task_budget.suite import process_alive
    if process_alive(state.get("pid"),state.get("created")) or process_alive(state.get("worker_pid"),state.get("worker_created")):
        raise RuntimeError("Pause research batch before calibration; no timing under competing research jobs")
    args.output.mkdir(parents=True,exist_ok=True)
    from cognitive_ultrasound.hardware import snapshot
    from cognitive_ultrasound.config import ROOT
    fingerprint=dict(hardware=snapshot(),source_identity=sha256(args.source/"identity.json"),
                     source_files={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"src/cognitive_ultrasound/task_budget").glob("*.py"))},
                     environment={k:os.environ.get(k) for k in ("LD_LIBRARY_PATH","NVIDIA_TF32_OVERRIDE","CUBLAS_WORKSPACE_CONFIG")})
    cache_file=args.output/"calibration_cache_identity.json"
    if cache_file.exists() and read_json(cache_file)!=fingerprint:
        raise RuntimeError("Hardware/source/environment changed: use a new calibration output directory")
    atomic_json(cache_file,fingerprint)
    atomic_json(args.output/"identity.json",dict(source=str(args.source),source_identity=sha256(args.source/"identity.json"),
                commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                scientific_config=cfg, decision="user decides after report; never auto resumes"))
    profiles = [("baseline",4,"compressed","eager"),
                *[(f"uncompressed_t{t}",t,"uncompressed","eager") for t in (1,2,4,8)],
                ("tmpfs_t4",4,"tmpfs","eager")]
    screens = []
    reference = None
    for name,threads,ipc,execution in profiles:
        c=copy.deepcopy(cfg);c["runtime"].update(threads=threads,ipc_mode=ipc,ef_execution=execution,evaluation_workers=1)
        result=child(args,name,c,"ef",180)
        if result["status"]=="completed":
            arrays=npz(args.output/name/"arrays.npz")
            if reference is None:reference=arrays
            result["verification"]=compare(reference,arrays)
            result["selection_identical"]=np.array_equal(reference["lines"],arrays["lines"])
        result["name"]=name;screens.append(result)
        atomic_json(args.output/"screening.json",screens)
    valid=[r for r in screens if r.get("verification",{}).get("passed") and r.get("selection_identical")]
    if not valid:raise RuntimeError("No validated EF configuration")
    best=min(valid,key=lambda r:r["mean_s"])
    chosen=copy.deepcopy(cfg);chosen["runtime"].update(best["runtime"])
    comp=copy.deepcopy(chosen);comp["runtime"]["ef_execution"]="compile"
    result=child(args,"ef_compile",comp,"ef",240)
    if result["status"]=="completed":
        arrays=npz(args.output/"ef_compile/arrays.npz")
        result["verification"]=compare(reference,arrays)
        result["selection_identical"]=np.array_equal(reference["lines"],arrays["lines"])
        if result["verification"]["passed"] and result["selection_identical"] and result["mean_s"]<best["mean_s"]:
            chosen=comp
    result["name"]="ef_compile";screens.append(result);atomic_json(args.output/"screening.json",screens)
    serial=child(args,"loop_baseline",cfg,"rollout",600)
    fast=child(args,"loop_fast",chosen,"rollout",600)
    loop_check=compare(npz(args.output/"loop_baseline/arrays.npz"),npz(args.output/"loop_fast/arrays.npz")) if serial["status"]==fast["status"]=="completed" else dict(passed=False)
    if loop_check["passed"]:
        loop_check["actions_identical"] = read_json(args.output/"loop_baseline/discrete.json")==read_json(args.output/"loop_fast/discrete.json")
    parallel_cfg=copy.deepcopy(chosen);parallel_cfg["runtime"]["evaluation_workers"]=2
    parallel=child(args,"loop_parallel2",parallel_cfg,"rollout",600)
    parallel_check=compare(npz(args.output/"loop_baseline/arrays.npz"),npz(args.output/"loop_parallel2/arrays.npz")) if parallel["status"]==serial["status"]=="completed" else dict(passed=False)
    if parallel_check["passed"]:
        parallel_check["actions_identical"] = read_json(args.output/"loop_baseline/discrete.json")==read_json(args.output/"loop_parallel2/discrete.json")
    gs=child(args,"gs_eager_vs_jit",chosen,"gs",900)
    recommendation=copy.deepcopy(chosen)
    if parallel_check.get("passed") and parallel_check.get("actions_identical") and parallel.get("process_work_s",1e9)<fast.get("process_work_s",0):
        recommendation["runtime"]["evaluation_workers"]=2
    if gs.get("verification",{}).get("passed") and gs.get("changed_inputs_verification",{}).get("passed") and gs["jit"]["mean_s"]<gs["eager"]["mean_s"]:
        recommendation["runtime"]["gs_execution"]="jit"
    e2=child(args,"E2_fast_two_updates",recommendation,"train_E2",360)
    e1=child(args,"E1_fast_two_updates",recommendation,"train_E1",1500)
    atomic_json(args.output/"recommendation.json",dict(config=recommendation,adopted=False,
                condition="Whole-loop and training equivalence still govern reuse; user decides whether to run"))
    atomic_json(args.output/"result.json",dict(status="completed",screens=screens,baseline=serial,fast=fast,
                fast_verification=loop_check,parallel=parallel,parallel_verification=parallel_check,gs=gs,
                training=dict(E1=e1,E2=e2),
                decision="not resumed; user decides",production_results_untouched=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--action",choices=("coordinate","ef","rollout","gs","train_E1","train_E2"),default="coordinate")
    parser.add_argument("--trial-config",type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.action=="coordinate":return coordinate(args)
    cfg=read_json(args.trial_config);manifest=read_json(args.source/"manifest.json")
    try:
        if args.action=="ef":
            folders=sorted(args.source.glob("jobs/*/.ipc/input.npz"),key=lambda p:p.stat().st_mtime,reverse=True)
            result=ef_trial(cfg,args.output,folders[0],folders[0].parent/"coordinates.npz")
        elif args.action=="rollout":result=rollout_trial(cfg,manifest,args.output)
        elif args.action=="gs":result=gs_trial(cfg,manifest,args.output)
        else:result=training_trial(cfg,manifest,args.output,args.source,args.action.removeprefix("train_"))
        atomic_json(args.output/"result.json",result)
    except Exception as error:
        atomic_json(args.output/"result.json",dict(status="failed",error=f"{type(error).__name__}: {error}"))
        raise


if __name__=="__main__":main()
