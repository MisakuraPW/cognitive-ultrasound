"""Frozen final controls:2 updates/arm/runtime and2 full videos, no parameter search."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from cognitive_ultrasound.preparation.common import atomic_json, read_json
from calibrate_task_budget_ef import child, compare, npz


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    root=args.output
    source=Path(read_json(root/"identity.json")["source"])
    base=read_json(source/"config.json")
    fast=read_json(root/"recommendation.json")["config"]
    controls=argparse.Namespace(source=source,output=root/"final_controls",frames=64)
    controls.output.mkdir(exist_ok=True)
    validations={}
    for method in ("E2","E1"):
        slow_cfg=copy.deepcopy(base);slow_cfg["runtime"]["audit_training"]=True
        fast_cfg=copy.deepcopy(fast);fast_cfg["runtime"]["audit_training"]=True
        a=child(controls,method+"_original",slow_cfg,"train_"+method,900)
        b=child(controls,method+"_optimized",fast_cfg,"train_"+method,900)
        if a["status"]!= "completed" or b["status"]!="completed":
            validations[method]=dict(passed=False,reason="real-update control incomplete",original=a,optimized=b)
            continue
        original=controls.output/(method+"_original")
        optimized=controls.output/(method+"_optimized")
        checks=[]
        for file in sorted((original/"audit").glob("*.npz")):
            x=npz(file);y=npz(optimized/"audit"/file.name)
            audit=compare(x,y)
            audit["actions_identical"]=all(np.array_equal(x[k],y[k]) for k in ("first_mask","second_mask","budgets"))
            state=compare(npz(original/"updates"/file.name),npz(optimized/"updates"/file.name))
            checks.append(dict(update=int(file.stem),forward_gradient_actions=audit,parameters_optimizer_baseline=state))
        validations[method]=dict(passed=bool(checks) and all(x["forward_gradient_actions"]["passed"] and
            x["forward_gradient_actions"]["actions_identical"] and x["parameters_optimizer_baseline"]["passed"] for x in checks),
            original=a,optimized=b,checks=checks)
        atomic_json(root/"training_equivalence.json",validations)
    controls.frames=0
    serial=copy.deepcopy(fast);serial["runtime"]["evaluation_workers"]=1
    parallel=copy.deepcopy(fast);parallel["runtime"]["evaluation_workers"]=2
    s=child(controls,"full_serial",serial,"rollout",600)
    t=child(controls,"full_parallel2",parallel,"rollout",600)
    verdict=dict(passed=False)
    if s["status"]==t["status"]=="completed":
        verdict=compare(npz(controls.output/"full_serial/arrays.npz"),npz(controls.output/"full_parallel2/arrays.npz"))
        verdict["actions_identical"]=read_json(controls.output/"full_serial/discrete.json")==read_json(controls.output/"full_parallel2/discrete.json")
    atomic_json(root/"full_video_verification.json",dict(serial=s,parallel=t,verification=verdict))
    print(json.dumps(dict(training={k:v["passed"] for k,v in validations.items()},full_video=verdict),indent=2),flush=True)


if __name__=="__main__":main()
