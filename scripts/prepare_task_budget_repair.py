"""Prepare a separate learning version using existing TRAIN traces and E0 records."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.provenance import sha256
from cognitive_ultrasound.task_budget.data import configuration, lock_manifest
from cognitive_ultrasound.task_budget.suite import identity, process_alive


def fit_scales(traces):
    if not traces:raise ValueError("No training feature traces")
    x=np.concatenate(traces)
    if x.ndim!=2 or x.shape[1]!=12 or not np.isfinite(x).all():raise ValueError("Invalid training states")
    scale=np.quantile(np.abs(x),.9,axis=0)
    return np.where(scale>0,scale,1.).tolist()


def prepare(source,engineering,output,config):
    cfg=configuration(config)
    if (output/"config.json").exists():
        old=read_json(output/"config.json")
        if {k:v for k,v in old.items() if k not in ("feature_scales","feature_calibration")}!=cfg:
            raise ValueError("Repair configuration changed; use a new output directory")
        return output/"config.json"
    s=read_json(source/"status.json")
    if read_json(source/"config.json").get("functional_fixture"):
        raise ValueError("Synthetic fixtures cannot supply scientific baseline results")
    if process_alive(s.get("pid"),s.get("created")) or process_alive(s.get("worker_pid"),s.get("worker_created")):
        raise ValueError("Old research batch must remain stopped")
    manifest=lock_manifest(cfg,source)
    allowed=set(manifest["cohorts"]["train"]);traces=[];origins=[]
    for method in ("E1","E2"):
        directory=engineering/"final_controls"/(method+"_original")
        for file in sorted((directory/"audit").glob("*.npz")):
            record=read_json(directory/"updates"/(file.stem+".json"))
            if record["case"] not in allowed:raise ValueError("Feature fit would use a non-TRAIN patient")
            with np.load(file,allow_pickle=False) as arrays:
                for name in ("state_before","state_intermediate"):traces.append(arrays[name][1:].copy())
            origins.append(dict(case=record["case"],update=record["update"],sha256=sha256(file),path=str(file)))
    cfg["feature_scales"]=fit_scales(traces)
    cfg["feature_calibration"]=dict(scope="TRAIN ONLY; no development/confirmation fit",origins=origins,
                                    transform="existing sign-log state divided by trainingp90 then sign-log compressed")
    previous=read_json(source/"identity.json");current=identity(cfg,manifest)
    for key in ("manifest","checkpoint","ef_weights","ef_stats","upstream"):
        if previous[key]!=current[key]:raise ValueError("Frozen assets/data changed: "+key)
    compatible_hardware=all(previous[k]==current[k] for k in ("hardware","task_environment"))
    output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/"manifest.json",output/"manifest.json")
    copied=[]
    if compatible_hardware:
        # E0 acquisition never reads actor features, so quality inference remains reusable.
        seed=cfg["seeds"][0];pair=cfg["budgets"]["fixed"]
        for cohort in ("development","confirmation"):
            name=f"E0_{pair[0]}_{pair[1]}_s{seed}_{cohort}";old=source/"jobs"/name;new=output/"jobs"/name
            if old.exists():
                for complete in old.glob("*/complete.json"):
                    record=read_json(complete)
                    if record["case"] not in manifest["cohorts"][cohort] or record["seed"]!=seed or record["frames"]!=manifest["files"][record["case"]]["frames"]:
                        raise ValueError("Incomplete or mismatched E0 case cannot be reused")
                    shutil.copytree(complete.parent,new/complete.parent.name,dirs_exist_ok=True)
                    copied.append(str((new/complete.parent.name/"complete.json").relative_to(output)))
                p=old/"result.json"
                if p.exists() and read_json(p).get("status")=="completed":
                    new.mkdir(parents=True,exist_ok=True);shutil.copy2(p,new/"result.json")
        for p in source.glob(".cache/full_ef_reference/*.json"):
            dest=output/".cache/full_ef_reference"/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    atomic_json(output/"config.json",cfg)
    atomic_json(output/"repair_lineage.json",dict(source=str(source),source_identity=sha256(source/"identity.json"),
                reused_e0_cases=copied,hardware_compatible=compatible_hardware,
                controller_weights_reused=False,reason="Changed state representation and learning estimator; old results preserved, never relabeled as repaired training",
                full_dps_derivative_mode="preserved with strict rejection; repaired E1 explicitly uses local_projection_v2"))
    return output/"config.json"


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--engineering",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--config",default="configs/task_budget_ef_repair.yaml")
    a=p.parse_args();print(prepare(a.source.resolve(),a.engineering.resolve(),a.output.resolve(),a.config))
