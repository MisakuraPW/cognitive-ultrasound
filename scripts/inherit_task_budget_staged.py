"""Audited reuse into a new runtime identity; never modify historical batch identities."""

import argparse
import ast
import copy
import hashlib
from pathlib import Path
import shutil
import subprocess

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.preparation.suite import run_lock
from cognitive_ultrasound.provenance import sha256
from cognitive_ultrasound.task_budget.data import configuration, lock_manifest
from cognitive_ultrasound.task_budget.suite import identity, process_alive


def source_at(path):
    return subprocess.check_output(["git","show",f"fa779d3:{path}"],cwd=ROOT)


def function(text,name):
    return next(n for n in ast.parse(text).body if getattr(n,"name",None)==name)


def equal(a,b):
    return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def audit(source_identity):
    allowed={f"src/cognitive_ultrasound/task_budget/{x}.py" for x in
             ("__main__","suite","task","ef_worker","episode","experiment")}
    for path,h in source_identity["source"].items():
        if hashlib.sha256(source_at(path)).hexdigest()!=h:
            raise ValueError("Unexpected original source snapshot: "+path)
        if path not in allowed and sha256(ROOT/path)!=h:
            raise ValueError("Unreviewed scientific dependency change: "+path)
    ep="src/cognitive_ultrasound/task_budget/episode.py"
    old=source_at(ep).decode();new=(ROOT/ep).read_text(encoding="utf-8")
    for name in ("rollout","gs_local_objective","projected_observation"):
        if not equal(function(old,name),function(new,name)):
            raise ValueError("Scientific rollout/objective changed: "+name)
    class DropDisabledJit(ast.NodeTransformer):
        def visit_If(self,node):
            if "gs_execution" in ast.unparse(node.test):
                return node.orelse
            return self.generic_visit(node)
    if not equal(function(old,"gs_gradient"),DropDisabledJit().visit(function(new,"gs_gradient"))):
        raise ValueError("Original eager GS gradient changed")
    exp="src/cognitive_ultrasound/task_budget/experiment.py"
    old=source_at(exp).decode();new=(ROOT/exp).read_text(encoding="utf-8")
    a=function(old,"train");b=function(new,"train")
    if [x.arg for x in b.args.args[-2:]]!=["end_update","audit_callback"]:
        raise ValueError("Unexpected training wrapper")
    b.args.args=b.args.args[:-2];b.args.defaults=b.args.defaults[:-2]
    old_iter=next(n.iter for n in ast.walk(a) if isinstance(n,ast.For) and ast.unparse(n.target)=="update")
    class DropAudit(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(x,ast.Name) and x.id=="final_update" for x in node.targets):return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if "audit_callback" in ast.unparse(node.test):return None
            return self.generic_visit(node)
        def visit_For(self,node):
            if ast.unparse(node.target)=="update":node.iter=copy.deepcopy(old_iter)
            return self.generic_visit(node)
    if not equal(a,DropAudit().visit(b)):
        raise ValueError("Training/RNG/optimizer algorithm changed")
    ef="src/cognitive_ultrasound/task_budget/ef_worker.py"
    a=function(source_at(ef).decode(),"EFModel");b=function((ROOT/ef).read_text(),"EFModel")
    for name in ("preprocess","evaluate"):
        aa=next(n for n in a.body if getattr(n,"name",None)==name)
        bb=next(n for n in b.body if getattr(n,"name",None)==name)
        if not equal(aa,bb):raise ValueError("EF preprocessing/gradient changed")
    task="src/cognitive_ultrasound/task_budget/task.py"
    a=function(source_at(task).decode(),"EFService");b=function((ROOT/task).read_text(),"EFService")
    for name in ("score","video"):
        if not equal(next(n for n in a.body if getattr(n,"name",None)==name),next(n for n in b.body if getattr(n,"name",None)==name)):
            raise ValueError("EF scoring/video aggregation changed")
    return dict(rollout_objective_ast_identical=True,eager_gs_ast_identical=True,
                training_rng_optimizer_ast_identical=True,ef_preprocess_gradient_ast_identical=True,
                case_scoring_aggregation_ast_identical=True,transport="bitwise engineering validation")


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True)
    p.add_argument("--engineering",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",default="configs/task_budget_ef_staged.yaml");args=p.parse_args()
    source,out=args.source.resolve(),args.output.resolve()
    if out.exists() or source==out:raise ValueError("New absent output required")
    cfg=configuration(args.config);old_cfg=read_json(source/"config.json")
    science=lambda c:{k:v for k,v in c.items() if k not in ("runtime","execution")}
    if science(cfg)!=science(old_cfg):raise ValueError("Scientific points or training schedule changed")
    if any(cfg["runtime"].get(k)!=v for k,v in dict(threads=2,ipc_mode="uncompressed",ef_execution="eager",gs_execution="eager",evaluation_workers=1).items()):
        raise ValueError("Only validated engineering profile is eligible")
    evidence=read_json(args.engineering/"result.json")
    if not evidence["fast_verification"]["bitwise"] or not evidence["fast_verification"]["actions_identical"]:
        raise ValueError("Runtime closed-loop validation not passed")
    controls=read_json(args.engineering/"training_equivalence.json")
    if not controls["E2"]["passed"]:raise ValueError("E2 training equivalence not passed")
    with run_lock(source):
        state=read_json(source/"status.json")
        if process_alive(state.get("pid"),state.get("created")) or process_alive(state.get("worker_pid"),state.get("worker_created")):
            raise ValueError("Original batch must be stopped")
        original=read_json(source/"identity.json");review=audit(original)
        manifest=lock_manifest(cfg,source);current=identity(cfg,manifest)
        for key in ("manifest","checkpoint","ef_weights","ef_stats","hardware","python","task_environment","upstream"):
            if original[key]!=current[key]:raise ValueError("Assets/environment changed: "+key)
        out.mkdir(parents=True);copied={};cases=0
        def preserve(file,target):
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
            if sha256(file)!=sha256(target):raise ValueError("Copy validation failed")
            copied[str(target.relative_to(out))]=sha256(target)
        preserve(source/"manifest.json",out/"manifest.json")
        atomic_json(out/"config.json",cfg)
        for job in (source/"jobs").iterdir():
            if not job.is_dir():continue
            for complete in job.glob("*/complete.json"):
                r=read_json(complete)
                if r["case"] not in manifest["files"] or r["frames"]!=manifest["files"][r["case"]]["frames"]:
                    raise ValueError("Only full committed case results may be reused")
                for f in complete.parent.iterdir():
                    if f.is_file() and f.suffix!=".tmp":preserve(f,out/"jobs"/job.name/complete.parent.name/f.name)
                cases+=1
            for n in ("result.json","task.json"):
                file=job/n
                if file.exists() and (n=="task.json" or read_json(file).get("status")=="completed"):
                    preserve(file,out/"jobs"/job.name/n)
            for f in (job/"updates").glob("*"):
                if f.suffix in (".npz",".json"):preserve(f,out/"jobs"/job.name/"updates"/f.name)
        # Both are EXACT original scientific training prefixes, not failed JIT weights.
        for method in ("E1","E2"):
            shadow=args.engineering/"final_controls"/(method+"_original")
            c=read_json(shadow/"trial_config.json")
            if science(c)!=science(old_cfg):raise ValueError("Shadow training science differs")
            if read_json(shadow/"result.json").get("status")!="completed":raise ValueError("Incomplete shadow prefix")
            target=out/"jobs"/(method+"_l0_s42_train")/"updates"
            for f in (shadow/"updates").glob("*"):
                if f.suffix not in (".npz",".json"):continue
                if (target/f.name).exists():
                    if sha256(f)!=sha256(target/f.name):raise ValueError("Conflicting overlapping checkpoint")
                else:preserve(f,target/f.name)
        for file in source.glob(".cache/full_ef_reference/*.json"):
            preserve(file,out/".cache/full_ef_reference"/file.name)
        for n in ("identity.json","status.json","config.json"):
            preserve(source/n,out/"lineage"/("original_"+n))
        preserve(args.engineering/"training_equivalence.json",out/"lineage/engineering_training.json")
        preserve(args.engineering/"qualified_recommendation.json",out/"lineage/qualified_runtime.json")
        atomic_json(out/"inheritance.json",dict(source=str(source),review=review,copied_files=copied,
                    full_cases=cases,committed_prefix_steps=dict(E1=2,E2=11),
                    scientific_configuration_unchanged=True,original_results_untouched=True,
                    excluded="EF compile and GS JIT states; incomplete cases;64-frame evaluations",
                    timing_note="Historical timing retained under original runtime; new timings separately traceable"))
        print(f"Inherited {cases} full cases, E1 steps1-2, E2 steps1-11; no old result was overwritten.")


if __name__=="__main__":main()
