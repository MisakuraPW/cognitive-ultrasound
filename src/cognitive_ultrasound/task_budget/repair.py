"""Finite checkpointed learning repair; all model/task weights remain frozen."""
from collections import Counter
import numpy as np


def repair_jobs(cfg):
    specs=[dict(id="probe",kind="probe")]
    seed=cfg["seeds"][0];weight=cfg["training"]["lambdas"][0]
    pair=cfg["budgets"]["fixed"]
    specs.append(dict(id=f"E0_{pair[0]}_{pair[1]}_s{seed}_development",kind="evaluate",method="E0",seed=seed,fixed=pair,cohort="development"))
    specs.append(dict(id="untrained_development",kind="evaluate",method="E1",seed=seed,cohort="development",untrained=True,checkpoint_update=0,case_limit=4,**{"lambda":weight}))
    previous={}
    for update in cfg["execution"]["repair_milestones"]:
        for method in ("E1","E2"):
            name=f"{method}_u{update:04d}_train"
            s=dict(id=name,kind="train",method=method,seed=seed,end_update=update,**{"lambda":weight})
            if method in previous:s["resume_parent"]=previous[method]
            specs.append(s)
            specs.append(dict(id=f"{method}_u{update:04d}_development",kind="evaluate",method=method,seed=seed,
                              parent=name,cohort="development",checkpoint_update=update,
                              case_limit=4 if update<cfg["training"]["updates"] else cfg["cohorts"]["development"],**{"lambda":weight}))
            previous[method]=name
    specs.append(dict(id=f"E0_{pair[0]}_{pair[1]}_s{seed}_confirmation",kind="evaluate",method="E0",seed=seed,fixed=pair,cohort="confirmation"))
    for method in ("E1","E2"):
        specs.append(dict(id=method+"_confirmation",kind="evaluate",method=method,seed=seed,parent=previous[method],
                          cohort="confirmation",checkpoint_update=cfg["training"]["updates"],**{"lambda":weight}))
    return specs


def joint_budget_schedule(rows):
    """Post-hoc cyclic control matches the JOINT pair histogram, not just total lines."""
    first=(rows[0]["k1"],rows[0]["k2"])
    counts=Counter((r["k1"],r["k2"]) for r in rows[1:]);keys=sorted(counts)
    used={k:0 for k in keys};n=len(rows)-1;sequence=[first]
    for i in range(n):
        available=[k for k in keys if used[k]<counts[k]]
        k=max(available,key=lambda k:((i+1)*counts[k]/n-used[k],-keys.index(k)))
        sequence.append(k);used[k]+=1
    return np.asarray(sequence,np.int32)
