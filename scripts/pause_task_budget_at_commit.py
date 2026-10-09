"""Pause one owned coordinator after its worker's next atomic update/case commit."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import psutil

p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
r=args.output.resolve();s=json.loads((r/"status.json").read_text())
coordinator=psutil.Process(s["pid"])
if abs(coordinator.create_time()-s["created"])>.05:raise RuntimeError("Coordinator ownership changed")
if s.get("status")!="running":raise RuntimeError("Batch is not running")
job=r/"jobs"/s["job"]
pattern="updates/*.npz" if s["job"].endswith("_train") else "*/complete.json"
before={str(f) for f in job.glob(pattern)}
os.kill(coordinator.pid,signal.SIGSTOP)
try:
    deadline=time.monotonic()+420
    while time.monotonic()<deadline:
        after={str(f) for f in job.glob(pattern)}
        if after-before or (job/"result.json").exists():break
        try:
            w=psutil.Process(s["worker_pid"])
            if abs(w.create_time()-s["worker_created"])>.05 or w.status()==psutil.STATUS_ZOMBIE:break
        except psutil.NoSuchProcess:break
        time.sleep(.25)
    (r/"STOP").write_text("User requested diagnostic pause after next committed result\n")
    print(json.dumps(dict(job=s["job"],committed_new=sorted(after-before),stop_requested=True)),flush=True)
finally:
    if coordinator.is_running():os.kill(coordinator.pid,signal.SIGCONT)
