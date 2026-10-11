"""Finite experiment-local watcher; close at136 without editing a live worker's source."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import psutil

from cognitive_ultrasound.preparation.common import atomic_json, digest, read_json
from cognitive_ultrasound.preparation.suite import run_lock
from cognitive_ultrasound.provenance import sha256
from cognitive_ultrasound.task_budget.stage_protocol import ARMS
from cognitive_ultrasound.task_budget.suite import process_alive, terminate_owned_worker
from cognitive_ultrasound.task_budget.value_storage import committed

BOUNDARY = "full_pass_2_evaluate"
CANCELLED = [
    "full_pass_4_train",
    "full_pass_4_evaluate",
    "select",
    "confirmation_baselines",
    "confirmation_policies",
]


def latest_updates(root):
    return {
        arm: max(
            (
                int(p.parent.name)
                for p in (root / "training" / arm / "updates").glob("*/complete.json")
            ),
            default=0,
        )
        for arm in ARMS
    }


def boundary_ready(root):
    batch = digest(read_json(root / "identity.json"))
    value = committed(root / "jobs" / BOUNDARY, dict(batch=batch, phase=BOUNDARY))
    if value is None:
        return False
    names = read_json(root / "manifest.json")["cohorts"]["development"]
    expected = {(arm, name, 42) for arm in ARMS for name in names}
    actual = {(x["arm"], x["case"], x["seed"]) for x in value["cases"]}
    if value["status"] != "completed" or len(value["cases"]) != len(expected) or actual != expected:
        raise ValueError("Incomplete/mismatched136-case development boundary")
    if any(x["update"] != 136 for x in value["cases"]):
        raise ValueError("Wrong development checkpoint at the stop boundary")
    for arm in ARMS:
        cp = committed(
            root / "training" / arm / "updates" / "00136", dict(batch=batch, arm=arm, update=136)
        )
        if cp is None or cp["update"] != 136:
            raise ValueError(f"Missing/corrupted {arm} checkpoint136")
    return True


def request_stop(root):
    (root / "STOP").write_text(
        "User requested close after136 updates and development evaluation.\n", encoding="utf-8"
    )


def wait_stopped(root, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        s = read_json(root / "status.json")
        if not process_alive(s.get("pid"), s.get("created")) and not process_alive(
            s.get("worker_pid"), s.get("worker_created")
        ):
            return dict(coordinator_exited=True, forced=False)
        time.sleep(0.2)
    # Only the batch's fingerprinted processes may be terminated.
    s = read_json(root / "status.json")
    terminate_owned_worker(s)
    if process_alive(s.get("pid"), s.get("created")):
        p = psutil.Process(s["pid"])
        p.terminate()
        try:
            p.wait(10)
        except psutil.TimeoutExpired:
            p.kill()
            p.wait(10)
    return dict(coordinator_exited=True, forced=True)


def export(root, stop_result):
    if not boundary_ready(root):
        raise ValueError("Cannot call an incomplete136 batch completed")
    manifest = read_json(root / "manifest.json")
    state = read_json(root / "status.json")
    scope = dict(
        status="completed",
        scope="three arms to136 updates plus16-case development",
        target_updates=136,
        selected_checkpoints={arm: 136 for arm in ARMS},
        latest_preserved_updates=latest_updates(root),
        cancelled_phases=CANCELLED,
        confirmation_evaluated=False,
        matched_periodic_at136_evaluated=False,
        original_coordinator_status=state["status"],
        finished_epoch=time.time(),
        original_failures=state.get("failures", []),
        original_error=state.get("error"),
        reason="User explicitly chose136 instead of264",
        **stop_result,
    )
    atomic_json(root / "scope_closure_136.json", scope)
    text = [
        "# 动态预算136次更新收尾报告",
        "",
        "用户已将本轮范围缩短为T1／Q2／T2各136次更新及16病例开发评价。",
        "264次训练、检查点选择及32病例双种子确认已取消；没有新增GPU比较实验。",
        "原协调器status记录原完整队列的结束方式；scope_closure_136.json=completed表示经产物校验，用户指定的缩减范围完成。原状态或错误没有覆盖。",
        "控制器、Adam、奖励EMA、对偶变量与所有历史结果保留。没有修改模型源码／配置／identity，没有从零重训。",
        "",
        "|检查点|方法|开发病例|EF MAE↓|平均线数|相对固定M的MSE|p90变化|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for phase in ["full_pass_1_evaluate", BOUNDARY]:
        rows = read_json(root / "jobs" / phase / "result.json")["cases"]
        for arm in ARMS:
            a = [x for x in rows if x["arm"] == arm]
            if not a:
                continue
            text.append(
                f"|{a[0]['update']}|{arm}|{len(a)}|{np.mean([x['ef_error'] for x in a]):.4f}|"
                f"{np.mean([x['mean_lines'] for x in a]):.3f}|{np.mean([x['mse_ratio'] for x in a]):.2%}|"
                f"{np.mean([x['p90_ratio'] for x in a]):.2%}|"
            )
    text.extend(
        [
            "",
            f"开发名单共{len(manifest['cohorts']['development'])}例；当前只使用评价种子42。",
            "第136次模型的同资源周期对照未运行，因此上述固定参照差异不能全部归因于学习了采集时机。小规模第16次的周期结果仍保留。",
            "这是一份探索性阶段参照，不证明收敛、临床质量保持或最终泛化。Q2重建代价与EF结果分别解释，EF实验不以MSE变化判失败。",
            "粗动作档位与精确视频总预算配额均沿用本轮协议；后续改变设置应使用独立版本及可比资源对照。",
            "分析包不包含帧数组、GRU／Adam检查点或预训练权重；这些证据留在服务器。",
        ]
    )
    (root / "SCOPE_136_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    files = [
        root / n
        for n in [
            "SCOPE_136_REPORT.md",
            "scope_closure_136.json",
            "REPORT_中文.md",
            "PILOT_REPORT.md",
            "summary.json",
            "evaluation_cases.csv",
            "manifest.json",
            "config.json",
            "requested_config.json",
            "identity.json",
            "plan.json",
            "status.json",
            "pilot_gate.json",
        ]
        if (root / n).exists()
    ]
    files.extend(
        p for p in sorted((root / "jobs").glob("*/result.json")) if p.parent.name not in CANCELLED
    )
    files.extend(
        p
        for p in sorted((root / "training").glob("*/updates/*/record.json"))
        if int(p.parent.name) <= 136
    )
    files.extend(sorted((root / "evaluations").glob("*/budget_trace.csv")))
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    target = root.parent / f"{root.name}.at136.analysis_{time.time_ns()}.tar.gz"
    partial = target.with_suffix(".partial")
    with tarfile.open(partial, "w:gz") as t:
        for p in files:
            t.add(p, arcname=str(Path(root.name) / p.relative_to(root)), recursive=False)
    with tarfile.open(partial, "r:gz") as t:
        for member in t:
            name = str(Path(member.name).relative_to(root.name))
            if hashlib.sha256(t.extractfile(member).read()).hexdigest() != hashes[name]:
                raise AssertionError("136 analysis archive verification failed")
    partial.replace(target)
    receipt = dict(
        archive=str(target),
        bytes=target.stat().st_size,
        sha256=sha256(target),
        verified=True,
        source_hashes=hashes,
        scope="user-authorized136 closure; no frames or weights",
    )
    atomic_json(root / "scope_136_bundle_receipt.json", receipt)
    return receipt


def watch(root):
    root = Path(root).resolve()
    control = root / ".finish_136"
    control.mkdir(exist_ok=True)
    with run_lock(control):
        if (root / "scope_136_bundle_receipt.json").exists():
            receipt = read_json(root / "scope_136_bundle_receipt.json")
            if not receipt["verified"] or sha256(receipt["archive"]) != receipt["sha256"]:
                raise ValueError("Existing136 archive corrupted")
            print(json.dumps(dict(event="ALREADY_CLOSED", **receipt)), flush=True)
            return
        meta = dict(
            status="watching",
            pid=os.getpid(),
            created=psutil.Process().create_time(),
            started_epoch=time.time(),
            boundary=BOUNDARY,
            target_updates=136,
            cancelled_phases=CANCELLED,
        )
        atomic_json(control / "status.json", meta)
        try:
            last = None
            while True:
                s = read_json(root / "status.json")
                ready = boundary_ready(root)
                if ready or s.get("phase") in CANCELLED:
                    request_stop(root)
                    stop_result = wait_stopped(root)
                    receipt = export(root, stop_result)
                    meta.update(status="completed", finished_epoch=time.time(), receipt=receipt)
                    atomic_json(control / "status.json", meta)
                    print(json.dumps(dict(event="CLOSED_AT136", **receipt)), flush=True)
                    return
                if s["status"] in ["failed", "stopped", "completed"] or not process_alive(
                    s.get("pid"), s.get("created")
                ):
                    raise RuntimeError(
                        "Original queue ended before136 development completion; preserve data, no automatic resume"
                    )
                key = (s.get("phase"), s.get("progress", {}).get("update"))
                if key != last:
                    meta.update(original_phase=s.get("phase"), last_seen_epoch=time.time())
                    atomic_json(control / "status.json", meta)
                    print(json.dumps(dict(event="WAIT_FOR136", phase=s.get("phase"))), flush=True)
                    last = key
                time.sleep(0.2)
        except BaseException as error:
            # Fail closed: never silently let a broken stop watcher authorize264.
            request_stop(root)
            meta.update(status="failed", error=f"{type(error).__name__}: {error}")
            atomic_json(control / "status.json", meta)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    watch(args.output)
