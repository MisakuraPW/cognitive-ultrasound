"""Sequential umbrella for closure-v4 and the Torch comparison; no new experiments."""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from .config import ROOT, load, path
from .preparation.common import atomic_json, read_json
from .preparation.suite import run_lock

MODULES = ["cognitive_ultrasound.preparation.closure", "cognitive_ultrasound.torch_casl"]


def stages_from(file):
    stages = load(path(file))["stages"]
    if [s["module"] for s in stages] != MODULES or [s["name"] for s in stages] != [
        "closure",
        "torch",
    ]:
        raise ValueError("Only the two fixed preparation stages are allowed")
    for s in stages:
        s["config"] = str(path(s["config"]).resolve())
        s["output"] = str(path(s["output"]).resolve())
    a, b = [Path(s["output"]) for s in stages]
    if a == b or a in b.parents or b in a.parents:
        raise ValueError("Stage output directories must be independent")
    return stages


def state_at(output):
    file = Path(output) / "status.json"
    return read_json(file) if file.exists() else {}


def is_finished(output):
    output = Path(output)
    state = state_at(output)
    if (output / "STOP").exists() or state.get("status") not in ("completed", "finished_with_gaps"):
        return False
    for job in state.get("jobs", {}).values():
        if job.get("status") in ("running", "paused", "stop_requested", "interrupted"):
            return False
        if "Stopped by request" in str(job.get("reason", "")):
            return False
    return (output / "REPORT.md").exists()


def assert_idle(state):
    records = [state, *state.get("jobs", {}).values(), *state.get("stages", {}).values()]
    for item in records:
        pid = item.get("worker_pid", item.get("pid"))
        created = item.get("worker_created", item.get("created"))
        if not pid or created is None:
            continue
        try:
            p = psutil.Process(pid)
            if abs(p.create_time() - created) < 0.01 and p.status() != psutil.STATUS_ZOMBIE:
                raise RuntimeError(
                    f"Existing coordinator/worker {pid} is still running; wait for it"
                )
        except psutil.NoSuchProcess:
            pass


def report(output, stages):
    lines = [
        "# 全部剩余准备实验：总览",
        "",
        "固定顺序：closure-v4 → PyTorch CASL 对照。已完成的 v1/v3 不重跑。",
        "子任务累计计算上限合计 4.5 小时；初始化、检查、报告与打包另计。不是运行完成时间保证。",
        "",
        "| 阶段 | 状态 | 详细报告 | 结果包 |",
        "|---|---|---|---|",
    ]
    for stage in stages:
        folder = Path(stage["output"])
        status = state_at(folder).get("status", "not_started")
        archive = folder.parent / (folder.name + ".results.tar.gz")
        archive_text = str(archive) if archive.exists() else "尚未生成"
        lines.append(f"| {stage['name']} | {status} | {folder / 'REPORT.md'} | {archive_text} |")
    lines += [
        "",
        "completed 表示执行完成；finished_with_gaps 要查看缺资产、负结果或超限原因。",
        "某阶段失败后继续尝试另一个独立阶段；暂停会阻止后续阶段启动。",
        "实例不会自动关机。两个子结果包各下载一次，无需重新下载旧 v1/v3 结果。",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def child_command(stage, action="run", retry_failed=False):
    args = [sys.executable, "-m", stage["module"], action, "--output", stage["output"]]
    if action == "run":
        args += ["--config", stage["config"]]
        if (Path(stage["output"]) / "identity.json").exists():
            args += ["--resume"]
        if retry_failed and stage["name"] == "closure":
            args += ["--retry-failed"]
    return args


def run(output, stages, resume=False, retry_failed=False):
    output.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        child = Path(stage["output"])
        if output == child or output in child.parents or child in output.parents:
            raise ValueError("Controller and child outputs must be independent")
    with run_lock(output):
        previous = state_at(output)
        if previous.get("plan") and previous["plan"] != stages:
            raise ValueError("Existing controller has another plan; use a new controller directory")
        assert_idle(previous)
        for stage in stages:
            child = Path(stage["output"])
            assert_idle(state_at(child))
            child.mkdir(parents=True, exist_ok=True)
            # Probe the child lock before any stage starts; never race an independent job.
            with run_lock(child):
                pass
        if resume:
            (output / "STOP").unlink(missing_ok=True)
            for stage in stages:
                (Path(stage["output"]) / "STOP").unlink(missing_ok=True)
        elif (output / "STOP").exists() or any(
            (Path(s["output"]) / "STOP").exists() for s in stages
        ):
            raise RuntimeError("Explicitly paused; use run_all_preparation.sh resume")
        state = dict(plan=stages, status="running", stages={})
        atomic_json(output / "status.json", state)
        for index, stage in enumerate(stages, 1):
            name = stage["name"]
            child = Path(stage["output"])
            if (output / "STOP").exists():
                state["status"] = "paused"
                break
            done = is_finished(child)
            archive = child.parent / (child.name + ".results.tar.gz")
            action = (
                "report"
                if done and (not archive.exists() or not Path(str(archive) + ".sha256").exists())
                else "run"
            )
            if done and action == "run":
                state["stages"][name] = {
                    "status": state_at(child)["status"],
                    "skipped_existing": True,
                }
                print(f"SKIP [{index}/2] {name}: existing terminal result", flush=True)
                atomic_json(output / "status.json", state)
                continue
            command = child_command(stage, action, retry_failed)
            print(f"STAGE [{index}/2]: {name} ({action}) -> {child}", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"
            record = state["stages"][name] = {"status": "running", "action": action}
            with (output / f"{name}.console.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                def forward_log():
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(f"[{name}] {line}", end="", flush=True)

                forwarding = threading.Thread(target=forward_log, daemon=True)
                forwarding.start()
                record.update(pid=process.pid, created=psutil.Process(process.pid).create_time())
                atomic_json(output / "status.json", state)
                try:
                    while process.poll() is None:
                        if (output / "STOP").exists():
                            (
                                child / "STOP"
                            ).touch()  # child's coordinator stops its own GPU workers
                        time.sleep(1)
                except BaseException:
                    (child / "STOP").touch()
                    process.wait()
                    raise
                finally:
                    forwarding.join(timeout=5)
            record.update(
                returncode=process.returncode, status=state_at(child).get("status", "failed")
            )
            if process.returncode or not is_finished(child):
                record["status"] = "failed"
                atomic_json(output / "status.json", state)
                # Another independent launch may have won the child's lock after our probe.
                # Do not then launch the other suite alongside its live GPU worker.
                assert_idle(state_at(child))
            if (output / "STOP").exists() or (child / "STOP").exists():
                record["status"] = "paused"
                state["status"] = "paused"
                atomic_json(output / "status.json", state)
                break
            atomic_json(output / "status.json", state)
            print(f"STAGE_END [{index}/2]: {name} {record['status']}", flush=True)
            report(output, stages)
        if state["status"] != "paused":
            state["status"] = (
                "completed"
                if all(s["status"] == "completed" for s in state["stages"].values())
                else "finished_with_gaps"
            )
        atomic_json(output / "status.json", state)
        report(output, stages)
        print(f"ALL_PREPARATION_END: {state['status']}; {output / 'REPORT.md'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["run", "plan", "report"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", default="configs/all_preparation.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    a = p.parse_args()
    stages = stages_from(a.config)
    if a.action == "plan":
        print("closure-v4 (3h cap) -> Torch/JAX comparison (90min cap); completed stages skipped")
        for stage in stages:
            print(stage)
    elif a.action == "report":
        report(a.output, stages)
    else:
        run(
            a.output.resolve(),
            stages,
            resume=a.resume or os.environ.get("ALL_PREPARATION_RESUME") == "1",
            retry_failed=a.retry_failed,
        )


if __name__ == "__main__":
    main()
