"""Bounded Git-deployed Torch/JAX experiment, isolated processes and resumable jobs."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..config import ROOT, load, path
from ..data import read_splits
from ..hardware import snapshot
from ..preparation.common import atomic_json, digest, read_json, source_identity
from ..preparation.suite import bundle_results, run_lock, stop_process
from ..provenance import command, sha256


def configuration(file):
    cfg = load(file)
    for key in ("data_root", "checkpoint", "split_manifest"):
        cfg[key] = str(path(cfg[key]).resolve())
    if (
        cfg["steps"] != [50, 25]
        or not 1 <= cfg["cases"] <= 4
        or not 3 <= cfg["frames"] <= 12
        or not 1 <= cfg["repeats"] <= 10
        or cfg["budget"] != 14
        or not 0 < cfg["job_minutes"] <= cfg["max_minutes"] <= 180
        or cfg["modes"] != ["eager", "compile", "graph"]
    ):
        raise ValueError("Outside bounded matched-work protocol")
    if read_splits(cfg["split_manifest"]) != read_splits(ROOT / "configs/splits/split.yaml"):
        raise ValueError("Expected official pinned split manifest")
    return cfg


def torch_python():
    candidates = [os.environ.get("TORCH_PYTHON"), sys.executable, "/root/miniconda3/bin/python"]
    errors = []
    probe = (
        "import json,torch,numpy,h5py,yaml,psutil; "
        'assert torch.cuda.is_available(), "no CUDA torch"; '
        'assert hasattr(torch,"compile") and hasattr(torch,"func"); '
        "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
        'gpu=torch.cuda.get_device_name(),python=__import__("sys").version)))'
    )
    for candidate in dict.fromkeys(x for x in candidates if x):
        try:
            p = subprocess.run([candidate, "-c", probe], text=True, capture_output=True, timeout=45)
            if p.returncode == 0:
                import json

                return candidate, json.loads(p.stdout.splitlines()[-1])
            errors.append(f"{candidate}: {p.stderr[-1000:]}")
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
    raise RuntimeError(
        "No usable GPU Torch interpreter. Set TORCH_PYTHON to the image Torch "
        "environment; this runner does not install/replace CUDA.\n" + "\n".join(errors)
    )


def report(output):
    import numpy as np

    status = read_json(output / "status.json") if (output / "status.json").exists() else {}
    ref = read_json(output / "reference.json") if (output / "reference.json").exists() else {}
    lines = [
        "# CASL PyTorch 算子与计算图对照",
        "",
        "冻结官方 EMA；FP32、TF32 关闭；2 粒子、W=3、14 条。50 与 25 步分开比较，DPS 每步执行。",
        "固定历史的双方输入相同，预生成随机数均不计时；JAX 已整段 JIT。",
        "这是小样本工程比较，不是论文质量复现或普遍实时性保证。",
        "",
        "| 步数 | 实现 | 稳定核心 FPS | 对 JAX 倍率 | 样本/选线核对 |",
        "|---|---|---:|---:|---|",
    ]
    comparisons = []

    def replay_passed(rows):
        warm = [r for r in rows if not r["cold"]]
        return bool(warm) and all((r.get("replay_check") or {}).get("passed", False) for r in warm)

    for steps in (50, 25):
        rr = [r for r in ref.get("rows", []) if r["steps"] == steps and not r["cold"]]
        seconds = [t for r in rr for t in r["matched_seconds"]]
        baseline = float(np.mean(seconds)) if seconds else None
        lines.append(
            f"| {steps} | JAX 原算法整段 JIT | "
            + (f"{1 / baseline:.3f}" if baseline else "未完成")
            + f" | 1 | {'重放核对通过' if replay_passed(rr) else '重放核对未通过或缺失'} |"
        )
        for mode in ("eager", "compile", "graph"):
            file = output / f"{mode}.json"
            if not file.exists():
                lines.append(f"| {steps} | Torch {mode} | 缺失 | — | 看任务状态 |")
                continue
            data = read_json(file)
            rows = [r for r in data["rows"] if r["steps"] == steps and not r["cold"]]
            ts = [t for r in rows for t in r["seconds"]]
            complete = bool(
                data["completed"] and len(rows) == len(rr) and rows and ref.get("completed")
            )
            parity = (
                complete
                and replay_passed(rr)
                and all(r["parity"]["passed"] and r["selected_equal"] for r in rows)
            )
            seconds_t = float(np.mean(ts)) if ts else None
            fps = 1 / seconds_t if seconds_t else None
            speedup = baseline / seconds_t if baseline and seconds_t else None
            comparisons.append(
                dict(
                    steps=steps,
                    mode=mode,
                    complete=complete,
                    parity=parity,
                    jax_replay_passed=replay_passed(rr),
                    p50_s=float(np.median(ts)) if ts else None,
                    p95_s=float(np.quantile(ts, 0.95)) if ts else None,
                    kernel_fps=fps,
                    speedup=speedup,
                )
            )
            lines.append(
                f"| {steps} | Torch {mode} | "
                + (f"{fps:.3f}" if fps else "—")
                + " | "
                + (f"{speedup:.2f}×" if speedup else "—")
                + f" | {'通过' if parity else '未通过或不完整'} |"
            )
    lines += [
        "",
        "JAX 随机数在图内生成与图外传入会改变 GPU 浮点融合边界；",
        "重放差异单独保存在 reference.json 的 replay_check，容差不放宽。",
        "任何该项未通过的速度只作诊断，不能认定等价加速或 32 FPS 成功。",
        "Torch parity 对照官方原轨迹；matched_parity 另对照相同外部噪声的 JAX 核心。",
        "",
        "## 原仓库适配器的实际速度",
        "",
        "| 步数 | 原仓库算法 FPS（含 RNG） | 含指标导出的适配器 FPS |",
        "|---|---:|---:|",
    ]
    for steps in (50, 25):
        original_rows = [
            r["original"]
            for r in ref.get("rows", [])
            if r["steps"] == steps and r.get("frame", 0) >= 2
        ]
        if original_rows:
            algorithm = len(original_rows) / sum(r["algorithm_s"] for r in original_rows)
            wall = len(original_rows) / sum(r["adapter_wall_s"] for r in original_rows)
            lines.append(f"| {steps} | {algorithm:.3f} | {wall:.3f} |")
    lines += ["", "此表跳过每段头两帧（冷启动与首次 warm JIT），不用于固定输入的倍速计算。"]
    lines += [
        "",
        "FPS = 计时帧数 / 同步总秒数；不对逐帧 FPS 取平均。",
        "核心 FPS 包含全部迭代、DPS、熵选线、硬投影；不含随机数生成、I/O、H2D/D2H。",
        "冷帧为 500 步，编译/图捕获与首调用独立记录，不计入稳定 FPS。",
        "",
        "## 连续闭环与 32 FPS 判据",
        "",
        "| 实现/步数 | 含传输稳定 FPS | Torch/JAX MAE | 完整冷帧与稳定核对 | 32 FPS 本批证据 |",
        "|---|---:|---|---|---|",
    ]
    for mode in ("eager", "compile", "graph"):
        file = output / f"{mode}.trajectory.json"
        if not file.exists():
            continue
        data = read_json(file)
        fixed = read_json(output / f"{mode}.json")
        for steps in (50, 25):
            rows = [r for r in data["rows"] if r["steps"] == steps and not r["cold"]]
            if not rows:
                continue
            fps = len(rows) / sum(r["host_wall_s"] for r in rows)
            a = float(np.mean([r["mae"] for r in rows]))
            b = float(np.mean([r["reference_mae"] for r in rows]))
            all_fixed = [r for r in fixed["rows"] if r["steps"] == steps]
            parity = bool(
                fixed["completed"]
                and data["completed"]
                and ref.get("completed")
                and replay_passed([r for r in ref.get("rows", []) if r["steps"] == steps])
                and all(
                    r["parity"]["passed"]
                    and r["selected_equal"]
                    and r.get("selected_count", 0) == 14
                    for r in all_fixed
                )
            )
            # Strict all-trajectory action equality; small sample is not adoption evidence.
            closed = all(r["selected_equal"] for r in data["rows"] if r["steps"] == steps)
            valid = parity and closed and a <= b * 1.01 + 1e-4
            evidence = "达到（仅本批）" if valid and fps >= 32 else "未证明"
            lines.append(
                f"| {mode}/{steps} | {fps:.3f} | {a:.6f}/{b:.6f} | "
                f"{'通过' if valid else '未通过或不完整'} | {evidence} |"
            )
    lines += [
        "",
        "连续闭环使用自身历史与选线，包含预加载帧的传输、缓冲更新和输出同步；",
        "不含磁盘读取、显示、设备采集、CPU 随机数生成。不能据此声称临床采集端到端 32 FPS。",
        "原仓库 Adapter 的实际 algorithm/adapter_wall 计时保留在 reference.json，",
        "其 CPU 指标导出开销不与上述核心口径混算。",
        "",
        "## 状态与使用决定",
        "",
        "本批不自动替换原 CASL，也不以失败的等价性检查换取加速结论。",
        "compiled/graph 的后端不支持或超时均单独记录，不冒充成功回退。",
        "",
        "```json",
        __import__("json").dumps(status, ensure_ascii=False, indent=2),
        "```",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(output / "comparison.json", comparisons)
    # Fixed diagnostic frame positions only, never choose examples by quality.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for file in sorted((output / "graph").glob("*_f002.npz"))[:2]:
        with np.load(file, allow_pickle=False) as z:
            fig, axes = plt.subplots(1, 4, figsize=(10, 3))
            for ax, key in zip(axes, ["target", "reference", "prediction", "error"]):
                image = np.abs(z["prediction"] - z["reference"]) if key == "error" else z[key]
                ax.imshow(
                    image,
                    cmap="magma" if key == "error" else "gray",
                    **({} if key == "error" else dict(vmin=-1, vmax=1)),
                )
                ax.set_title(key)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(output / (file.stem + ".png"), dpi=140)
            plt.close(fig)


def recover_interrupted(state):
    import psutil

    for job in state["jobs"].values():
        if job.get("status") != "running":
            continue
        if job.get("pid"):
            try:
                worker = psutil.Process(job["pid"])
                if (
                    abs(worker.create_time() - job["created"]) < 0.01
                    and worker.status() != psutil.STATUS_ZOMBIE
                ):
                    raise RuntimeError(
                        "Previous worker still alive; do not start a duplicate GPU job"
                    )
            except psutil.NoSuchProcess:
                pass
        state["elapsed_s"] += job.get("running_s", 0.0)
        job["status"] = "interrupted"
    return state


def run(cfg, output, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        py, torch_info = torch_python()  # Check before spending time on the JAX reference.
        import numpy as np

        cases = (
            np.random.default_rng(cfg["seed"])
            .permutation(read_splits(cfg["split_manifest"])["val"])
            .tolist()[: cfg["cases"]]
        )
        identity = dict(
            data={name: sha256(Path(cfg["data_root"]) / "val" / name) for name in cases},
            config=cfg,
            source=source_identity(),
            hardware=snapshot(),
            torch=torch_info,
            torch_python=py,
            commit=command(["git", "rev-parse", "HEAD"]),
            weights={
                n: sha256(Path(cfg["checkpoint"]) / n) for n in ("config.json", "model.weights.h5")
            },
            split_sha256=sha256(cfg["split_manifest"]),
        )
        fingerprint = digest(identity)
        state = {"jobs": {}, "elapsed_s": 0.0}
        if (output / "identity.json").exists():
            if not resume or read_json(output / "identity.json")["fingerprint"] != fingerprint:
                raise ValueError(
                    "Existing output: resume requires identical code/config/assets/device"
                )
            state = read_json(output / "status.json")
            recover_interrupted(state)
        else:
            atomic_json(output / "identity.json", dict(fingerprint=fingerprint, **identity))
            atomic_json(output / "config.json", cfg)
        for job in ["reference", *cfg["modes"]]:
            if state["jobs"].get(job, {}).get("status") == "completed":
                continue
            if (output / "STOP").exists():
                state["status"] = "paused"
                atomic_json(output / "status.json", state)
                return
            remaining = cfg["max_minutes"] * 60 - state["elapsed_s"]
            if remaining <= 0:
                state["jobs"][job] = {"status": "capped"}
                continue
            if (
                job != "reference"
                and state["jobs"].get("reference", {}).get("status") != "completed"
            ):
                state["jobs"][job] = {"status": "blocked", "reason": "reference incomplete"}
                continue
            if shutil.disk_usage(output).free < 5 * 1024**3:
                raise RuntimeError("Need 5 GiB free")
            args = [
                sys.executable if job == "reference" else py,
                "-m",
                "cognitive_ultrasound.torch_casl",
                "worker",
                "--output",
                str(output),
                "--job",
                job,
            ]
            tick = time.perf_counter()
            state["jobs"][job] = {"status": "running"}
            atomic_json(output / "status.json", state)
            print(f"STAGE: torch_casl/{job}", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"
            with (output / f"{job}.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    args,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                )
                import psutil

                state["jobs"][job].update(
                    pid=process.pid, created=psutil.Process(process.pid).create_time()
                )
                atomic_json(output / "status.json", state)
                limit = min(remaining, cfg["job_minutes"] * 60)
                last_saved = 0.0
                try:
                    while process.poll() is None:
                        elapsed = time.perf_counter() - tick
                        if (output / "STOP").exists() or elapsed >= limit:
                            stop_process(process)
                            state["jobs"][job] = {
                                "status": "paused" if (output / "STOP").exists() else "capped"
                            }
                            break
                        if elapsed - last_saved >= 5:
                            state["jobs"][job]["running_s"] = elapsed
                            atomic_json(output / "status.json", state)
                            last_saved = elapsed
                        time.sleep(1)
                    else:
                        state["jobs"][job] = {
                            "status": "completed" if process.returncode == 0 else "failed",
                            "returncode": process.returncode,
                        }
                finally:
                    stop_process(process)
                    state["elapsed_s"] += time.perf_counter() - tick
                    atomic_json(output / "status.json", state)
            print(f"STAGE_END: {job} {state['jobs'][job]}", flush=True)
        state["status"] = (
            "completed"
            if all(j["status"] == "completed" for j in state["jobs"].values())
            else "finished_with_gaps"
        )
        atomic_json(output / "status.json", state)
        report(output)
        bundle_results(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["plan", "run", "worker", "report"])
    p.add_argument("--config", default=str(ROOT / "configs/torch_casl.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job", choices=["reference", "eager", "compile", "graph"])
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.action == "worker":
        cfg = read_json(a.output / "config.json")
        if a.job == "reference":
            from .reference import run_reference

            run_reference(cfg, a.output)
        else:
            from .worker import run_torch

            run_torch(cfg, a.output, a.job)
    elif a.action == "report":
        report(a.output)
        bundle_results(a.output)
    else:
        cfg = configuration(a.config)
        if a.action == "plan":
            print(cfg)
            print("Isolated JAX reference -> native eager -> compile -> CUDA Graph; no training.")
        else:
            run(cfg, a.output.resolve(), a.resume)


if __name__ == "__main__":
    main()
