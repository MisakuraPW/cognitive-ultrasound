"""Final fixed preparation batch. Completion closes preparation, never starts a search."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..config import load, path
from ..provenance import sha256
from .common import atomic_json, emit, read_json
from .suite import Coordinator, bundle_results, run, run_lock


def source_receipt(cfg):
    excluded = set()
    files = {}
    ancestors = []
    for folder in cfg["closure"]["previous_runs"]:
        p = Path(folder).resolve()
        for name in ("manifest.json", "identity.json", "selection.json"):
            file = p / name
            if not file.is_file():
                raise FileNotFoundError(f"Required previous-run receipt: {file}")
            files[str(file)] = sha256(file)
        m = read_json(p / "manifest.json")
        excluded.update(v for k, names in m["cohorts"].items() if k != "train" for v in names)
        ancestors.append(read_json(p / "identity.json"))
    bf = Path(cfg["closure"]["bf_source"]) / "jobs/bf_filter/training/checkpoint.npz"
    if bf.is_file():
        files[str(bf)] = sha256(bf)
    return dict(
        files=files, excluded_validation_cases=sorted(excluded), bf_available=bf.is_file()
    ), ancestors


class Closure(Coordinator):
    def initialize(self):
        validate_closure(self.cfg)
        receipt, ancestors = source_receipt(self.cfg)
        for folder in self.cfg["closure"]["previous_runs"]:
            p = Path(folder).resolve()
            if self.root == p or self.root in p.parents or p in self.root.parents:
                raise ValueError("Closure output must be separate from all ancestors")
        old = self.root / "closure_sources.json"
        if old.exists() and read_json(old) != receipt:
            raise ValueError("Inherited source changed")
        self.cfg["excluded_validation_cases"] = receipt["excluded_validation_cases"]
        super().initialize()
        current = read_json(self.root / "identity.json")
        for ancestor in ancestors:
            for key in ("weights", "casl_commit", "zea_commit"):
                if current[key] != ancestor[key]:
                    raise ValueError("Inherited official asset mismatch: " + key)
        atomic_json(old, receipt)
        atomic_json(
            self.root / "closure_protocol.json",
            dict(
                scope="Final finite preparation; no automatic follow-up or parameter search",
                controls=[
                    "fixed",
                    "random",
                    "current_uncertainty",
                    "simple_forecast",
                    "future_forecast",
                ],
                fixed_total="14 * clip_frames for every policy/case/seed",
                bf=f"{self.cfg['closure']['bf_steps']} updates x 12 frames in each arm; identical parent/frozen components/sample stream; reset interval only",
                holds="All prior validation-use patients excluded before reading outcomes; no test data",
                stopping="All fixed jobs attempted or explicitly blocked/capped; negative evidence closes a question too",
            ),
        )

    def execute(self):
        from .closure_budget import calibrate_selection

        for k in self.cfg["closure"]["budgets"]:
            for variant in ("reference", "official25_fp16"):
                self.job(
                    f"anchor_{k}_{variant}",
                    "closure_anchor",
                    "A",
                    budget=k,
                    variant=variant,
                    cohort="debug",
                    method="uniform",
                )
        chosen = calibrate_selection(self.cfg, self.root)
        reason = (
            None
            if chosen["ready"]
            else "Reference calibration incomplete; scientific GPU collection blocked"
        )
        for cohort in ("train", "development", "confirmation"):
            self.job(
                "budget_data_" + cohort, "closure_collect", "C", cohort=cohort, condition=reason
            )
        self.job(
            "budget_fit",
            "closure_fit",
            "C",
            dependencies=("budget_data_train", "budget_data_development"),
        )
        self.job(
            "budget_probe_confirm",
            "closure_probe",
            "C",
            dependencies=("budget_fit", "budget_data_confirmation"),
        )
        for cohort in ("development", "confirmation"):
            self.job(
                "funded_loop_" + cohort,
                "closure_loop",
                "C",
                cohort=cohort,
                dependencies=("budget_fit",),
                condition=reason,
            )
        receipt = read_json(self.root / "closure_sources.json")
        bf_reason = (
            None
            if receipt["bf_available"]
            else "Inherited completed BF filter checkpoint absent; no scratch training requested"
        )
        for name, interval in (("reset3", 3), ("continuous12", 12)):
            self.job(
                "bf_history_" + name,
                "closure_bf_train",
                "B",
                reset_interval=interval,
                condition=bf_reason,
            )
        for cohort in ("development", "confirmation"):
            self.job(
                "bf_history_eval_" + cohort,
                "closure_bf_eval",
                "B",
                cohort=cohort,
                dependencies=("bf_history_reset3", "bf_history_continuous12"),
            )
        spec = self.cfg["tbig"]
        ready = all(
            spec.get(k)
            for k in ("python", "repo", "config", "expected_commit", "sequences", "asset_files")
        )
        self.job(
            "tbig",
            "tbig",
            "A",
            max_seconds=min(600, spec["max_seconds"]),
            condition=None
            if ready
            else "External TBIG environment/weights/compatible sequences absent; closed as unavailable, not a mainline prerequisite",
        )
        self.state.update(
            status="completed"
            if all(j["status"] == "completed" for j in self.state["jobs"].values())
            else "finished_with_gaps",
            preparation_closed=True,
            automatic_followup=False,
        )
        self.save()
        self.render_report()

    def render_report(self):
        render_report(self.root)


def validate_closure(cfg):
    spec = cfg["closure"]
    if spec["budgets"] != [7, 14, 28] or cfg["budget"] != 14:
        raise ValueError("Closure protocol fixes budgets 7/14/28 and total 14*frames")
    if not 1 <= spec["bf_steps"] <= 500 or spec["bf_clip_frames"] != 12:
        raise ValueError("Closure BF is capped at 500 updates of 12 frames per arm")
    if not 0 < spec["quality_mae"] <= 2:
        raise ValueError("Quality threshold uses normalized [-1,1] MAE")
    if cfg["max_hours"] > 3 or not spec["previous_runs"]:
        raise ValueError("Closure requires prior receipts and at most three task-hours")
    if spec.get("cpu_only", False):
        raise ValueError("Full closure is a GPU run; CPU is only for isolated functional tests")


def dispatch(task, cfg, manifest, output, root):
    kind = task["kind"]
    if kind == "closure_anchor":
        from .casl import run_trajectory

        run_trajectory(task, dict(cfg, budget=task["budget"]), manifest, output)
    elif kind.startswith("closure_bf_"):
        from .closure_belief import qualify, train

        (train if kind.endswith("train") else qualify)(task, cfg, manifest, output, root)
    else:
        from . import closure_budget as b

        {
            "closure_collect": b.collect,
            "closure_fit": b.fit,
            "closure_probe": b.evaluate_probe,
            "closure_loop": b.closed_loop,
        }[kind](task, cfg, manifest, output, root)


def render_report(root):
    """Robust even with empty, failed or capped tasks; never asserts a positive finding."""
    root = Path(root)
    status = read_json(root / "status.json")
    lines = [
        "# 最后一批准备实验：交接报告",
        "",
        "本批按固定任务清单结束，不自动加实验。完成运行与方法有效是两件事；缺条件、超限与负结果均保留。",
        "",
        "|任务|状态|累计秒数|说明|",
        "|---|---|---:|---|",
    ]
    export = []
    raw = {}
    for name, job in status.get("jobs", {}).items():
        note = job.get("reason", "")
        lines.append(f"|{name}|{job['status']}|{job.get('elapsed_s', 0):.1f}|{note}|")
        export.append(
            dict(
                experiment_id=name,
                status=job["status"],
                seconds=round(job.get("elapsed_s", 0), 2),
                fact=note,
                result_path=f"jobs/{name}/result.json",
                prediction="",
                prediction_lock="",
                surprise="",
                belief_update="",
                next_decision="",
            )
        )
        p = root / "jobs" / name / "result.json"
        if p.exists():
            raw[name] = read_json(p)
            if job["status"] == "completed":
                export[-1]["fact"] = "已完成；指标与设置见 result.json，不代表方法有效"
    selection = root / "budget_selection.json"
    if selection.exists():
        s = read_json(selection)
        lines += [
            "",
            "## 运行底座",
            f"本机选择：`{s['variant']}`；原版检查完整：{s['ready']}。三个预算的门槛详情见 budget_selection.json。",
        ]
    closure_summary = {}
    for cohort in ("development", "confirmation"):
        name = "funded_loop_" + cohort
        value = raw.get(name, {})
        if value.get("status") == "completed":
            grouped = {}
            for r in value["records"]:
                for policy, scores in r["scores"].items():
                    if scores["total_lines"] != value["total_budget_per_clip"]:
                        raise ValueError("Unmatched total in report")
                    grouped.setdefault(policy, {}).setdefault(r["case"], []).append(scores)
            means = {
                p: {
                    k: float(
                        np.mean([np.mean([r[k] for r in records]) for records in patients.values()])
                    )
                    for k in next(iter(next(iter(patients.values())))).keys()
                }
                for p, patients in grouped.items()
            }
            closure_summary[name] = means
            for row in export:
                if row["experiment_id"] == name:
                    row["fact"] = "同总预算全图MAE（病例平均）：" + json.dumps(
                        {p: round(v["all_pixel_mae"], 6) for p, v in means.items()}
                    )
            lines += [
                "",
                f"## 同总预算闭环：{cohort}",
                "所有策略的总线数在运行前规定，逐病例核对一致。以下先在病例内合并种子，再平均病例。",
                "",
                "|策略|全图 MAE|共同未测 MAE|低质量帧比例|平均回退帧数|总线数/片段|",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for p, v in means.items():
                lines.append(
                    f"|{p}|{v['all_pixel_mae']:.6f}|{v['common_unobserved_mae']:.6f}|{v['low_quality_fraction']:.3f}|{v['fallback_frames']:.2f}|{v['total_lines']:.0f}|"
                )
            future = means["future_forecast"]["all_pixel_mae"]
            baseline = min(v["all_pixel_mae"] for p, v in means.items() if p != "future_forecast")
            lines.append(
                f"\n未来预测策略相对最好的已列参照：{'此子集较好' if future < baseline else '此子集未胜出'}。不进行额外调参；两者平均误差差值={future - baseline:+.6f}。"
            )
        bf = raw.get("bf_history_eval_" + cohort, {})
        if bf.get("status") == "completed":
            lines += [
                "",
                f"## BF 连续历史对照：{cohort}",
                str(bf["means"]),
                "两训练臂冻结同一 codec/prior，从同一 filter 开始，训练更新数、帧数、采样顺序相同。只改变训练中状态重置间隔；评估均连续运行。完整曲线和插值参照见任务 result.json。",
            ]
    probe = raw.get("budget_probe_confirm", {})
    if probe.get("status") == "completed":
        lines += [
            "",
            "## 预算条件预测",
            str(probe["mean_mae"]),
            "目标是候选预算下下一帧全图误差。预测分数和策略效能分别报告；不得互相替代。",
        ]
    lines += [
        "",
        "## 准备阶段结束后的使用方式",
        "- 本报告不启动新一轮准备实验。负结果用于缩小正式研究范围，缺 TBIG 资产不阻塞 EchoNet 主线。",
        "- 先读确认病例的预算对照及 BF 对照，再由你在科研 Excel 逐项填写研究问题、Prediction 和 Prediction Lock。",
        "- `excel_facts.csv` 只导出已发生的状态与事实，Prediction / Surprise / Belief Update 留空，不代替你的判断。",
        "- 正式创新实验从独立输出目录开始。更换硬件重新预检，不混写历史结果。",
        "- 本批不证明新颖性、临床安全、完整 CASL/TBIG/学长论文复现，也不自动解锁完整训练。",
    ]
    for name in render_figures(root, raw, closure_summary):
        lines += ["", f"![准备实验诊断图]({name})"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(
        root / "closure_summary.json",
        dict(
            status=status["status"],
            preparation_closed=status.get("preparation_closed", False),
            automatic_followup=False,
            summaries=closure_summary,
        ),
    )
    with (root / "excel_facts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(export[0]) if export else ["experiment_id", "status"]
        )
        writer.writeheader()
        writer.writerows(export)


def render_figures(root, raw, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    for job, values in summary.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = list(values)
        ax.bar(labels, [values[p]["all_pixel_mae"] for p in labels])
        ax.set(ylabel="All-pixel MAE (case mean)", title=job + ": identical total lines")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        name = job + ".png"
        fig.savefig(root / name, dpi=140)
        plt.close(fig)
        names.append(name)
    for cohort in ("development", "confirmation"):
        bf = raw.get("bf_history_eval_" + cohort, {})
        if bf.get("status") != "completed":
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for mode in ("frozen", "reset3", "continuous12"):
            curves = [
                [r["unobserved_mae"] for r in c["rows"]] for c in bf["records"] if c["mode"] == mode
            ]
            ax.plot(np.mean(curves, axis=0), label=mode)
        baseline = [
            [r["interpolation_mae"] for r in c["rows"]]
            for c in bf["records"]
            if c["mode"] == "frozen"
        ]
        ax.plot(np.mean(baseline, axis=0), label="interpolation", linestyle="--")
        ax.set(
            xlabel="Frame", ylabel="Unobserved MAE", title="Matched training exposure: " + cohort
        )
        ax.legend()
        fig.tight_layout()
        name = "bf_history_" + cohort + ".png"
        fig.savefig(root / name, dpi=140)
        plt.close(fig)
        names.append(name)
    # Deterministic representative: first sorted case/seed and final stored frame.
    loop = root / "jobs/funded_loop_confirmation"
    cases = sorted(p for p in loop.glob("*/*") if (p / "complete.json").is_file())
    if cases:
        case = cases[0]
        policies = ["fixed", "random", "current_uncertainty", "simple_forecast", "future_forecast"]
        sample = sorted((case / "fixed").glob("frame_*.npz"))[-1].name
        fig, axes = plt.subplots(1, 6, figsize=(15, 3))
        for index, policy in enumerate(policies, 1):
            with np.load(case / policy / sample, allow_pickle=False) as data:
                axes[index].imshow(data["prediction"].squeeze(), cmap="gray", vmin=-1, vmax=1)
                axes[index].set_title(policy, fontsize=9)
                if index == 1:
                    axes[0].imshow(data["target"].squeeze(), cmap="gray", vmin=-1, vmax=1)
                    axes[0].set_title("target")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        name = "funded_loop_example.png"
        fig.savefig(root / name, dpi=140)
        plt.close(fig)
        names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "plan", "report"], nargs="?", default="run")
    parser.add_argument("--config", default="configs/preparation_closure.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        cfg = load(path(args.config))
        validate_closure(cfg)
        emit(
            "CLOSURE_PLAN",
            config=cfg,
            gpu_execution=False,
            automatic_followup=False,
        )
    elif args.command == "report":
        with run_lock(Path(args.output)):
            render_report(args.output)
            emit("RESULTS", archive=bundle_results(Path(args.output)))
    else:
        run(
            load(path(args.config)),
            args.output,
            args.resume,
            args.retry_failed,
            coordinator_type=Closure,
        )


if __name__ == "__main__":
    main()
