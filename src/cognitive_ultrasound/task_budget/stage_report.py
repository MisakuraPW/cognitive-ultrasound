"""Separate exploration signals, quality costs and runtime failures in Chinese."""

import csv
import hashlib
import tarfile
import time
from pathlib import Path

import numpy as np

from ..preparation.common import atomic_json, read_json
from ..provenance import sha256
from .stage_protocol import ARMS
from .value_protocol import bootstrap


def pilot_gate(root):
    root = Path(root)
    cfg = read_json(root / "config.json")
    trained = read_json(root / "jobs/pilot_train/result.json")["models"]
    evaluated = read_json(root / "jobs/pilot_evaluate/result.json")["cases"]
    reasons = []
    for arm in ARMS:
        x = next(r for r in trained if r["arm"] == arm)
        records = [
            read_json(root / "training" / arm / "updates" / f"{i:05d}" / "record.json")
            for i in range(1, x["update"] + 1)
        ]
        if not x["parameters_changed"] or not any(r["gradient_norm"] > 1e-12 for r in records):
            reasons.append(f"{arm}: no parameter/task learning signal")
        if not any(r["task_gradient_norm"] > 1e-12 for r in records):
            reasons.append(f"{arm}: task component absent; reject cost-only learning")
        for r in records:
            if not all(
                np.isfinite(r[k]) for k in ["gradient_norm", "objective", "ef_error", "mse"]
            ):
                reasons.append(f"{arm}: nonfinite training state")
        for r in [e for e in evaluated if e["arm"] == arm]:
            if arm != "Q2" and r["total_lines"] != r["reference"]["total_lines"]:
                reasons.append(f"{arm}: quota mismatch")
            if (
                r["perception_calls"] != r["periodic"]["perception_calls"]
                or r["pairs"] != r["periodic"]["pairs"]
            ):
                reasons.append(f"{arm}: periodic cost mismatch")
    value = dict(
        status="passed" if not reasons else "failed",
        reasons=reasons,
        gate="execution/finite training/physical quotas only; no positive-effect or CI requirement",
        automatic_full_continuation=not reasons,
        functional_fixture=cfg["stage_budget"]["functional_fixture"],
    )
    atomic_json(root / "pilot_gate.json", value)
    report(root)
    return value


def aggregate(rows, key):
    # Aggregate seeds inside each case first. Frames/seeds are not independent patients.
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r[key])
    return bootstrap([np.mean(v) for v in by.values()]) if by else None


def report(root):
    root = Path(root)
    cfg = read_json(root / "config.json")
    m = read_json(root / "manifest.json")
    status = read_json(root / "status.json") if (root / "status.json").exists() else {}
    jobs = {p.parent.name: read_json(p) for p in sorted((root / "jobs").glob("*/result.json"))}
    rows = [
        r
        for name, v in jobs.items()
        if name.endswith("evaluate") or name == "confirmation_policies"
        for r in v.get("cases", [])
    ]
    summary = dict(
        status=status.get("status", "prepared"),
        functional_fixture=cfg["stage_budget"]["functional_fixture"],
        cohorts={k: len(v) for k, v in m["cohorts"].items()},
        completed_phases=list(jobs),
        arms={},
        no_full_casl_training=True,
        segmentation_evaluated=False,
        statistics="case-level seed aggregation; descriptive CI does not gate exploration",
    )
    lines = [
        "# 两阶段动态预算：单帧省预算与EF时间分配",
        "",
        f"状态：{summary['status']}。TRAIN／开发／确认：{summary['cohorts']}。",
        "**SYNTHETIC FUNCTIONAL ONLY，不能提供医学效果证据。**"
        if summary["functional_fixture"]
        else "真实权重闭环；CASL、EF与选线规则冻结。",
        "",
        "T1只学第一阶段EF分配；Q2只学第二阶段重建省预算；T2只学第二阶段EF分配。",
        "T1/T2按视频精确匹配14条平均预算；Q2重建约束只按点估计筛选。EF实验的MSE变化是代价记录，不是失败门槛。",
        "",
        "|阶段|状态|秒|",
        "|---|---|---:|",
    ]
    for name, v in jobs.items():
        lines.append(f"|{name}|{v.get('status')}|{v.get('seconds', 0):.1f}|")
    lines.extend(
        [
            "",
            "|阶段／实验|视频|种子|EF MAE|平均线数|相对M重建MSE|EF差：策略减周期|95%病例区间|",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for phase in [
        "pilot_evaluate",
        "full_pass_1_evaluate",
        "full_pass_2_evaluate",
        "full_pass_4_evaluate",
        "confirmation_policies",
    ]:
        values = jobs.get(phase, {}).get("cases", [])
        for arm in ARMS:
            a = [v for v in values if v["arm"] == arm]
            if not a:
                continue
            outcome = {
                k: aggregate(a, k)
                for k in ["ef_error", "mean_lines", "mse_ratio", "p90_ratio", "ef_vs_fixed"]
            }
            matched = [x for x in a if "ef_vs_periodic" in x]
            outcome["ef_vs_periodic"] = aggregate(matched, "ef_vs_periodic")
            outcome["mse_vs_periodic"] = aggregate(matched, "mse_vs_periodic")
            summary["arms"][phase + "/" + arm] = outcome
            diff = outcome["ef_vs_periodic"]
            lines.append(
                f"|{phase}/{arm}|{len(set(x['case'] for x in a))}|{sorted(set(x['seed'] for x in a))}|"
                f"{outcome['ef_error']['mean']:.4f}|{outcome['mean_lines']['mean']:.3f}|"
                f"{outcome['mse_ratio']['mean']:.2%}|{diff['mean'] if diff else '未运行对照'}|{diff['ci95'] if diff else '—'}|"
            )
    selection = jobs.get("select", {}).get("selection")
    if selection:
        summary["selection"] = selection
        lines.extend(
            [
                "",
                "开发集锁定检查点："
                + str({a: x["chosen"]["update"] for a, x in selection.items()}),
                "确认集不参与选择；Q2没有合格点时选择最小约束违例者并明确登记，不能称为质量保持成功。",
            ]
        )
    lines.extend(
        [
            "",
            "## 训练与证据边界",
            "",
            "小规模8视频×2遍是共同课程的前16次更新；正式第1遍只补剩余56视频，后续继续完整64视频。264次更新含试跑，不重新初始化模型、Adam或奖励EMA。",
            "训练和主评价均采样策略；没有把argmax结果混入主表。时序GRU只接收可观测历史；当前真值、未来像素、EF标签与完整输入参考只用于离线奖励或评价。",
            "训练种子为42，两个评价种子不等于两个独立训练种子。63帧TRAIN视频沿用已登记输入，EF直接覆盖不足在coverage中保留，不声称所有帧都直接被任务窗口读取。",
            "周期对照是事后联合预算组成匹配，用于隔离时间安排，不是在线可部署策略。零补采会少一次DPS，调用数另记，不把省线直接称为GPU加速。",
            "置信区间只描述不确定性；没有报错不证明收敛。分割、联合多任务、WM与第一阶段单帧省预算未在本批实现或计算。",
        ]
    )
    atomic_json(root / "summary.json", summary)
    text = "\n".join(lines) + "\n"
    (root / "REPORT_中文.md").write_text(text, encoding="utf-8")
    if "pilot_evaluate" in jobs:
        pilot_text = (
            "\n".join(lines[:5])
            + "\n\n## 小规模健康门槛\n"
            + str(
                read_json(root / "pilot_gate.json")
                if (root / "pilot_gate.json").exists()
                else "待核对"
            )
            + "\n\n"
        )
        for arm in ARMS:
            outcome = summary["arms"].get("pilot_evaluate/" + arm)
            if outcome:
                pilot_text += f"### {arm}\n{outcome}\n\n"
        pilot_text += "小规模只检查执行与学习信号；效果不好、区间跨零不阻止已授权的正式实验。\n"
        (root / "PILOT_REPORT.md").write_text(pilot_text, encoding="utf-8")
    keys = [
        "arm",
        "update",
        "cohort",
        "case",
        "seed",
        "ef_error",
        "mean_lines",
        "total_lines",
        "perception_calls",
        "mse",
        "p90",
        "mse_ratio",
        "p90_ratio",
        "ef_vs_fixed",
        "ef_vs_periodic",
        "mse_vs_periodic",
        "quota_forced_fraction",
        "zero_second_fraction",
    ]
    with (root / "evaluation_cases.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def bundle(root):
    root = Path(root)
    report(root)
    files = [
        root / n
        for n in [
            "REPORT_中文.md",
            "PILOT_REPORT.md",
            "summary.json",
            "evaluation_cases.csv",
            "config.json",
            "requested_config.json",
            "manifest.json",
            "identity.json",
            "plan.json",
            "status.json",
            "pilot_gate.json",
        ]
        if (root / n).exists()
    ]
    files.extend(sorted((root / "jobs").glob("*/result.json")))
    files.extend(sorted((root / "training").glob("*/updates/*/record.json")))
    files.extend(sorted((root / "evaluations").glob("*/budget_trace.csv")))
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    receipt = root / "bundle_receipt.json"
    if receipt.exists():
        old = read_json(receipt)
        if (
            old["source_hashes"] == hashes
            and Path(old["archive"]).exists()
            and sha256(old["archive"]) == old["sha256"]
        ):
            return old
    target = root.parent / f"{root.name}.analysis_{time.time_ns()}.tar.gz"
    partial = target.with_suffix(".partial")
    with tarfile.open(partial, "w:gz") as t:
        for p in files:
            t.add(p, arcname=str(Path(root.name) / p.relative_to(root)), recursive=False)
    with tarfile.open(partial, "r:gz") as t:
        for member in t:
            relative = str(Path(member.name).relative_to(root.name))
            h = hashlib.sha256(t.extractfile(member).read()).hexdigest()
            if h != hashes[relative]:
                raise AssertionError("Analysis bundle content verification failed")
    partial.replace(target)
    value = dict(
        archive=str(target),
        bytes=target.stat().st_size,
        sha256=sha256(target),
        verified=True,
        source_hashes=hashes,
        scope="analysis and training records; excludes frames, controller/Adam states and frozen weights",
    )
    atomic_json(receipt, value)
    return value
