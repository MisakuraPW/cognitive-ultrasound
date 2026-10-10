"""Chinese staged diagnostic report; no automatic baseline adoption."""

import csv
import hashlib
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..preparation.common import atomic_json, read_json
from ..provenance import sha256
from .value_protocol import bootstrap, rank_correlation


def _csv(file, rows):
    if not rows:
        return
    with file.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def schedule_statistics(records, cfg):
    refs = {(r["case"], r["seed"]): r for r in records if "baseline_M" in r["aliases"]}
    groups = defaultdict(list)
    for r in records:
        for alias in r["aliases"]:
            groups[alias].append(r)
    outputs = []
    for alias, rows in sorted(groups.items()):
        by_case = defaultdict(list)
        for r in rows:
            by_case[r["case"]].append(r)
        errors = []
        mse = []
        p90 = []
        lines = []
        seeds = []
        for name, values in by_case.items():
            e = []
            m = []
            p = []
            case_lines = []
            for r in values:
                baseline = refs[(name, r["seed"])]
                base_q = baseline["quality"]
                e.append(
                    r["ef"]["all_starts_v1"]["absolute_error"]
                    - baseline["ef"]["all_starts_v1"]["absolute_error"]
                )
                m.append(
                    (r["quality"]["mse"] - base_q["mse"]) / base_q["mse"]
                    if base_q["mse"] > 0
                    else None
                )
                p.append(
                    (r["quality"]["frame_mse_p90"] - base_q["frame_mse_p90"])
                    / base_q["frame_mse_p90"]
                    if base_q["frame_mse_p90"] > 0
                    else None
                )
                case_lines.append(r["mean_lines"])
                seeds.append(r["seed"])
            errors.append(float(np.mean(e)))
            lines.append(float(np.mean(case_lines)))
            mse.append(float(np.mean(m)) if all(x is not None for x in m) else None)
            p90.append(float(np.mean(p)) if all(x is not None for x in p) else None)
        stats = dict(
            alias=alias,
            videos=len(by_case),
            seeds=sorted(set(seeds)),
            mean_lines=float(np.mean(lines)),
            ef_delta=bootstrap(errors),
            mse_relative=bootstrap(mse) if all(x is not None for x in mse) else None,
            p90_relative=bootstrap(p90) if all(x is not None for x in p90) else None,
        )
        d = cfg["value_diagnostics"]
        enough = all(
            len(v) == 2 and {r["seed"] for r in v} == {42, 31415} for v in by_case.values()
        )
        stats["engineering_margin_screen"] = dict(
            both_seeds=enough,
            ef=stats["ef_delta"]["upper95"] <= d["ef_margin_pp"],
            mse=stats["mse_relative"]["upper95"] <= d["mse_margin"]
            if stats["mse_relative"]
            else None,
            p90=stats["p90_relative"]["upper95"] <= d["p90_margin"]
            if stats["p90_relative"]
            else None,
            scope="posthoc screening; not clinical/noninferiority proof or online generalization",
        )
        stats["screen_passed"] = bool(
            enough
            and all(stats["engineering_margin_screen"][k] is True for k in ["ef", "mse", "p90"])
        )
        outputs.append(stats)
    return outputs


def paired_schedule_statistics(records):
    aliases = defaultdict(dict)
    for r in records:
        for alias in r["aliases"]:
            aliases[alias][(r["case"], r["seed"])] = r
    result = []
    for alias, values in sorted(aliases.items()):
        if not alias.startswith(("same_", "saving_")) or alias.endswith(("periodic", "reverse")):
            continue
        arm = alias.split("_", 1)[0]
        for control in [arm + "_periodic", alias + "_reverse"]:
            if control not in aliases:
                continue
            by_case = defaultdict(list)
            for key, r in values.items():
                if key not in aliases[control]:
                    continue
                ref = aliases[control][key]
                if (
                    r["total_lines"] != ref["total_lines"]
                    or r["perception_calls"] != ref["perception_calls"]
                ):
                    raise ValueError(
                        "Paired timing controls do not have identical observation/call costs"
                    )
                by_case[key[0]].append(
                    (
                        r["ef"]["all_starts_v1"]["absolute_error"]
                        - ref["ef"]["all_starts_v1"]["absolute_error"],
                        r["quality"]["mse"] - ref["quality"]["mse"],
                    )
                )
            if by_case:
                result.append(
                    dict(
                        candidate=alias,
                        control=control,
                        videos=len(by_case),
                        ef_delta=bootstrap(
                            [np.mean([x[0] for x in rs]) for rs in by_case.values()]
                        ),
                        mse_delta=bootstrap(
                            [np.mean([x[1] for x in rs]) for rs in by_case.values()]
                        ),
                        scope="same joint counts and calls; posthoc diagnostic, not online efficacy",
                    )
                )
    return result


def report(root):
    root = Path(root)
    if not (root / "config.json").exists():
        return {}
    cfg = read_json(root / "config.json")
    manifest = read_json(root / "manifest.json")
    status = read_json(root / "status.json") if (root / "status.json").exists() else {}
    phases = {}
    for phase in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        p = root / "jobs" / phase / "result.json"
        if p.exists():
            phases[phase] = read_json(p)
    fixture = cfg["value_diagnostics"]["functional_fixture"]
    lines = [
        "# 动态预算调度空间与补采价值诊断报告",
        "",
        "**CPU合成功能测试，不能用于科学结论。**"
        if fixture
        else "冻结CASL与EF；不训练预算网络，不自动选择正式baseline。",
        "",
        f"运行状态：{status.get('status', 'not_started')}；病例数：{len(manifest['files'])}。",
        "本轮主EF读出为all_starts_v1，legacy_v4独立保留。病例是统计单位；缓存拼接与真实闭环分开。",
        "",
        "|阶段|状态|进程时间（秒）|",
        "|---|---|---:|",
    ]
    for phase in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        value = phases.get(phase, {})
        lines.append(
            f"|{phase}|{value.get('status', 'pending')}|{value.get('process_wall_s', 0):.1f}|"
        )
    fixed = phases.get("P1", {}).get("records", [])
    fixed_rows = []
    if fixed:
        lines += [
            "",
            "## 固定预算响应",
            "",
            "|档位|视频数|平均线数|EF MAE（百分点）|完整输入预测偏移|浮点MSE|PSNR（dB）|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for level in ["L", "M", "H"]:
            values = [r for r in fixed if r["level"] == level]

            def m(f):
                return float(np.mean([f(x) for x in values]))

            lines.append(
                f"|{level}|{len(values)}|{m(lambda x: x['mean_lines']):.3f}|{m(lambda x: x['ef']['all_starts_v1']['absolute_error']):.4f}|{m(lambda x: x['ef']['all_starts_v1']['preservation_error']):.4f}|{m(lambda x: x['quality']['mse']):.6f}|{m(lambda x: x['psnr']):.3f}|"
            )
            for r in values:
                fixed_rows.append(
                    dict(
                        case=r["case"],
                        level=level,
                        seed=r["seed"],
                        lines=r["mean_lines"],
                        mse=r["quality"]["mse"],
                        mse_p90=r["quality"]["frame_mse_p90"],
                        psnr=r["psnr"],
                        ssim=r["ssim"],
                        temporal_mse=r["quality"]["temporal_mse"],
                        ef_error=r["ef"]["all_starts_v1"]["absolute_error"],
                        ef_preservation=r["ef"]["all_starts_v1"]["preservation_error"],
                        legacy_ef_error=r["ef"]["legacy_v4"]["absolute_error"],
                        legacy_coverage=r["ef"]["legacy_v4"]["coverage"]["direct_fraction"],
                        all_start_coverage=r["ef"]["all_starts_v1"]["coverage"]["direct_fraction"],
                        calls=r["perception_calls"],
                    )
                )
        lines += [
            "",
            "低档7＋0有一次感知，中／高档有两次；不能把低档到中档差异全部归为观测数量。外部状态记录与读出开销另列，不声称采集省线就是GPU提速。",
        ]
    marginal = phases.get("P3", {}).get("records", [])
    cached = phases.get("P2", {}).get("records", [])
    if cached:
        lines += [
            "",
            "## 缓存时间窗口敏感性",
            "",
            "|病例|修复4窗收益范围（百分点）|破坏4窗收益范围（百分点）|",
            "|---|---|---|",
        ]
        for case in sorted(set(r["case"] for r in cached)):
            values = []
            for arm in ["repair", "damage"]:
                gains = [
                    r["ef_gain_from_background"]
                    for r in cached
                    if r["case"] == case and r["arm"] == arm
                ]
                values.append(f"[{min(gains):.4f},{max(gains):.4f}]")
            lines.append(f"|{case}|{values[0]}|{values[1]}|")
        lines += [
            "",
            "正值表示相对对应缓存背景的EF误差减少。两种背景保持共同冷帧和M尾部。缓存替换不是真实采集轨迹，不允许把该指标写成动态采集成效。",
        ]
    marginal_rows = []
    if marginal:
        lines += [
            "",
            "## 同状态真实补采",
            "",
            "每条分支新增7条真实观测，并重新运行后缀；正收益为A损失减B损失。该数值是实现后效果，不是校准的条件期望价值。",
        ]
        for stage in ["first", "second"]:
            values = [r for r in marginal if r["stage"] == stage]
            corr = rank_correlation(
                [r["features"][12] for r in values], [r["gain"]["ef_gain"] for r in values]
            )
            lines.append(
                f"- {stage}：{len(values)}分支；EF收益均值{np.mean([r['gain']['ef_gain'] for r in values]):.4f}个百分点；MSE收益均值{np.mean([r['gain']['mse_gain'] for r in values]):.6f}；评分与EF收益排序相关{corr}。锚点不是独立病例。"
            )
        for r in marginal:
            marginal_rows.append(
                dict(
                    case=r["case"],
                    stage=r["stage"],
                    frame=r["at"],
                    extra_lines=7,
                    mse_gain=r["gain"]["mse_gain"],
                    ef_gain=r["gain"]["ef_gain"],
                    preservation_gain=r["gain"]["preservation_gain"],
                    candidate_score=r["features"][12],
                    variance=r["features"][13],
                    sensitivity=r["features"][14],
                    cancellation_ratio=r["features"][15],
                    uncertainty_change=r["uncertainty_change"],
                )
            )
    p4 = phases.get("P4", {})
    statistics = schedule_statistics(p4.get("records", []), cfg) if p4.get("records") else []
    paired = paired_schedule_statistics(p4.get("records", []))
    if statistics:
        lines += [
            "",
            "## 锁定的真实预算安排",
            "",
            "|安排|病例|种子|平均线数|相对M的EF差|EF差95%区间|相对MSE变化|工程筛查|",
            "|---|---:|---|---:|---:|---|---:|---|",
        ]
        for s in statistics:
            q = s["mse_relative"]
            relative = f"{100 * q['mean']:.2f}%" if q else "未评估"
            lines.append(
                f"|{s['alias']}|{s['videos']}|{s['seeds']}|{s['mean_lines']:.3f}|{s['ef_delta']['mean']:.4f}|{s['ef_delta']['ci95']}|{relative}|{s['screen_passed']}|"
            )
        lines += [
            "",
            "### 相同观测及调用成本的时间安排比较",
            "",
            "|候选|对照|视频|EF差，候选减对照|95%病例区间|MSE差|",
            "|---|---|---:|---:|---|---:|",
        ]
        for s in paired:
            lines.append(
                f"|{s['candidate']}|{s['control']}|{s['videos']}|{s['ef_delta']['mean']:.4f}|{s['ef_delta']['ci95']}|{s['mse_delta']['mean']:.6f}|"
            )
        lines += [
            "",
            "候选来自种子42的事后缓存选择，31415不重选。筛查区间没有校正选择效应；通过不等于临床非劣效或在线策略有效。相同H/L直方图对照控制线数与调用数；全M同总线数但调用次数不同，另行解释。",
        ]
    elif p4.get("status") == "skipped":
        lines += [
            "",
            "## 真实分配阶段跳过",
            "",
            p4["reason"],
            "门槛只决定是否进入有限验证，不表示动态预算不可能有效。",
        ]
    if "P5" in phases:
        lines += [
            "",
            "## 可观测状态预测筛查",
            "",
            "|阶段|标签|视频|留一视频RMSE|训练折常数RMSE|排序相关|",
            "|---|---|---:|---:|---:|---|",
        ]
        for x in phases["P5"]["predictability"]:
            if "rmse" in x:
                lines.append(
                    f"|{x['stage']}|{x['label']}|{x['videos']}|{x['rmse']:.5f}|{x['constant_rmse']:.5f}|{x['rank_correlation']}|"
                )
        lines += [
            "",
            "只拟合固定alpha的CPU线性筛查器，按视频隔离训练／验证。没有训练预算策略，结果不证明部署泛化。",
        ]
    lines += [
        "",
        "## 判读与证据边界",
        "",
        "先检查真实预算响应，再检查时间差异与真实闭环收益，最后区分可达收益和在线可预测收益。重建有空间而EF较平时保留两者事实，不强迫EF敏感。",
        "旧16确认病例不用于这轮选择；扩展16诊断病例须显式开启独立输出。没有新增预算档位搜索或自动扩展。",
        "NaN、缓存／状态不一致、读出覆盖失败会停止；运行失败与科学结果不理想分开记录。",
        "原始完整缓存和恢复状态留在服务器；分析包不含模型权重或帧数组。",
    ]
    if status.get("error"):
        lines += ["", f"运行异常：{status['error']}"]
    summary = dict(
        status=status.get("status"),
        functional_fixture=fixture,
        phases={k: v["status"] for k, v in phases.items()},
        schedules=statistics,
        paired_schedules=paired,
        observations=dict(fixed_videos=len(fixed), marginal_branches=len(marginal)),
        no_automatic_adoption=True,
        unrun_phases=[p for p in ["P0", "P1", "P2", "P3", "P4", "P5"] if p not in phases],
        scope="descriptive, posthoc diagnostics; not clinical equivalence",
    )
    atomic_json(root / "summary.json", summary)
    (root / "REPORT_中文.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _csv(root / "fixed_cases.csv", fixed_rows)
    _csv(root / "marginal_cases.csv", marginal_rows)
    if fixed and not cfg["value_diagnostics"].get("skip_fixture_plots", False):
        figures(root, fixed, marginal, fixture, cached)
    return summary


def figures(root, fixed, marginal, fixture, cached=()):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        for case in sorted(set(r["case"] for r in fixed)):
            rows = sorted([r for r in fixed if r["case"] == case], key=lambda r: r["mean_lines"])
            axes[0].plot(
                [r["mean_lines"] for r in rows],
                [r["ef"]["all_starts_v1"]["absolute_error"] for r in rows],
                "-o",
                alpha=0.65,
            )
            axes[1].plot(
                [r["mean_lines"] for r in rows],
                [r["quality"]["mse"] for r in rows],
                "-o",
                alpha=0.65,
            )
        axes[0].set(xlabel="Scan lines / frame", ylabel="EF absolute error (pp)")
        axes[1].set(xlabel="Scan lines / frame", ylabel="Floating image MSE")
        fig.suptitle(
            ("SYNTHETIC FUNCTIONAL / " if fixture else "")
            + "Observed paired budget responses; one line per video"
        )
        fig.savefig(root / "fixed_response.png", dpi=180, facecolor="white")
        plt.close(fig)
        if marginal:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
            for stage, marker in [("first", "o"), ("second", "^")]:
                rows = [x for x in marginal if x["stage"] == stage]
                for ax, label in zip(axes, ["ef_gain", "mse_gain"]):
                    ax.scatter(
                        [x["features"][12] for x in rows],
                        [x["gain"][label] for x in rows],
                        marker=marker,
                        label=stage,
                        alpha=0.65,
                    )
                    ax.axhline(0, color="gray", linestyle="--")
                    ax.set(xlabel="Estimated combined score", ylabel=label)
                    ax.legend()
            fig.suptitle(
                ("SYNTHETIC FUNCTIONAL / " if fixture else "")
                + "Realized +7-line gains; anchors are not independent videos"
            )
            fig.savefig(root / "marginal_response.png", dpi=180, facecolor="white")
            plt.close(fig)
        if cached:
            cases = sorted(set(x["case"] for x in cached))
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
            matrixes = []
            for arm in ["repair", "damage"]:
                matrixes.append(
                    np.array(
                        [
                            [
                                next(
                                    x["ef_gain_from_background"]
                                    for x in cached
                                    if x["case"] == case and x["arm"] == arm and x["high"] == [i]
                                )
                                for i in range(4)
                            ]
                            for case in cases
                        ]
                    )
                )
            limit = max(float(np.abs(m).max()) for m in matrixes) or 1.0
            for ax, arm, m in zip(axes, ["repair", "damage"], matrixes):
                im = ax.imshow(m, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
                ax.set(
                    title=arm,
                    xlabel="Warm time block",
                    xticks=range(4),
                    xticklabels=[1, 2, 3, 4],
                    yticks=range(len(cases)),
                    yticklabels=[f"Video{i + 1}" for i in range(len(cases))],
                )
                fig.colorbar(im, ax=ax, label="EF error reduction (pp)")
            fig.suptitle(
                ("SYNTHETIC FUNCTIONAL / " if fixture else "")
                + "Cached image sensitivity only; not physical acquisition"
            )
            fig.savefig(root / "cached_sensitivity.png", dpi=180, facecolor="white")
            plt.close(fig)


def bundle(root):
    root = Path(root)
    report(root)
    names = [
        "REPORT_中文.md",
        "summary.json",
        "config.json",
        "manifest.json",
        "identity.json",
        "plan.json",
        "status.json",
        "p4_gate.json",
        "fixed_cases.csv",
        "marginal_cases.csv",
        "fixed_response.png",
        "marginal_response.png",
        "cached_sensitivity.png",
    ]
    files = [root / n for n in names if (root / n).exists()]
    files.extend(sorted((root / "jobs").glob("*/result.json")))
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    previous = root / "bundle_receipt.json"
    if previous.exists():
        old = read_json(previous)
        if (
            old.get("source_hashes") == hashes
            and Path(old["archive"]).exists()
            and sha256(old["archive"]) == old["sha256"]
        ):
            return old
    target = root.parent / f"{root.name}.analysis_{time.time_ns()}.tar.gz"
    partial = target.with_suffix(".partial")
    with tarfile.open(partial, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(Path(root.name) / p.relative_to(root)), recursive=False)
    with tarfile.open(partial, "r:gz") as tar:
        for member in tar:
            relative = str(Path(member.name).relative_to(root.name))
            h = hashlib.sha256()
            with tar.extractfile(member) as f:
                for b in iter(lambda: f.read(2**20), b""):
                    h.update(b)
            if h.hexdigest() != hashes[relative]:
                raise AssertionError("Analysis archive verification failed")
    partial.replace(target)
    receipt = dict(
        archive=str(target),
        bytes=target.stat().st_size,
        sha256=sha256(target),
        verified=True,
        source_hashes=hashes,
        scope="small reports/protocol; no frame arrays/model weights",
    )
    atomic_json(previous, receipt)
    return receipt
