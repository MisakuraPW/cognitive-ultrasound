"""Case-equal reports and separately labelled analysis/model archives."""

import csv
import hashlib
import html
import json
import tarfile
from pathlib import Path

import numpy as np

from .data import atomic, read, sha


def archive(root, names, target):
    expected = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in names}
    tmp = target.with_suffix(".partial")
    with tarfile.open(tmp, "w:gz") as t:
        for p in names:
            t.add(p, arcname=str(Path(root.name) / p.relative_to(root)), recursive=False)
    with tarfile.open(tmp, "r:gz") as t:
        for member in t:
            key = "/".join(Path(member.name).parts[1:])
            if hashlib.sha256(t.extractfile(member).read()).hexdigest() != expected[key]:
                raise ValueError("Archive member checksum mismatch")
    tmp.replace(target)
    return dict(
        archive=str(target),
        bytes=target.stat().st_size,
        sha256=sha(target),
        verified=True,
        files=expected,
    )


def report(root):
    root = Path(root)
    old = root / "export.json"
    if old.exists():
        receipt = read(old)
        if all(
            Path(r["archive"]).exists() and sha(r["archive"]) == r["sha256"]
            for r in receipt.values()
        ):
            return receipt
        raise ValueError("Existing result archive corrupted; preserve it for diagnosis")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = read(root / "config.json")
    functional = read(root / "identity.json")["cpu_functional"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), layout="constrained")
    for i, stage in enumerate(["codec", "prior", "filter"]):
        file = root / stage / "training.jsonl"
        records = (
            [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
            if file.exists()
            else []
        )
        val = (
            read(root / stage / "validation.json")
            if (root / stage / "validation.json").exists()
            else []
        )
        axes[i, 0].plot(
            [r["update"] for r in records], [r["total"] for r in records], color="#0072B2", lw=1
        )
        axes[i, 0].set(
            title=f"{stage}: actual batch objective",
            xlabel="Optimizer update",
            ylabel="Objective (stage-specific)",
        )
        axes[i, 1].plot(
            [r["epoch"] for r in val], [r["mse"] for r in val], color="#D55E00", marker="o"
        )
        axes[i, 1].set(
            title=f"{stage}: case-equal validation", xlabel="Epoch", ylabel="Float MSE, data range2"
        )
    if functional:
        fig.suptitle("CPU FUNCTIONAL TEST ONLY - NOT SCIENTIFIC RESULTS", color="#A00000")
    for ext in ["png", "pdf"]:
        fig.savefig(root / f"training_curves.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    galleries = []
    for cohort in ["development", "test"]:
        paths = sorted((root / "evaluation" / cohort / "greedy").glob("*/samples/frame_*.npz"))
        if not paths:
            continue
        first = paths[0].parent.parent.name
        paths = [p for p in paths if p.parent.parent.name == first]
        fig, axes = plt.subplots(
            len(paths), 6, figsize=(13, 2.2 * len(paths)), squeeze=False, layout="constrained"
        )
        for i, p in enumerate(paths):
            with np.load(p, allow_pickle=False) as z:
                original = {k: z[k] for k in z.files}
            other = root / "evaluation" / cohort / "uniform" / first / "samples" / p.name
            with np.load(other, allow_pickle=False) as z:
                uniform = z["reconstruction"]
            pictures = [
                original["target"],
                original["reconstruction"],
                uniform,
                original["mask"],
                np.abs(original["target"] - original["reconstruction"]),
                original["uncertainty"],
            ]
            limits = [(-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 2), (0, 4.1)]
            titles = [
                "Ground truth",
                "Greedy10+4",
                "Uniform10+4",
                "Greedy observed mask",
                "Absolute error",
                "Uncertainty proxy",
            ]
            for j, (picture, (lo, hi), title) in enumerate(zip(pictures, limits, titles)):
                axes[i, j].imshow(picture, cmap="gray" if j < 4 else "viridis", vmin=lo, vmax=hi)
                axes[i, j].set_title(f"{title}\n[{lo}, {hi}]", fontsize=9)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            axes[i, 0].set_ylabel(p.stem.replace("frame_", "Frame "))
        banner = "CPU FUNCTIONAL ONLY | " if functional else ""
        fig.suptitle(f"{banner}{cohort} | {first} | Fixed display ranges")
        image_name = f"samples_{cohort}.png"
        fig.savefig(root / image_name, dpi=180, facecolor="white")
        plt.close(fig)
        galleries.append(image_name)
    rows = read(root / "results.json") if (root / "results.json").exists() else []
    summaries = []
    for cohort in ["development", "test"]:
        for policy in ["greedy", "uniform", "interpolation"]:
            rs = [r for r in rows if r["cohort"] == cohort and r["policy"] == policy]
            if not rs:
                continue
            item = dict(cohort=cohort, policy=policy, cases=len(rs))
            for key in [
                "mse",
                "mae",
                "psnr",
                "ssim",
                "unobserved_mae",
                "early_mse",
                "late_mse",
                "warm_frame_seconds",
                "total_seconds",
            ]:
                item[key] = float(np.mean([r[key] for r in rs]))
            if all("same_observation_interpolation_mse" in r for r in rs):
                item["same_observation_interpolation_mse"] = float(
                    np.mean([r["same_observation_interpolation_mse"] for r in rs])
                )
            if all("ef" in r for r in rs):
                item["ef_mae"] = float(np.mean([r["ef"]["absolute_error"] for r in rs]))
            summaries.append(item)
    atomic(
        root / "summary.json",
        dict(
            rows=summaries,
            precision="fp32",
            loss=cfg["loss"],
            paper_exact_reproduction=False,
            cpu_functional=functional,
        ),
    )
    if rows:
        keys = [
            "cohort",
            "policy",
            "case",
            "mse",
            "psnr",
            "ssim",
            "unobserved_mae",
            "warm_frame_seconds",
            "total_seconds",
            "early_mse",
            "late_mse",
        ]
        with (root / "cases.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: r[k] for k in keys} for r in rows)
    table = "<table><tr><th>名单</th><th>策略</th><th>病例</th><th>PSNR dB</th><th>SSIM</th><th>未测MAE</th><th>同步暖帧秒</th></tr>"
    for row in summaries:
        table += (
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row[k]) if k in ['cohort', 'policy', 'cases'] else f'{row[k]:.5f}')}</td>"
                for k in [
                    "cohort",
                    "policy",
                    "cases",
                    "psnr",
                    "ssim",
                    "unobserved_mae",
                    "warm_frame_seconds",
                ]
            )
            + "</tr>"
        )
    table += "</table>"
    text = """<!doctype html><meta charset='utf-8'><title>独立前馈信念滤波结果</title>
<style>body{max-width:1100px;margin:30px auto;font:16px/1.8 sans-serif;padding:20px}img{max-width:100%}td,th{border:1px solid #bbb;padding:7px}table{border-collapse:collapse}</style>
<h1>独立动态逐帧信念滤波基线</h1>"""
    if functional:
        text += "<p><b>CPU合成功能测试；不得作为科研结果。</b></p>"
    text += "<p>依据论文第四章机制独立实现，没有学长源码或权重。Codec、prior与filter均自行训练；只有filter阶段冻结前两者。网络宽度、潜维度、初始化、损失归一化、历史截断、残差传播、分数归一化和训练预算是本实现选择。固定10+4、FP32，不调用DPS或训练EF模型。</p>"
    text += (
        table
        + "<img src='training_curves.png' alt='Actual training objectives and case-level validation MSE'/><p>左列是实际优化目标，右列是独立病例验证MSE，数值单位不同；未按训练曲线形状宣布收敛。主质量指标为[-1,1]浮点极坐标PSNR/SSIM/MAE，不与旧uint8或其他域数字直接拼表。</p>"
    )
    ft = cfg["training"]["filter"]
    text += f"<p>Codec评价使用完整输入，是表示诊断；prior的一步预测与复制上一帧均使用真实上一帧，递归预测另标为首帧特权输入诊断，不是在线成像结果。Filter完整视频不周期清空软记忆；每帧硬观测清空，最终已测列逐像素一致。训练有{ft['burn_in']}帧burn-in及{ft['bptt']}帧TBPTT，而非旧版每帧截断。片段训练与无限长推理仍存在差异，需要看长序列误差。</p>"
    text += "<p>greedy与uniform都运行相同filter和两次更新；插值为固定14条均匀观测，不经过学习网络。same_observation_interpolation_mse额外使用学习策略的实际观测位置进行插值，只检验给定观测下的重建增益。推理速度为同步模型/策略/观测时间，排除质量计算及写盘，完整病例耗时单列。冷帧、p50/p95见逐病例记录；不是论文硬件/数据的精确复现，也未重跑CASL比较。</p>"
    text += (
        "<p>EF："
        + (
            "只在最终评价读取已有冻结权重，不计算选线梯度。"
            if cfg["evaluation"]["ef_enabled"]
            else "本轮未启用，未评估；缺失不冒充任务性能保持。"
        )
        + "</p><p>结果包只含配置、名单、日志、指标和少量样例；模型与Adam检查点单独打包。没有完整训练轨迹或DPS历史缓存。流程completed仅表示执行完毕，不证明质量达到论文或可直接推广其他预算。</p>"
    )
    for name in galleries:
        text += f"<img src='{name}' alt='Locked first-case samples at fixed ranges'/><p>固定名单首病例、四个等距帧，不按效果挑图。误差与不确定性采用固定显示范围，后者是未校准代理，不与误差数值直接等同。</p>"
    (root / "REPORT_中文.html").write_text(text, encoding="utf-8")
    if not (root / "complete.json").exists():
        return {}
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix not in [".pt", ".tmp"]
        and not p.name.startswith(".")
        and p.name not in ["export.json", "STOP"]
        and p != root / "status.json"
    ]
    analysis = archive(root, files, root.with_name(root.name + ".analysis.tar.gz"))
    weights = [p for p in root.glob("*/*.pt")] + [
        root / "identity.json",
        root / "config.json",
        root / "manifest.json",
    ]
    models = archive(root, weights, root.with_name(root.name + ".models.tar.gz"))
    receipt = dict(analysis=analysis, models=models)
    atomic(root / "export.json", receipt)
    return receipt
