"""Six bounded retrospective blocks, using existing136 assets on CPU only."""
# ruff: noqa: E402

import argparse
import html
import json
import os
import platform
import tarfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "2"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from budget136_analysis_core import (
    ARMS,
    LEVELS,
    Source,
    analyze_models,
    forward,
    read,
    samples,
    sha,
    softmax,
    tree_load,
    write_csv,
)
from queue_task_budget_analysis_136 import output_lock

COLORS = {"T1": "#0072B2", "Q2": "#009E73", "T2": "#D55E00"}
MARKERS = {"T1": "o", "Q2": "s", "T2": "^"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def save(fig, out, name, synthetic):
    if synthetic:
        fig.suptitle("SYNTHETIC FUNCTIONAL ONLY", color="#A00000", fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def trailing_median(values, width=16):
    x = np.asarray(values)
    return np.array([np.median(x[max(0, i - width + 1) : i + 1]) for i in range(len(x))])


def plots(source, out, models, pcas, sample_data, synthetic):
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "figure.facecolor": "white"}):
        fig, axs = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
        for row, arm in enumerate(ARMS):
            records = source.training[arm]
            updates = [r["update"] for r in records]
            fields = ["mse" if arm == "Q2" else "ef_error", "mean_lines", "gradient_norm"]
            for col, (key, label) in enumerate(
                zip(
                    fields,
                    [
                        "MSE" if arm == "Q2" else "EF error (pp)",
                        "Acquired lines/frame",
                        "Gradient norm",
                    ],
                )
            ):
                vals = [r[key] for r in records]
                ax = axs[row, col]
                ax.scatter(updates, vals, color="#888888", alpha=0.35, s=7, label="raw episodes")
                ax.plot(
                    updates,
                    trailing_median(vals),
                    color=COLORS[arm],
                    label="trailing median,16 updates",
                )
                if key == "gradient_norm":
                    ax.plot(
                        updates,
                        trailing_median([r["task_gradient_norm"] for r in records]),
                        color="#333333",
                        linestyle="--",
                        label="task component",
                    )
                    ax.set_yscale("symlog", linthresh=1e-5)
                if key == "mean_lines":
                    ax.set_ylim(0, 28)
                for boundary in [16, 72]:
                    ax.axvline(boundary, color="#777777", linestyle=":", alpha=0.6)
                ax.set(xlabel="Optimizer updates", ylabel=label, title=arm)
                ax.grid(alpha=0.15)
                if row == 0:
                    ax.legend(fontsize=7)
        save(fig, out, "training_curves", synthetic)
        fig, axs = plt.subplots(3, 1, figsize=(12, 7), layout="constrained")
        for ax, arm in zip(axs, ARMS):
            r = source.training[arm]
            u = [x["update"] for x in r]
            v = [x["objective"] for x in r]
            ax.scatter(u, v, color="#777777", alpha=0.35, s=7)
            ax.plot(u, trailing_median(v), color=COLORS[arm], label="trailing median16")
            ax.set(xlabel="Updates", ylabel="Declared episode objective", title=arm)
            ax.legend()
            ax.grid(alpha=0.15)
        save(fig, out, "episode_objectives", synthetic)
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for arm in ARMS:
            values = []
            for phase in ["full_pass_1_evaluate", "full_pass_2_evaluate"]:
                rows = [r for r in source.cases[phase] if r["arm"] == arm]
                values.append(
                    (
                        rows[0]["update"],
                        np.mean([r["ef_error"] for r in rows]),
                        np.mean([r["mean_lines"] for r in rows]),
                        np.mean([r["mse_ratio"] for r in rows]) * 100,
                    )
                )
            for ax, col in zip(axs, [1, 2, 3]):
                ax.plot(
                    [v[0] for v in values],
                    [v[col] for v in values],
                    marker=MARKERS[arm],
                    color=COLORS[arm],
                    label=arm,
                )
        for ax, label in zip(axs, ["EF MAE (pp)", "Lines/frame", "MSE change vs M (%)"]):
            ax.set(xlabel="Updates; same development videos", ylabel=label, xticks=[72, 136])
            ax.grid(alpha=0.15)
            ax.legend()
        axs[2].axhline(0, color="#888888", linestyle="--")
        save(fig, out, "development_checkpoints", synthetic)
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, arm in zip(axs, ARMS):
            values = models[arm]["checkpoints"]
            steps = [1, 16, 72, 136]
            ax.plot(
                steps,
                [values[str(k)]["expected_unmasked"] for k in steps],
                marker="o",
                label="unmasked expected active-stage lines",
            )
            ax.axhline(
                7,
                color="#777777",
                linestyle="--",
                label="quota target" if arm != "Q2" else "fixed-M reference",
            )
            ax.set(
                xlabel="Checkpoint, common136 states",
                ylabel="Expected active-stage lines",
                title=arm,
                ylim=(0, 14),
            )
            ax.legend(fontsize=7)
            ax.grid(alpha=0.15)
        save(fig, out, "policy_learning_replay", synthetic)
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, arm in zip(axs, ARMS):
            p, ids, frames, clock, quota = pcas[arm]
            point = ax.scatter(p[:, 0], p[:, 1], c=clock, cmap="viridis", s=5, alpha=0.6)
            ratio = models[arm]["pca_variance_ratio"]
            ax.set(
                xlabel=f"PC1 ({ratio[0] * 100:.1f}%)",
                ylabel=f"PC2 ({ratio[1] * 100:.1f}%)",
                title=f"{arm}: separate PCA coordinates",
            )
            fig.colorbar(point, ax=ax, label="Transformed horizon/clock input")
        save(fig, out, "representation_pca", synthetic)
        fig, axs = plt.subplots(3, 1, figsize=(12, 7), layout="constrained")
        for ax, arm in zip(axs, ARMS):
            ax.bar(np.arange(18), models[arm]["local_sensitivity_sd_scaled"], color=COLORS[arm])
            ax.set(
                xticks=np.arange(18),
                xlabel="Feature index; semantics in report",
                ylabel="|local derivative| x feature SD",
                title=arm,
            )
        save(fig, out, "input_sensitivity", synthetic)
        for ci, (name, data) in enumerate(sample_data.items()):
            fig, axs = plt.subplots(4, 5, figsize=(12, 10), layout="constrained")
            for ri, at in enumerate(data["indexes"]):
                for col, arm in enumerate(["GT", "M", *ARMS]):
                    image = data["target"][ri] if arm == "GT" else data["models"][arm]["images"][ri]
                    axs[ri, col].imshow(
                        image, cmap="gray", vmin=-1, vmax=1, interpolation="nearest"
                    )
                    budget = (
                        ""
                        if arm == "GT"
                        else f" {data['models'][arm]['rows'][ri]['k1']}+{data['models'][arm]['rows'][ri]['k2']}"
                    )
                    axs[ri, col].set_title(f"frame {at} / {arm}{budget}", fontsize=8)
                    axs[ri, col].set_xticks([])
                    axs[ri, col].set_yticks([])
            save(fig, out, f"sample_{ci + 1}_reconstruction", synthetic)
            fig, axs = plt.subplots(4, 4, figsize=(10, 10), layout="constrained")
            vmax = max(
                float(np.abs(m["images"] - data["target"]).max()) for m in data["models"].values()
            )
            for col, arm in enumerate(["M", *ARMS]):
                for ri in range(4):
                    im = axs[ri, col].imshow(
                        np.abs(data["models"][arm]["images"][ri] - data["target"][ri]),
                        cmap="magma",
                        vmin=0,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                    axs[ri, col].set_title(f"{arm}, frame {data['indexes'][ri]}", fontsize=8)
                    axs[ri, col].set_xticks([])
                    axs[ri, col].set_yticks([])
            fig.colorbar(
                im,
                ax=list(axs.ravel()),
                label="Absolute normalized-pixel error; common range within case",
            )
            save(fig, out, f"sample_{ci + 1}_error", synthetic)
            fig, axs = plt.subplots(4, 8, figsize=(15, 10), layout="constrained")
            cmap = plt.get_cmap("gray").with_extremes(bad="#000000")
            from matplotlib.colors import ListedColormap

            line_cmap = ListedColormap(["#111111", "#0072B2", "#E69F00"])
            for col, arm in enumerate(["M", *ARMS]):
                model = data["models"][arm]
                for ri in range(4):
                    mask = model["masks"][ri].astype(bool)
                    observation = np.ma.masked_where(~mask, data["target"][ri])
                    lines = np.zeros_like(mask, dtype=np.uint8)
                    lines[:, model["rows"][ri]["lines1"]] = 1
                    lines[:, model["rows"][ri]["lines2"]] = 2
                    if not np.array_equal(lines > 0, mask):
                        raise ValueError("Selected-line image differs from cached physical mask")
                    axs[ri, 2 * col].imshow(
                        observation, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest"
                    )
                    axs[ri, 2 * col + 1].imshow(
                        lines, cmap=line_cmap, vmin=0, vmax=2, interpolation="nearest"
                    )
                    axs[ri, 2 * col].set_title(f"{arm} sparse observation", fontsize=7)
                    axs[ri, 2 * col + 1].set_title("blue=stage1; orange=stage2", fontsize=6)
            for ax in axs.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            for ri, at in enumerate(data["indexes"]):
                axs[ri, 0].set_ylabel(f"frame {at}")
            save(fig, out, f"sample_{ci + 1}_observations", synthetic)
            fig, axs = plt.subplots(4, 4, figsize=(10, 10), layout="constrained")
            finite = [m["variance"][np.isfinite(m["variance"])] for m in data["models"].values()]
            variance_max = max([float(a.max()) for a in finite if len(a)] or [1.0])
            variance_max = variance_max if variance_max > 0 else 1.0
            for col, arm in enumerate(["M", *ARMS]):
                for ri in range(4):
                    im = axs[ri, col].imshow(
                        data["models"][arm]["variance"][ri],
                        cmap=plt.get_cmap("viridis").with_extremes(bad="#777777"),
                        vmin=0,
                        vmax=variance_max,
                        interpolation="nearest",
                    )
                    axs[ri, col].set_title(f"{arm},frame{data['indexes'][ri]}", fontsize=8)
                    axs[ri, col].set_xticks([])
                    axs[ri, col].set_yticks([])
            fig.colorbar(
                im,
                ax=list(axs.ravel()),
                label="Raw posterior variance,2 particles; not calibrated confidence",
            )
            save(fig, out, f"sample_{ci + 1}_particle_variance", synthetic)
            fig, axs = plt.subplots(3, 2, figsize=(12, 8), layout="constrained")
            rows = {
                arm: next(
                    r
                    for r in source.cases["full_pass_2_evaluate"]
                    if r["arm"] == arm and r["case"] == name
                )
                for arm in ARMS
            }
            for ri, arm in enumerate(ARMS):
                seq = source.json(source.inside(rows[arm]["trajectory"]) / "rows.json")
                axs[ri, 0].step(
                    [r["frame"] for r in seq],
                    [r["k1"] for r in seq],
                    where="post",
                    label="stage1",
                    color=COLORS[arm],
                )
                axs[ri, 0].step(
                    [r["frame"] for r in seq],
                    [r["k2"] for r in seq],
                    where="post",
                    label="stage2",
                    color="#555555",
                    linestyle="--",
                )
                axs[ri, 0].set(title=arm, ylabel="Lines", ylim=(0, 15))
                axs[ri, 0].legend(fontsize=8)
                cache = source.inside(rows[arm]["trajectory"])
                trace = tree_load(cache / "controller.npz")["trace"]
                x = np.stack([t["sequence"] for t in trace])
                v = np.stack([t["valid"] for t in trace])
                prob = softmax(forward(source.params[arm][136], x, v)[0])
                expected = prob @ LEVELS[arm]
                axs[ri, 1].plot(
                    [t["frame"] for t in trace],
                    expected,
                    label="unmasked expectation",
                    color=COLORS[arm],
                )
                forced = np.array([t["quota_forced"] for t in trace])
                xs = np.array([t["frame"] for t in trace])
                axs[ri, 1].scatter(
                    xs[forced],
                    expected[forced],
                    marker="x",
                    color="black",
                    s=15,
                    label="quota-forced",
                )
                axs[ri, 1].set(
                    title="Policy preference before quota",
                    ylabel="Active-stage lines",
                    ylim=(0, 14),
                )
                axs[ri, 1].legend(fontsize=7)
            for ax in axs.ravel():
                ax.set_xlabel("Frame index")
                ax.grid(alpha=0.15)
            save(fig, out, f"sample_{ci + 1}_budget", synthetic)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for arm in ARMS:
            by_case = {
                r["case"]: r for r in source.cases["full_pass_2_evaluate"] if r["arm"] == arm
            }
            rows = [by_case[name] for name in source.case_names]
            for ax, key, label in [
                (axs[0], "ef_vs_fixed", "EF error minus M (pp)"),
                (axs[1], "mse_ratio", "MSE ratio minus M"),
            ]:
                vals = [r[key] for r in rows]
                ax.scatter(
                    range(1, len(rows) + 1), vals, color=COLORS[arm], marker=MARKERS[arm], label=arm
                )
                ax.set(xlabel="All development cases, manifest order", ylabel=label)
                ax.axhline(0, color="#777777", linestyle="--")
                ax.legend()
                ax.grid(alpha=0.15)
        save(fig, out, "case_heterogeneity", synthetic)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        q = source.training["Q2"]
        for i, label in enumerate(["mean MSE dual", "p90 dual"]):
            axs[0].plot([r["update"] for r in q], [r["dual_after"][i] for r in q], label=label)
        axs[0].set(xlabel="Q2 updates", ylabel="Constraint multiplier")
        axs[0].legend()
        for arm in ARMS:
            r = source.training[arm]
            axs[1].plot(
                [x["update"] for x in r],
                trailing_median([x["advantage"] for x in r]),
                label=arm,
                color=COLORS[arm],
            )
        axs[1].set(xlabel="Updates", ylabel="Reward advantage; trailing median16")
        axs[1].axhline(0, color="#777777", linestyle="--")
        axs[1].legend()
        save(fig, out, "reward_and_constraints", synthetic)


def make_report(source, out, models, gaps, synthetic):
    body = [
        "<h1>动态预算136次：已有权重与缓存的六项离线分析</h1>",
        "<p><b>CPU-only、无重训、无新增DPS或EF推理。</b>只使用已完成的开发集和既有轨迹。未运行136次周期对照，不能由这些诊断宣布时间分配有效。</p>",
        "<p>"
        + (
            "SYNTHETIC FUNCTIONAL ONLY，不能提供医学效果证据。"
            if synthetic
            else "真实实验记录的回顾性分析；只含单次训练、评价种子42。"
        )
        + "</p>",
        "<h2>1. 训练是否持续改善</h2><p>原始点是不同患者的整视频更新；灰点保留，彩线为过去16次更新的中位数。16、72为课程边界。训练曲线不等于相同病例上的收敛曲线。开发72→136始终使用同一名单。</p>",
    ]
    for name in [
        "training_curves",
        "episode_objectives",
        "development_checkpoints",
        "reward_and_constraints",
    ]:
        body.append(f'<img src="{name}.png" alt="{name}"/>')
    body.append(
        "<p>Q2的对偶变量随训练变化，目标函数值的变化也含权重变化；不能仅由该曲线宣布质量收敛。</p>"
    )
    body.extend(
        [
            "<h2>2–3. 预算变化来自模型还是配额；控制器学了什么</h2>",
            "<p>共同输入取各模型136次真实轨迹；重放1、16、72、136权重。第1次是早期检查点，不冒充未保存的初始化。只改变权重，保持观察历史与合法动作掩码；这些重放不是重新执行采集闭环。</p>",
            "<table><tr><th>模型</th><th>未屏蔽期望线数</th><th>配额强制比例</th><th>状态输出 vs 仅输出偏置 JS</th><th>帧内输出 vs 视频均值 JS</th><th>最近一帧重放 JS</th></tr>",
        ]
    )
    for arm in ARMS:
        d = models[arm]["checkpoints"]["136"]
        body.append(
            f"<tr><td>{arm}</td><td>{d['expected_unmasked']:.3f}</td><td>{d['forced_fraction']:.2%}</td><td>{d['state_vs_bias_js']:.6f}</td><td>{d['within_video_probability_js']:.6f}</td><td>{models[arm]['latest_only']['js_case_mean']:.6f}</td></tr>"
        )
    body.extend(
        [
            "</table><p>JS是概率分布差异，不能换算为EF收益。配额之后的高置信概率可能是只剩一个动作；应结合未屏蔽概率和偏置对照解释。预算波动不自动证明利用了状态。CPU重放概率与实际记录按atol=rtol=2e-4核对。</p>",
            '<img src="policy_learning_replay.png" alt="common-state weight replay"/>',
            "<h2>4. 重建与预算样例</h2><p>固定开发名单前三例、每例四个四等分区间中点。展示极坐标深度×角度索引域，非原始RF。原生数据112×112，图中仅nearest显示放大；灰度统一[-1,1]，未逐图拉伸。误差色标在每病例四种方法间统一。原始样例数组保留selected_samples.npz。</p>",
        ]
    )
    for i, name in enumerate(source.case_names[:3]):
        body.append(f"<h3>{html.escape(name)}</h3>")
        for kind in ["reconstruction", "error", "observations", "particle_variance", "budget"]:
            body.append(
                f'<img src="sample_{i + 1}_{kind}.png" alt="{kind} of {html.escape(name)}"/>'
            )
    body.extend(
        [
            "<p>缓存两粒子方差保留在样例数组，不把两粒子方差视为可靠置信区间；若缺快照会列为缺失并显示灰色，不重新跑采样。sample_uncertainty_error.csv同时记录未观测误差、粒子方差与显示范围外像素比例。</p>",
            "<h2>5. 表征与输入依赖</h2><p>PCA仅描述各控制器的GRU隐状态，视频等权计算协方差；三个坐标系独立，不按PC轴或神经元编号作跨模型语义比较。不用二维聚类证明心动周期理解或泛化。</p>",
            '<img src="representation_pca.png" alt="separate hidden-state PCA"/>',
            "<p>特征0–2为不确定性；4–5为观测残差/覆盖；6–9为任务摘要；10–11为历史进度/上次预算；16–17为额度/时钟。12–15在T1与第二阶段语义不同：T1含历史图像变化，第二阶段含采集前预测创新、中间后验变化及初采量。具体18项来自原策略代码。</p>",
            "<p>局部敏感性只扰动当前已变换特征±0.001，乘当前开发状态的描述性标准差；不是训练归一化或物理干预。representation_cue_correlations.csv按病例分别记录输出/PC与任务、不确定性、创新及资源时钟的Spearman相关；不把同一视频帧当独立样本，不给相关性作因果解释。历史截断、屏蔽资源组是可能偏离分布的输入重放，不能当作真实闭环任务消融。</p>",
            '<img src="input_sensitivity.png" alt="local input sensitivity"/>',
            "<h2>6. 全病例变化与优化线索</h2>",
            '<img src="case_heterogeneity.png" alt="all development cases"/>',
            "<p>先检查输出是否主要由偏置及额度驱动，再看状态/历史依赖及病例差异；如Q2靠近约束边界，应结合难帧而非只看均值。若表征依赖时钟，这提示后续对预算规则或输入做独立实验，不能由相关性直接断言原因。</p>",
            "<p>没有新增盲测、分割训练、任务模型、planning/WM或GPU周期对照。该包用于解释和设计下一轮，不自动选用新baseline。</p>",
            "<h2>缺失与数据</h2><pre>"
            + html.escape(json.dumps(gaps, ensure_ascii=False, indent=2))
            + "</pre>",
        ]
    )
    body.append(
        "<p>数据：analysis.json、policy_frames.csv、weight_changes.csv、representation_pca.csv、local_sensitivity.csv、training_records.csv、development_cases.csv、provenance.json。</p>"
    )
    text = (
        '<!doctype html><meta charset="utf-8"><title>136离线分析</title><style>body{max-width:1150px;margin:30px auto;font:16px/1.65 sans-serif;padding:0 20px}img{max-width:100%;height:auto}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:7px}pre{white-space:pre-wrap}</style>'
        + "\n".join(body)
    )
    (out / "REPORT_中文.html").write_text(text, encoding="utf-8")


def run(source_root, out):
    out = Path(out).resolve()
    source_root = Path(source_root).resolve()
    if out == source_root or out.is_relative_to(source_root):
        raise ValueError("Analysis output must not overwrite the source batch")
    with output_lock(out / ".analysis.lock"):
        return run_impl(source_root, out)


def run_impl(source_root, out):
    source_root = Path(source_root).resolve()
    out = Path(out).resolve()
    if out == source_root or out.is_relative_to(source_root):
        raise ValueError("Analysis output must not overwrite the source batch")
    out.mkdir(parents=True, exist_ok=True)
    code_files = [
        Path(__file__),
        Path(__file__).with_name("budget136_analysis_core.py"),
        Path(__file__).with_name("queue_task_budget_analysis_136.py"),
    ]
    code = {p.name: sha(p) for p in code_files}
    marker = out / "complete.json"
    if marker.exists():
        previous = read(marker)
        if previous["code"] != code:
            raise ValueError("Analysis code changed; use a new analysis output")
        if not all(sha(p) == v for p, v in previous["source_hashes"].items()):
            raise ValueError("Frozen analysis source changed")
        if not all(sha(out / p) == v for p, v in previous["output_hashes"].items()):
            raise ValueError("Analysis output corrupted")
        return previous
    started = time.monotonic()
    atomic_json(out / "run_status.json", dict(status="running", cpu_only=True))
    try:
        source = Source(source_root)
        synthetic = source.config.get("stage_budget", {}).get("functional_fixture", False)
        models, pcas = analyze_models(source, out)
        sample_data, gaps = samples(source, out)
        cols = [
            "arm",
            "update",
            "case",
            "section",
            "ef_error",
            "mse",
            "mean_lines",
            "objective",
            "reward",
            "advantage",
            "gradient_norm",
            "task_gradient_norm",
            "seconds",
        ]
        records = [{k: r.get(k) for k in cols} for arm in ARMS for r in source.training[arm]]
        write_csv(out / "training_records.csv", records)
        case_rows = [
            dict(
                phase=phase,
                **{
                    k: r.get(k)
                    for k in [
                        "arm",
                        "update",
                        "case",
                        "seed",
                        "ef_error",
                        "mean_lines",
                        "mse",
                        "mse_ratio",
                        "p90_ratio",
                        "ef_vs_fixed",
                        "quota_forced_fraction",
                    ]
                },
            )
            for phase, rows in source.cases.items()
            for r in rows
        ]
        write_csv(out / "development_cases.csv", case_rows)
        data = dict(
            models=models,
            gaps=gaps,
            cases=len(source.case_names),
            cpu_only=True,
            synthetic_functional_only=synthetic,
            scope="six retrospective blocks; no task inference or physical closed-loop ablation",
        )
        atomic_json(out / "analysis.json", data)
        plots(source, out, models, pcas, sample_data, synthetic)
        make_report(source, out, models, gaps, synthetic)
        provenance = dict(
            source=str(source_root),
            source_hashes=source.hashes,
            code=code,
            software=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                matplotlib=matplotlib.__version__,
            ),
            graphics=dict(
                png_dpi=180,
                formats=["png", "pdf"],
                audience="general research report; no publisher specified",
                smoothing="training trailing median16; raw points retained",
                pca="separate model PCA, equal video covariance weights",
                inference="CPU numpy GRU replay with saved inputs; not new ultrasound inference",
                examples="first3 locked development cases,4 deterministic temporal block centers",
                image_range=[-1, 1],
                image_interpolation="nearest",
                native_shape=[112, 112],
            ),
            gaps=gaps,
            seed=20261011,
        )
        atomic_json(out / "provenance.json", provenance)
        result = dict(
            status="completed_with_gaps" if gaps else "completed",
            seconds=time.monotonic() - started,
            cpu_only=True,
            code=code,
            source_hashes=source.hashes,
            output_hashes={
                p.name: sha(p)
                for p in out.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.name
                not in [
                    "complete.json",
                    "run_status.json",
                    "queue_status.json",
                    "analysis.console.log",
                    "bundle_receipt.json",
                ]
            },
        )
        atomic_json(out / "complete.json", result)
        atomic_json(
            out / "run_status.json",
            dict(status=result["status"], seconds=result["seconds"], cpu_only=True),
        )
        return result
    except BaseException as error:
        atomic_json(
            out / "run_status.json",
            dict(status="failed", error=f"{type(error).__name__}: {error}", cpu_only=True),
        )
        raise


def bundle(out):
    out = Path(out)
    read(out / "complete.json")
    files = [
        p
        for p in out.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.name not in ["bundle_receipt.json", "queue_status.json", "analysis.console.log"]
    ]
    hashes = {p.name: sha(p) for p in files}
    old_receipt = out / "bundle_receipt.json"
    if old_receipt.exists():
        old = read(old_receipt)
        if (
            old["source_hashes"] == hashes
            and Path(old["archive"]).exists()
            and sha(old["archive"]) == old["sha256"]
        ):
            return old
    target = out.parent / f"{out.name}.analysis_{time.time_ns()}.tar.gz"
    partial = target.with_suffix(".partial")
    with tarfile.open(partial, "w:gz") as t:
        for p in files:
            t.add(p, arcname=str(Path(out.name) / p.name), recursive=False)
    with tarfile.open(partial, "r:gz") as t:
        import hashlib

        for member in t:
            if (
                hashlib.sha256(t.extractfile(member).read()).hexdigest()
                != hashes[Path(member.name).name]
            ):
                raise ValueError("Analysis bundle verification failed")
    partial.replace(target)
    receipt = dict(
        archive=str(target),
        sha256=sha(target),
        bytes=target.stat().st_size,
        verified=True,
        source_hashes=hashes,
        scope="offline plots/tables and small selected arrays; excludes model and full-video weights/cache",
    )
    atomic_json(out / "bundle_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.source, args.output)
    print(json.dumps(bundle(args.output), ensure_ascii=False))
