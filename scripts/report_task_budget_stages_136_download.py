"""Summarize verified downloaded results. Offline only; never starts research jobs."""

import argparse
import csv
import hashlib
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("T1", "Q2", "T2")
COLORS = {"T1": "#0072B2", "Q2": "#009E73", "T2": "#D55E00"}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def ci(values):
    # The unit is a case; not frames or EF windows. Descriptive, not a success gate.
    x = np.asarray(values)
    rng = np.random.default_rng(20261011)
    return np.percentile(x[rng.integers(0, len(x), (10000, len(x)))].mean(1), [2.5, 97.5]).tolist()


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows
        )
        + "</tbody></table>"
    )


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    closure = data / "closure/task_budget_stages_v1"
    offline = data / "offline/task_budget_stages_v1.offline136_v1"
    out.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for prefix, source in [("015", closure), ("016", offline)]:
        receipt = read(data / f"{prefix}.remote_receipt.json")
        archive = data / "archives" / Path(receipt["archive"]).name
        assert sha(archive) == receipt["sha256"]
        for name, value in receipt["source_hashes"].items():
            p = source / name
            assert sha(p) == value, name
            evidence[str(p)] = value
    scope = read(closure / "scope_closure_136.json")
    assert scope["status"] == "completed" and scope["selected_checkpoints"] == dict.fromkeys(
        ARMS, 136
    )
    assert not scope["confirmation_evaluated"] and not scope["matched_periodic_at136_evaluated"]
    done = read(offline / "complete.json")
    analysis = read(offline / "analysis.json")
    assert done["status"] == "completed" and not analysis["gaps"]
    manifest = read(closure / "manifest.json")
    names = manifest["cohorts"]["development"]
    assert len(names) == 16
    fixed = read(closure / "jobs/full_baselines/result.json")["cases"]
    fixed_groups = {}
    for pair in [(7, 0), (7, 4), (7, 7), (7, 14)]:
        rows = [r for r in fixed if tuple(r["pair"]) == pair]
        assert {r["case"] for r in rows} == set(names) and len(rows) == 16
        fixed_groups[f"{pair[0]}+{pair[1]}"] = rows
    baseline = {r["case"]: r for r in fixed_groups["7+7"]}
    fields = ["ef_error", "mean_lines", "psnr", "ssim", "mse", "inference_seconds"]
    fixed_summary = {k: {f: mean(v, f) for f in fields} for k, v in fixed_groups.items()}
    performance, per_case = {}, []
    for update, phase in [(72, "full_pass_1_evaluate"), (136, "full_pass_2_evaluate")]:
        rows = read(closure / "jobs" / phase / "result.json")["cases"]
        for arm in ARMS:
            rs = [r for r in rows if r["arm"] == arm]
            assert len(rs) == 16 and {r["case"] for r in rs} == set(names)
            assert all(r["update"] == update and r["seed"] == 42 for r in rs)
            for r in rs:
                assert np.isclose(r["ef_vs_fixed"], r["ef_error"] - baseline[r["case"]]["ef_error"])
                per_case.append(
                    {
                        k: r[k]
                        for k in [
                            "case",
                            "arm",
                            "update",
                            "ef_error",
                            "mean_lines",
                            "mse_ratio",
                            "p90_ratio",
                            "ef_vs_fixed",
                        ]
                    }
                )
            summary = {
                f: mean(rs, f)
                for f in fields
                + [
                    "mse_ratio",
                    "p90_ratio",
                    "ef_vs_fixed",
                    "zero_second_fraction",
                    "quota_forced_fraction",
                    "quota_constrained_fraction",
                ]
            }
            summary.update(
                ef_delta_ci=ci([r["ef_vs_fixed"] for r in rs]),
                ef_better_cases=sum(r["ef_vs_fixed"] < 0 for r in rs),
                both_quality_goals_cases=sum(
                    r["mse_ratio"] <= 0.05 and r["p90_ratio"] <= 0.1 for r in rs
                ),
                line_saving=1 - summary["mean_lines"] / 14,
            )
            performance[f"{arm}_{update}"] = summary
    training = {}
    for arm in ARMS:
        records = [
            read(closure / "training" / arm / "updates" / f"{i:05d}/record.json")
            for i in range(1, 137)
        ]
        assert [r["update"] for r in records] == list(range(1, 137))
        assert all(np.isfinite(r["gradient_norm"]) and r["gradient_norm"] > 0 for r in records)
        by_case = defaultdict(list)
        for r in records:
            by_case[r["case"]].append(r)
        assert len(by_case) == 64
        key = "mse" if arm == "Q2" else "ef_error"
        delta = [v[-1][key] - v[0][key] for v in by_case.values()]
        training[arm] = dict(
            updates=136,
            unique_cases=64,
            median_gradient=float(np.median([r["gradient_norm"] for r in records])),
            first_last_case_error_change=float(np.mean(delta)),
            first_last_improved=sum(x < 0 for x in delta),
            final_dual=records[-1]["dual_after"],
        )
    with (offline / "policy_frames.csv").open(encoding="utf-8-sig") as f:
        frames = list(csv.DictReader(f))
    quarter, variability = [], {}
    for arm in ARMS:
        by_case = defaultdict(list)
        for row in frames:
            if row["arm"] == arm:
                by_case[row["case"]].append(row)
        assert set(by_case) == set(names)
        variability[arm] = dict(
            chosen_stage_sd=float(
                np.mean([np.std([float(x["chosen_lines"]) for x in xs]) for xs in by_case.values()])
            ),
            unmasked_expectation_sd=float(
                np.mean(
                    [np.std([float(x["expected_unmasked"]) for x in xs]) for xs in by_case.values()]
                )
            ),
            observed_unmasked_preference_counts=dict(
                (
                    value,
                    sum(x["unmasked_preference"] == value for xs in by_case.values() for x in xs),
                )
                for value in sorted(
                    {x["unmasked_preference"] for xs in by_case.values() for x in xs}
                )
            ),
        )
        for q in range(4):
            case_values = []
            for name in names:
                n = manifest["files"][name]["frames"] - 1
                xs = [x for x in by_case[name] if min(3, int(4 * (int(x["frame"]) - 1) / n)) == q]
                assert xs
                case_values.append(
                    [
                        np.mean([float(x["chosen_lines"]) for x in xs]),
                        np.mean([float(x["expected_unmasked"]) for x in xs]),
                        np.mean([x["forced"] == "True" for x in xs]),
                    ]
                )
            v = np.mean(case_values, axis=0)
            quarter.append(
                dict(
                    arm=arm,
                    quarter=q + 1,
                    chosen_lines=float(v[0]),
                    raw_expected_lines=float(v[1]),
                    forced_fraction=float(v[2]),
                )
            )
    initial = {}
    for arm, levels in [("T1", [4, 7, 10, 14]), ("Q2", [0, 2, 4, 7, 14]), ("T2", [0, 2, 4, 7, 14])]:
        bias = np.full(len(levels), -0.5)
        bias[levels.index(7)] = 0.5
        p = np.exp(bias) / np.exp(bias).sum()
        initial[arm] = dict(bias_only_probabilities=p.tolist(), expected_lines=float(p @ levels))
    result = dict(
        performance=performance,
        fixed=fixed_summary,
        training=training,
        variability=variability,
        quarter=quarter,
        initial_bias=initial,
        cases=16,
        training_seed=42,
        evaluation_seed=42,
        case_bootstrap=dict(
            repeats=10000, seed=20261011, purpose="descriptive uncertainty; not exploration gate"
        ),
    )
    (out / "分析数据.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, rows in [("病例对照.csv", per_case), ("时间四分段预算.csv", quarter)]:
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axs = plt.subplots(2, 3, figsize=(12, 6), layout="constrained", sharex=True)
        for col, arm in enumerate(ARMS):
            rs = [r for r in quarter if r["arm"] == arm]
            axs[0, col].plot(
                [r["quarter"] for r in rs],
                [r["chosen_lines"] for r in rs],
                marker="o",
                color=COLORS[arm],
                label="actual sampled action",
            )
            axs[0, col].plot(
                [r["quarter"] for r in rs],
                [r["raw_expected_lines"] for r in rs],
                marker="s",
                ls="--",
                color="#333333",
                label="expectation before quota",
            )
            axs[0, col].set(
                title=arm, ylabel="Active-stage lines", ylim=(0, 14), xticks=[1, 2, 3, 4]
            )
            axs[0, col].legend(fontsize=8)
            axs[1, col].plot(
                [r["quarter"] for r in rs],
                [100 * r["forced_fraction"] for r in rs],
                color=COLORS[arm],
                marker="o",
            )
            axs[1, col].set(
                xlabel="Warm-video quarter",
                ylabel="Quota-forced frames (%)",
                ylim=(0, 100),
                xticks=[1, 2, 3, 4],
            )
            for ax in axs[:, col]:
                ax.grid(alpha=0.2)
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"budget_quarters.{ext}", dpi=180, facecolor="white")
        plt.close(fig)
    for name in [
        "training_curves.png",
        "development_checkpoints.png",
        "sample_1_budget.png",
        "sample_1_reconstruction.png",
        "input_sensitivity.png",
        "representation_pca.png",
    ]:
        shutil.copy2(offline / name, out / name)
    rows = [["固定7+7", "—", "14.000", f"{fixed_summary['7+7']['ef_error']:.4f}", "—", "0%", "0%"]]
    for arm in ARMS:
        for update in [72, 136]:
            r = performance[f"{arm}_{update}"]
            rows.append(
                [
                    arm,
                    update,
                    f"{r['mean_lines']:.3f}",
                    f"{r['ef_error']:.4f}",
                    f"{r['ef_vs_fixed']:+.4f}",
                    f"{r['mse_ratio']:+.2%}",
                    f"{r['p90_ratio']:+.2%}",
                ]
            )
    parts = [
        "<h1>动态预算T1/Q2/T2：136次更新实验分析</h1><p>2026-10-11 · 已下载校验 · 64训练视频、16开发视频 · 单次训练/评价种子42</p>",
        "<p class='lead'>本轮实现与收尾正常，获得了可用的学习、质量和配额诊断证据。T1/T2在72次更新时的EF改善没有延续到136次；Q2以约9%线数节省换来了约5%重建MSE增加。更关键的是，三个控制器对帧状态的预算概率响应很弱，T1/T2还存在明显的配额尾段强制。因此目前应归档为机制与实现探索，不能宣称已经学会了可靠的时序资源分配。</p>",
        "<h2>1. 实际完成了什么</h2><p>三模型各136次真实整视频更新，全部48个136检查点开发评价已提交。课程是8例先访问两遍（16更新），剩余56例（累计72），再访问全部64例（累计136）；不是136个epoch。CASL、EF和选线规则冻结，仅更新GRU预算头。264训练、最终选择、32例确认及136同资源周期对照依用户要求取消。原协调器stopped是正常收尾，缩减范围scope_closure_136=completed。CPU六项分析耗时56.5秒、缺失清单为空，训练/收尾/分析进程已退出，检查时GPU无计算进程。</p>",
        "<p>采集两阶段分别先根据历史、再根据当前观测与历史进行决策。T1学习第一阶段4/7/10/14、第二阶段固定7，用EF优化；T2第一阶段固定7、第二阶段学习0/2/4/7/14，用EF优化。两者总线数严格14×视频帧数。Q2第一阶段固定7、第二阶段学习相同档位，在平均MSE增幅≤5%、逐帧MSE p90增幅≤10%的训练目标下省预算。该质量目标按视频定义；下表则对16个视频的相对变化等权平均。</p>",
        "<h2>2. 最终性能与72次对照</h2>",
        table(
            [
                "方法",
                "更新",
                "平均线数",
                "EF MAE↓（百分点）",
                "EF相对固定变化↓",
                "病例平均MSE相对变化",
                "病例平均p90相对变化",
            ],
            rows,
        ),
        "<p>T1的EF改善从−0.3293降至−0.0158个百分点；T2从−0.1690降至−0.0078。136时基本回到固定7+7水平，不能由增加64次更新就期待稳定增益。MSE略改善不是这两个EF目标的成功门槛；也不能用MSE变坏单独否定EF探索。</p>",
        "<p>Q2在136时平均12.737条，比14条少9.02%；EF MAE增加0.0632个百分点。MSE增幅5.29%略超5%目标，p90增幅6.68%在10%之内。72时两项总体均值均在目标内，136并没有全面改善：p90改善、EF更接近参照，但MSE均值略变坏且省线略减少。136只有5/16病例同时满足两条目标，72为7/16，平均合格不意味着每个病例合格。5.29%越界很小，不是方向失败；也不能直接标成‘质量不变’。</p>",
        "<img src='development_checkpoints.png' alt='Same sixteen development cases at checkpoints72 and136'/>",
        "<h2>3. 预算变化主要说明了什么</h2><p>真正关键的量是屏蔽动作前的概率分布。动作通过随机抽样产生，随后受到剩余额度可行性限制；采集数量波动不等于网络知道何时多采。CPU重放已与真实缓存的合法动作最大概率与熵通过2e-4容差检查。</p>",
        table(
            [
                "模型",
                "未约束平均活动预算",
                "帧内期望预算SD，病例等权",
                "实际活动预算SD，病例等权",
                "全部暖帧配额强制",
                "最后1/4暖帧配额强制",
            ],
            [
                [
                    arm,
                    f"{analysis['models'][arm]['checkpoints']['136']['expected_unmasked']:.3f}",
                    f"{variability[arm]['unmasked_expectation_sd']:.4f}",
                    f"{variability[arm]['chosen_stage_sd']:.3f}",
                    f"{analysis['models'][arm]['checkpoints']['136']['forced_fraction']:.2%}",
                    f"{next(x for x in quarter if x['arm'] == arm and x['quarter'] == 4)['forced_fraction']:.2%}",
                ]
                for arm in ARMS
            ],
        ),
        "<p>三个网络的期望预算在视频内部只变化约0.01—0.02条，而实际动作SD约3—5条。未约束分布仍然有较高熵，说明动作的宽幅变化很大程度上可由随机采样和配额解释。不能断言完全没有状态依赖，但当前依赖很弱，不能把明显的锯齿预算图直接当作任务适应性的证据。</p><p>三个模型在各自已保存的2795个开发暖帧状态上，未约束概率最大的动作全部是7条。帧内概率与该视频平均分布的JS差异仅为T1的1.45×10⁻⁵、Q2的5.55×10⁻⁶、T2的5.07×10⁻⁶，进一步支持完整分布对状态的响应弱；不仅是不同分布恰好具有相同预算均值。这不等于已经运行了argmax闭环；换动作后历史也会变化。</p>",
        "<h3>你担心的尾段限制确实发生了</h3><p>T1在最后1/4暖帧，97.12%的帧只有一个可行动作；实际第一阶段平均4.08条，网络未约束期望仍约8.20条。T2最后1/4有53.42%帧被迫取唯一动作，实际第二阶段均值10.47条，网络期望仍约5.88条。这分别体现了前面消耗偏多、后面被压低，以及前面偏少、后面被迫补采的情况；不是可以仅靠看平均14条就忽略的小细节。</p>",
        "<img src='budget_quarters.png' alt='Case-equal means of actual and unmasked expected budget across four warm-frame quarters, and forced-action percentages'/><p>图为各病例先分四段、再对16病例等权平均；不包含首帧10+4，不含置信区间。这里按时间位置分段，不是心动周期分期。</p>",
        "<h3>一个已经能定位的原因：初始化偏好7，不等于分布均值7</h3><p>现有代码把7档logit设为0.5，其他档设为−0.5，随后仍对所有档位随机抽样。仅计算该偏置，T1的平均数约8.20，第二阶段约5.81，都与配额要求的7不同；第1次真实检查点也分别为8.207与5.824。136后仍未充分校正这个差距，所以即使网络几乎不读取图像状态，配额也会自动制造前多后少或前少后多。初始化分布与额度均值不匹配、最终尾段强制是已核实事实；它对EF变化贡献多少仍未通过独立闭环对照量化。</p>",
        "<img src='sample_1_budget.png' alt='First preselected case: actual stage budgets and raw policy preference, quota-forced frames marked'/>",
        "<h2>4. 是否训练根本没有生效</h2><p>不是所有参数都冻结或梯度断了：全部408次更新的梯度有限且大于零，参数核对显示输入、循环、输出权重均有变化，三模型隐状态的|h|&gt;0.95占比均为0。仅从这些证据看，不支持‘模型没更新’或‘GRU整体饱和’的解释。但是能更新与学到有用状态分配是两个不同要求。</p>",
        table(
            [
                "模型",
                "训练梯度中位数",
                "仅保留最近帧：期望预算平均绝对变化",
                "屏蔽资源/时钟：平均绝对变化",
                "仅保留资源/时钟：平均绝对变化",
            ],
            [
                [
                    arm,
                    f"{training[arm]['median_gradient']:.4f}",
                    f"{analysis['models'][arm]['latest_only']['expected_budget_abs_change']:.4f}",
                    f"{analysis['models'][arm]['without_resource_clock']['expected_budget_abs_change']:.4f}",
                    f"{analysis['models'][arm]['resource_clock_only']['expected_budget_abs_change']:.4f}",
                ]
                for arm in ARMS
            ],
        ),
        "<p>这些变化全部小于0.11条，支持目前网络对历史和状态的输出依赖弱。屏蔽输入可能离开训练分布，这里只重放控制器概率，没有重新执行采集、EF读出；不能把它称作移除历史后EF不变的消融。</p>",
        "<p>训练使用全视频一个奖励的REINFORCE，每一帧共享同一个优势信号，且配额仅剩一个动作时该帧概率恒为1，直接的动作梯度为零。对哪个时间点值得多采的信用分配较粗；这是代码层面的训练特征，可能使状态条件学习困难，但当前结果不足以把效果弱全归因于该因素、学习率或训练量。输入压缩、输出初始化尺度、动作档位、更新数量也是下一版可检查的因素。</p>",
        "<img src='training_curves.png' alt='Raw per-video training points and trailing16-update medians'/><p>训练原始点来自不同病例，16更新中位数不是严格收敛曲线。对同64训练病例的首末访问，T1平均EF误差变化−0.171、T2−0.320个百分点；病例访问使用不同噪声，不能据此单独证明学习收敛或过拟合。Q2的对偶变量从[1,1]增至[1.391,1.148]，表明约束反馈确实参与更新，不是完全没有质量惩罚。</p>",
        "<h2>5. 重建样例、表征和EF的解释</h2><p>样例固定开发名单前三例，各选四个时间块中心，没有按结果挑图。输出仍是极坐标图像域B-mode近似，并非真实RF或设备采集。所有原图/重建共享[-1,1]显示范围，误差和双粒子方差使用共同色域。完整图册还包括观测线、未观测区域误差及方差，缺失清单为空。</p>",
        "<img src='sample_1_reconstruction.png' alt='First locked example, four temporal positions, target and fixed/learned reconstructions'/>",
        "<p>各控制器PCA前两轴解释约72.6%(T1)、59.4%(Q2)、63.6%(T2)的隐状态方差。PCA能显示表征变化，却不能证明读出了心动周期；隐状态变化也未必传到预算输出。当前主证据是输出响应弱，而不是PCA图有没有漂亮的分组。</p>",
        "<img src='representation_pca.png' alt='Separate case-equal hidden-state PCA for each model'/>",
        "<p>固定7+0→7+14时，MSE从0.02480降至0.01070、PSNR从22.40升至26.03dB，而EF MAE从5.287变为4.823个百分点。EF在这个工作范围内的平均变化小于图像指标的变化，但不是完全不受预算影响。完整原视频的冻结EF模型在同16例MAE为5.199，部分重建读出的误差还略低于它；标签误差与模型对重建域的反应不是单调画质函数，不能简单用‘原图EF最好’推断重建或调度价值。</p>",
        "<h2>6. 资源收益与证据边界</h2><p>7+7保存的逐帧墙钟时间合计，病例平均79.67秒；136时T1为80.53、Q2为78.53、T2为78.99秒。Q2减少9.02%线数，记录时间仅减少约1.44%，线数节省不等于同幅度GPU提速。第一阶段固定存在，第二阶段在非零档仍运行扩散步骤，减少几个观测像素不直接按比例减少网络调用。这里的计时不包括视频EF读出及整个任务I/O，没有对运行时噪声作独立重复测速，不能把约1秒差异宣布为稳定工程加速。</p>",
        "<p>没有32例盲确认、独立训练种子或136同预算组成周期/随机对照。T1/T2同14条均值只控制总资源，未单独剥离预算组成、随机动作、配额尾段与采集时机的贡献。72次较好点原样保留，不自动替换成部署baseline。下附病例配对bootstrap只描述当前16例的不确定性，不用区间跨零阻止继续探索；它也不覆盖独立训练的随机性。</p>",
        table(
            ["136模型", "EF相对7+7均值", "病例bootstrap95%区间", "EF更好病例数"],
            [
                [
                    arm,
                    f"{performance[f'{arm}_136']['ef_vs_fixed']:+.4f}",
                    f"[{performance[f'{arm}_136']['ef_delta_ci'][0]:+.4f}, {performance[f'{arm}_136']['ef_delta_ci'][1]:+.4f}]",
                    f"{performance[f'{arm}_136']['ef_better_cases']}/16",
                ]
                for arm in ARMS
            ],
        ),
        "<h2>7. 对下一版的建议（本轮未执行）</h2><p>先解决已定位的‘未约束分布均值与配额不匹配＋尾段强制＋状态输出弱’，再决定是否长训或扩数据。下一版可以让初始分布的期望资源匹配目标，在保留公平资源比较的同时降低尾段强制；明确按当前观测学习的概率相对状态无关分布是否有差异。训练改进应优先让奖励更能区分不同时间点的补采价值，配合少量同预算组成的状态无关/周期对照，而不是仅增加大批病例看平均EF能否偶然降低。</p>",
        "<p>继续保留T1与T2的EF任务、Q2的单帧省预算问题分开研究。Q2需要检查难帧/病例的质量约束，不把EF作它的成功条件；T1/T2不要求单帧MSE同时改善。不能由本轮否定动态预算或EF，也不能说只要多训练就能解决。现有136和72权重、轨迹、统计与诊断已足够为新版本提供参照，本轮按约定结束，不加跑264或新的搜索。</p>",
        "<h2>8. 下载与复核</h2><p>本地结果目录：<code>G:\\科研项目\\毕设\\results\\task_budget_stages_v1_download</code>。收尾包1,360,381字节（539文件）；离线图册包14,080,194字节（60文件）；另保存控制器/Adam检查点包764,796字节（1/16/72/136×三模型，40文件）。没有重复下载CASL/EF预训练权重或完整视频缓存。</p>",
        "<p>本报告新增的分析数据.json、病例对照.csv、时间四分段预算.csv与budget_quarters图保存病例等权聚合和可复核数据。原下载报告与文件没有改写。完整六项图册在下载目录offline/task_budget_stages_v1.offline136_v1/REPORT_中文.html；复现脚本为cognitive-ultrasound/scripts/report_task_budget_stages_136_download.py。</p>",
    ]
    document = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>动态预算136实验分析</title><style>body{max-width:1100px;margin:30px auto;padding:0 22px;font:16px/1.8 system-ui,sans-serif;color:#20252b}h1{font-size:28px}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cdd5dd;padding:8px;text-align:left}th{background:#edf3f8}img{max-width:100%;height:auto}.lead{background:#edf4fa;padding:18px;border-left:5px solid #0072b2}code{overflow-wrap:anywhere}</style>"
        + "\n".join(parts)
        + "</html>"
    )
    (out / "实验分析.html").write_text(document, encoding="utf-8")
    (out / "analysis_provenance.json").write_text(
        json.dumps(
            dict(
                source_hashes=evidence,
                script_sha256=sha(__file__),
                aggregation="case-equal; warm-frame quartiles; no new model/task inference",
                uncertainty="10000 case paired bootstrap, seed20261011; descriptive only",
                formats=["HTML", "PNG180dpi", "PDF"],
                target="general research report, no publisher specified",
                numpy=np.__version__,
                matplotlib=matplotlib.__version__,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(report=str(out / "实验分析.html"), status="completed", cases=len(names)),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.data, a.output)
