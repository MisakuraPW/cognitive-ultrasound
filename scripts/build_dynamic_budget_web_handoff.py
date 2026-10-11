"""Curated local research handoff, plus72/136 audit. No server/model inference."""

import argparse
import csv
import hashlib
import html
import json
import shutil
import zipfile
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


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def quota_flags(actions, levels, enabled):
    """Reconstruct feasibility from recorded actions, never rerun the perception model."""
    if not enabled:
        return np.zeros(len(actions), bool)
    reachable = [1]
    for _ in actions:
        reachable.append(sum_bitsets(reachable[-1], levels))
    remaining = 7 * len(actions)
    flags = []
    for i, chosen in enumerate(actions):
        suffix = reachable[len(actions) - i - 1]
        legal = [k for k in levels if remaining >= k and (suffix >> (remaining - k)) & 1]
        assert chosen in legal, (i, chosen, legal)
        flags.append(len(legal) == 1)
        remaining -= int(chosen)
    assert remaining == 0
    return np.array(flags)


def sum_bitsets(bits, levels):
    value = 0
    for k in levels:
        value |= bits << k
    return value


def audit(project, out):
    source = project / "results/task_budget_stages_v1_download/closure/task_budget_stages_v1"
    offline = (
        project
        / "results/task_budget_stages_v1_download/offline/task_budget_stages_v1.offline136_v1"
    )
    manifest = read(source / "manifest.json")
    names = manifest["cohorts"]["development"]
    model = read(offline / "analysis.json")["models"]
    training, development, quarters, case_pairs = [], [], [], []
    all_records = {}
    for arm in ARMS:
        records = [
            read(source / "training" / arm / "updates" / f"{i:05d}/record.json")
            for i in range(1, 137)
        ]
        all_records[arm] = records
        before = {r["case"]: r for r in records[:72]}
        after = {r["case"]: r for r in records[72:]}
        assert set(before) == set(after) and len(after) == 64
        metric = "mse" if arm == "Q2" else "ef_error"
        changes = []
        for name in sorted(before):
            change = after[name][metric] - before[name][metric]
            changes.append(change)
            case_pairs.append(
                dict(
                    arm=arm,
                    case=name,
                    metric=metric,
                    earlier_update=before[name]["update"],
                    later_update=after[name]["update"],
                    earlier_error=before[name][metric],
                    later_error=after[name][metric],
                    change=change,
                )
            )
        training.append(
            dict(
                arm=arm,
                metric=metric,
                cases=64,
                before=float(np.mean([r[metric] for r in before.values()])),
                after=float(np.mean([r[metric] for r in after.values()])),
                change=float(np.mean(changes)),
                improved_cases=sum(x < 0 for x in changes),
            )
        )
    budget_by_checkpoint = {}
    for update, phase in [(72, "full_pass_1_evaluate"), (136, "full_pass_2_evaluate")]:
        rows = read(source / "jobs" / phase / "result.json")["cases"]
        for arm in ARMS:
            rs = [r for r in rows if r["arm"] == arm]
            assert len(rs) == 16 and {r["case"] for r in rs} == set(names)
            development.append(
                dict(
                    arm=arm,
                    update=update,
                    ef_error=float(np.mean([r["ef_error"] for r in rs])),
                    forced=float(np.mean([r["quota_forced_fraction"] for r in rs])),
                    constrained=float(np.mean([r["quota_constrained_fraction"] for r in rs])),
                )
            )
            values = []
            for r in rs:
                rel = Path(r["budget_trace"]).relative_to(
                    "/root/autodl-tmp/outputs_casl/task_budget_stages_v1"
                )
                with (source / rel).open(encoding="utf-8-sig") as f:
                    frames = list(csv.DictReader(f))[1:]
                assert len(frames) == manifest["files"][r["case"]]["frames"] - 1
                actions = np.array([int(x["k1" if arm == "T1" else "k2"]) for x in frames])
                flags = quota_flags(
                    actions, [4, 7, 10, 14] if arm == "T1" else [0, 2, 4, 7, 14], arm != "Q2"
                )
                assert np.isclose(flags.mean(), r["quota_forced_fraction"])
                budget_by_checkpoint[(arm, update, r["case"])] = actions
                t = np.minimum(3, (4 * np.arange(len(actions)) / len(actions)).astype(int))
                values.append(
                    [[float(actions[t == q].mean()), float(flags[t == q].mean())] for q in range(4)]
                )
            avg = np.mean(values, axis=0)
            for q in range(4):
                quarters.append(
                    dict(
                        arm=arm,
                        update=update,
                        quarter=q + 1,
                        active_budget=float(avg[q, 0]),
                        forced_fraction=float(avg[q, 1]),
                    )
                )
    differences = []
    for arm in ARMS:
        changes = [
            float(
                np.mean(
                    budget_by_checkpoint[(arm, 72, name)] != budget_by_checkpoint[(arm, 136, name)]
                )
            )
            for name in names
        ]
        differences.append(dict(arm=arm, different_action_fraction=float(np.mean(changes))))
    expected = [
        {"arm": arm, "update": update, **model[arm]["checkpoints"][str(update)]}
        for arm in ARMS
        for update in [1, 16, 72, 136]
    ]
    result = dict(
        training_latest_before72_vs_next_pass=training,
        development=development,
        quarters=quarters,
        differing_actions=differences,
        common136_state_replay=expected,
        scope="offline audit; training visits are not fixed-checkpoint train-set evaluation",
        loss_definition="recorded objective/reward are episode task/cost targets; actual REINFORCE surrogate loss per update was not logged",
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "核对数据.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, rows in [
        ("同病例训练访问.csv", case_pairs),
        ("开发72与136.csv", development),
        ("两检查点四分段预算.csv", quarters),
        ("共同输入权重重放.csv", expected),
    ]:
        write_csv(out / name, rows)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axs = plt.subplots(2, 3, figsize=(12, 6), layout="constrained", sharex=True)
        for col, arm in enumerate(ARMS):
            for update, marker, style in [(72, "o", "--"), (136, "s", "-")]:
                rs = [r for r in quarters if r["arm"] == arm and r["update"] == update]
                axs[0, col].plot(
                    [r["quarter"] for r in rs],
                    [r["active_budget"] for r in rs],
                    marker=marker,
                    ls=style,
                    color=COLORS[arm] if update == 136 else "#333333",
                    label=f"update{update}",
                )
                axs[1, col].plot(
                    [r["quarter"] for r in rs],
                    [100 * r["forced_fraction"] for r in rs],
                    marker=marker,
                    ls=style,
                    color=COLORS[arm] if update == 136 else "#333333",
                )
            axs[0, col].set(title=arm, ylabel="Active-stage lines", ylim=(0, 14))
            axs[0, col].legend()
            axs[1, col].set(
                ylabel="Quota-forced frames (%)",
                xlabel="Warm-video quarter",
                ylim=(0, 100),
                xticks=[1, 2, 3, 4],
            )
            for ax in axs[:, col]:
                ax.grid(alpha=0.2)
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"budget72_vs136.{ext}", dpi=180, facecolor="white")
        plt.close(fig)
    text = """72到136次更新的解释核对

用户的两个假说：过拟合；或训练后前面分配更大胆，使后面受到更严重的配额限制。

1. 过拟合是合理候选解释，但还没有确诊。
对64个相同训练病例，比较累计72之前最后一次训练访问与73—136的新一遍访问：T1平均EF误差2.9347→2.7573，36/64改善；T2为3.1844→2.8938，41/64改善。对应固定16开发病例，T1为4.8141→5.1276，T2为4.9744→5.1356。因此出现了“训练更好、开发更差”的现象。
但这些训练记录由各次不断变化的权重产生，不是把72和136的固定检查点各自评价同一训练集；不同访问也用不同采样/扩散噪声。单次训练、单个开发种子以及较弱的状态条件输出，都不允许仅据这两点宣布过拟合。训练噪声、策略漂移和闭环敏感性仍是候选原因。小数据置信区间只描述不确定性，不作为停止探索的理由。

2. “配额尾段强制”确实存在；“136比72明显更严重”不符合现有数据。
T1全暖帧强制率72次27.590%、136次27.513%；T2分别13.203%、13.223%。前两段T1预算72次8.024/8.343，136次8.015/8.335；T2分别5.666/6.041与5.666/6.046。没有出现明显的前半段分配更激进或总体配额更紧。
新增budget72_vs136.png给出各视频四分段再按病例等权聚合的实际预算和强制率。每帧的强制标记由保存动作和原可行性规则重建，聚合值已与原记录逐病例核对，不需要重跑CASL。

3. 必须区分最大概率动作、随机样本与均值。
初始化并没有规定每帧只取7；主训练/评价都是随机采样。7档logit=0.5，其他=-0.5，所以7是最可能的单个动作，但仅偏置的期望值T1约8.20、第二阶段约5.81，不等于额度均值7。第1次权重在共同136输入上已有类似均值，不是学习之后才产生所有数量变化。
72与136在同一136状态输入上的未约束期望：T1约8.173→8.184，T2约5.823→5.867。三个136模型在各自2795个保存开发状态上未约束最大概率动作均为7，状态内期望预算SD仅0.01—0.02条；完整概率分布的帧内JS差异也很小。动态数量是真的，但有用的状态条件分配尚未确立。缓存重放不是argmax闭环评价，也不能证明后者必定恒定。

4. Loss图应该怎样看。
图册已含training_curves.png（训练EF/MSE、线数和梯度）、episode_objectives.png（实际整视频目标）与reward_and_constraints.png（奖励、优势、Q2对偶变量）。本次算法是REINFORCE，不是普通监督分类，不能把EF误差、目标函数和策略优化器loss混为一个量。
T1/T2记录的objective=重建视频EF绝对误差−完整视频EF绝对误差，reward=−objective；Q2为归一化线数成本加平均MSE/p90约束的拉格朗日目标，约束权重随训练改变。真正求导的策略代理loss为−advantage×sum(log p(action|state,legal))/128，本轮没有逐更新保存该标量，不能编造一条代理loss曲线。奖励/目标图能观察训练，但不是固定分布上的严格收敛证明。

结论：你的过拟合解释值得保留；两大设计限制也值得改进。现有证据更直接支持“严重的额度均值不匹配及尾段强制早已存在，状态条件输出还弱”，不支持把72→136的退步直接归因为分配能力更强使后半段限制加重。这次只核对与打包，没有修改模型或再跑训练。
"""
    (out / "72到136_假说核对.txt").write_text(text, encoding="utf-8")
    (out / "72到136_假说核对.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>72到136假说核对</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.8 sans-serif;padding:20px}pre{white-space:pre-wrap}img{max-width:100%}</style><pre>"
        + html.escape(text)
        + "</pre><img src='budget72_vs136.png' alt='Actual budgets and reconstructed quota constraints at72 and136'>",
        encoding="utf-8",
    )
    return result


def build(project, destination, resume_partial=False):
    project, destination = Path(project).resolve(), Path(destination).resolve()
    if destination.exists() and (not resume_partial or (destination / "文件清单.json").exists()):
        raise FileExistsError("Use a fresh handoff directory; preserve prior bundles")
    destination.mkdir(parents=True, exist_ok=resume_partial)
    mapping = []

    def copy(src, target):
        src, target = Path(src), destination / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        mapping.append(
            dict(source=str(src), destination=str(target.relative_to(destination)), sha256=sha(src))
        )

    old = project / "整理/动态预算探索交接_20261010"
    for p in old.rglob("*"):
        if p.is_file() and p.suffix not in [".zip", ".npz", ".pdf"]:
            copy(p, Path("01_历史E0E1E2_截至20261010") / p.relative_to(old))
    repair = (
        project
        / "cognitive-ultrasound/results/task_budget_ef_repair_v4_download/task_budget_ef_repair_v4"
    )
    for name in [
        "summary.json",
        "config.json",
        "manifest.json",
        "identity.json",
        "status.json",
        "plan.json",
        "repair_lineage.json",
        "ef_resource_confirmation.png",
        "ef_resource_development.png",
    ]:
        copy(repair / name, Path("01_历史E0E1E2_截至20261010/补充原始指标") / name)
    value = project / "results/task_budget_value_v1_download/analysis/task_budget_value_v1"
    for p in value.rglob("*"):
        if p.is_file() and p.suffix in [".json", ".csv", ".md", ".png"]:
            copy(p, Path("02_调度空间与补采诊断P0P5") / p.relative_to(value))
    data = project / "results/task_budget_stages_v1_download"
    closure = data / "closure/task_budget_stages_v1"
    for name in [
        "SCOPE_136_REPORT.md",
        "scope_closure_136.json",
        "PILOT_REPORT.md",
        "config.json",
        "manifest.json",
        "identity.json",
        "pilot_gate.json",
        "evaluation_cases.csv",
    ]:
        copy(closure / name, Path("03_T1Q2T2_136正式范围") / name)
    for phase in [
        "pilot_evaluate",
        "full_baselines",
        "full_pass_1_evaluate",
        "full_pass_2_evaluate",
    ]:
        copy(
            closure / "jobs" / phase / "result.json",
            Path("03_T1Q2T2_136正式范围/阶段原始结果") / (phase + ".json"),
        )
    offline = data / "offline/task_budget_stages_v1.offline136_v1"
    for p in offline.iterdir():
        if p.is_file() and p.suffix in [".json", ".csv", ".png", ".html"]:
            copy(p, Path("04_136六项离线图册") / p.name)
    report = project / "reports/task_budget_stages_136_20261011"
    for p in report.iterdir():
        if p.is_file() and p.suffix in [".html", ".json", ".csv", ".png"]:
            copy(p, Path("05_136综合分析") / p.name)
    audit(project, destination / "06_72到136假说核对")
    repo = project / "cognitive-ultrasound"
    for name in ["task_budget_ef_repair.yaml", "task_budget_value.yaml", "task_budget_stages.yaml"]:
        copy(repo / "configs" / name, Path("07_协议与关键代码/配置") / name)
    for name in [
        "task_budget_ef.md",
        "task_budget_value_diagnostics_plan_20261010.md",
        "task_budget_value_validation.md",
        "task_budget_stages.md",
        "task_budget_stages_finish136.md",
        "task_budget_stages_offline136.md",
        "task_budget_stages_136_results.md",
    ]:
        copy(repo / "docs" / name, Path("07_协议与关键代码/协议原文") / name)
    for name in [
        "stage_policy.py",
        "stage_protocol.py",
        "stage_workers.py",
        "episode.py",
        "value_runtime.py",
    ]:
        copy(
            repo / "src/cognitive_ultrasound/task_budget" / name,
            Path("07_协议与关键代码/当前阶段实现") / name,
        )
    for name in [
        "download_verification.json",
        "015.remote_receipt.json",
        "016.remote_receipt.json",
        "018.remote_receipt.json",
    ]:
        copy(data / name, Path("08_来源校验") / name)
    copy(Path(__file__), "07_协议与关键代码/本包生成脚本.py")
    introduction = """动态预算实验交接 截至2026年10月12日

用途：把本文件夹同名ZIP上传给网页端GPT，理解已经完成的实验、证据与当前疑问；不是服务器代码部署包。

推荐阅读：00_先读本文件.txt → 06_72到136假说核对/72到136_假说核对.txt → 05_136综合分析/实验分析.html → 04_136六项离线图册/REPORT_中文.html → 02_调度空间与补采诊断P0P5/REPORT_中文.md → 01_历史E0E1E2_截至20261010/02_实验报告/中文实验报告.md。需要复核数字时读取对应CSV/JSON和07协议代码。

目前研究问题有两个作用，分开探索：A 不太影响单帧重建/分割指标时尽量少采；B 根据时序任务EF在不同时间点的补采价值分配预算。第一阶段只看到历史，第二阶段还能看到当前部分观测。当前已实现的是T1第一阶段EF、Q2第二阶段重建省预算、T2第二阶段EF；没有第一阶段单帧省预算、分割训练、联合多任务、planning或WM。

实验脉络：
1. 早期E0/E1/E2存在学习/梯度异常，旧91任务没有全部完成。修复v4真正完成18任务，32训练/8开发/16确认，E1/E2各200次更新；E1是local_projection_v2近似梯度，不是对完整CASL精确求导。最终第二阶段动作恒定，不能证明有效的两阶段动态分配。01中原设计和历史建议不是当前执行状态。
2. P0—P5诊断完成8例，冻结模型、没有预算训练；区分固定预算响应、缓存窗口敏感性、同状态真实+7线补采后缀、成本匹配时间安排、可观测状态预测。工程筛查False或小样本区间跨零不自动否定EF探索。缓存事后挑选安排不是已经可部署的在线策略。
3. T1/Q2/T2：64训练、16开发，3个因果GRU各136整视频更新，25步FP32、两粒子、CASL/EF及TBIG式选线冻结。主训练/评价都是随机动作；T1/T2每段视频精确平均14条，活动阶段额度平均7条。Q2不受总额度限制，以重建约束省线。72与136用同16开发病例、评价种子42。32个新确认病例虽锁定，但取消了确认计算；没有264训练或136成本组成匹配周期对照。

最新结果，EF单位为百分点：
固定7+7 EF MAE5.1434。T1：72为4.8141、136为5.1276，均14条；T2：72为4.9744、136为5.1356，均14条。Q2：72平均12.669条、MSE相对+4.93%；136平均12.737条（省9.02%）、MSE+5.29%、p90+6.68%、EF5.2066。单病例目标与病例平均目标分开看，136只有5/16病例同时满足Q2两个重建目标。

机制诊断：三个136模型在各自2795暖帧的未约束概率最大动作都是7，期望活动预算随帧的SD约0.01—0.02条，实际随机动作SD约3—5条，状态条件响应弱。T1/T2末1/4暖帧配额强制约97.12%/53.42%。初始化“7档概率最高”不等于“概率期望7条”：T1偏置期望约8.20，第二阶段约5.81。不能把锯齿数量图自动当作学习到心动周期或补采价值，也不能把全部差异归因为某一个因素。

用户现在的两个假说：72有提升、136下降，是否过拟合？或训练更好后前面更大胆，后面受配额限制更严重？新增06给出逐病例、同输入权重与时间四分段核对。训练访问误差下降/开发误差上升支持过拟合作为候选，但不是确诊；72与136的预算时序及強制比例几乎没变，不支持后一假说目前的具体描述。尾段限制的确真实存在，已在72发生。

Loss和图表：04包括training_curves、episode_objectives、reward_and_constraints、共同输入控制器重放、PCA/敏感性、三病例重建/误差/观测线/双粒子方差/预算图。训练记录是实际整视频奖励目标，不是普通监督loss。真正REINFORCE代理loss未逐步保存，不能补画并冒充已测曲线。PCA和缓存输入扰动是诊断，不是临床心动周期理解或真实闭环消融。

阅读边界：不同轮的数据、读出覆盖、冷启动与控制器不同，不能把跨轮EF均值直接拼成排名。旧文档的未执行事项已被后续实验推进，目录按日期标明；截至当前状态以本文件、03收尾和06为准。报告中的病例级置信区间用于表达不确定性，不作为继续探索的硬门槛。不要求本轮复现CASL全量训练或正式论文认证。

包内是可读报告、PNG图、CSV/JSON、原始设计和关键代码；不含原始视频、完整帧缓存、预训练/预算权重、优化器、嵌套tar/zip、SSH或认证资料。检查点已单独保存在本地results/task_budget_stages_v1_download/checkpoints，不需让网页GPT读取二进制权重。包内文件清单.json可核验来源、SHA256；目录科学内容不按结果挑删病例。
"""
    (destination / "00_先读本文件.txt").write_text(introduction, encoding="utf-8")
    prompt = """请先阅读00_先读本文件.txt及06的72到136假说核对，再阅读最新综合分析和原始CSV/JSON。
我的目标是在冻结CASL/EF底座上研究两阶段动态预算：单帧任务省资源与时序任务资源分配分开探索。请理解已经执行的版本，不把历史计划当作已完成实验，不把更新数当epoch，不把随机动作/配额强制当作网络已经学会时序分配。
我目前怀疑72→136的EF退步可能是过拟合，也可能是流式预算的前后约束和训练相互冲突。请逐条检查哪些数据支持、反对或不能区分这些解释；区分训练整视频目标/奖励与未保存的策略代理loss；不要因为小样本CI跨零就否定继续探索。之后讨论应该优先修正初始化/预算约束、输入和模型、信用分配还是训练量，以及如何以最少新计算区分候选原因。先分析，不泛化成新的创新点池或直接要求全量复现。
"""
    (destination / "00_网页对话提示词.txt").write_text(prompt, encoding="utf-8")
    inventory = {
        str(p.relative_to(destination)): dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in destination.rglob("*")
        if p.is_file()
    }
    (destination / "文件清单.json").write_text(
        json.dumps(
            dict(
                files=inventory,
                copied_sources=mapping,
                scope="local curated research materials; no live inference",
                created="2026-10-12",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = destination.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    files = [p for p in destination.rglob("*") if p.is_file()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, str(Path(destination.name) / p.relative_to(destination)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.infolist()) == len(files)
        for info in z.infolist():
            rel = Path(info.filename).relative_to(destination.name)
            assert hashlib.sha256(z.read(info)).hexdigest() == sha(destination / rel)
    archive.with_suffix(".zip.sha256").write_text(
        sha(archive) + "  " + archive.name + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                folder=str(destination),
                archive=str(archive),
                files=len(files),
                bytes=archive.stat().st_size,
                sha256=sha(archive),
                verified=True,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume-partial", action="store_true")
    a = p.parse_args()
    build(a.project, a.output, a.resume_partial)
