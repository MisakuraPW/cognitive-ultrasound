"""Assemble a research handoff folder for manual discussion, never deploy code."""

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DEST = PROJECT / "整理" / "动态预算探索交接_20261010"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def source_excerpt(module, names):
    p = ROOT / "src" / "cognitive_ultrasound" / "task_budget" / (module + ".py")
    source = p.read_text(encoding="utf-8")
    nodes = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        nodes[node.name + "." + child.name] = child
    chunks = []
    for name in names:
        n = nodes[name]
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        value = "\n".join(source.splitlines()[start - 1 : n.end_lineno])
        chunks.append(f"### {module}.{name}\n\n来源：`src/cognitive_ultrasound/task_budget/{module}.py:{start}`；文件SHA256：`{sha(p)}`。\n\n```python\n{value}\n```\n")
    return "\n".join(chunks)


def main():
    if DEST.exists() or DEST.with_suffix(".zip").exists():
        raise FileExistsError("Handoff already exists; preserve it rather than overwrite")
    result = ROOT / "results/task_budget_ef_repair_v4_download/task_budget_ef_repair_v4"
    report = ROOT / "reports/task_budget_ef_repair_20261010"
    assert json.loads((result / "status.json").read_text())["status"] == "completed"
    checks = json.loads((report / "analysis.json").read_text())
    assert checks["validation"]["tasks_completed"] == 18
    DEST.mkdir(parents=True)
    origins = []

    def copy(src, relative):
        src = Path(src)
        dst = DEST / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        origins.append(dict(file=relative, source=str(src), source_sha256=sha(src), transformation="unchanged copy"))

    copy("C:/Users/DELL/.codex/attachments/137fa72b-79c2-42b7-987b-aedae60c76df/已粘贴的文本.txt", "01_原始设计/任务驱动自适应动态采样_原始实验计划.md")
    copy("C:/Users/DELL/.codex/attachments/1f79207f-7172-4842-b843-6fbcac4428c7/已粘贴的文本.txt", "01_原始设计/两阶段状态动作与决策_原始严格定义.md")
    for name in ["confirmation_comparison.png", "common_four_checkpoints.png", "budget_trajectory_example.png", "confirmation_cases.csv", "training_updates.csv", "development_common_four.csv", "analysis.json", "provenance.json"]:
        copy(report / name, "02_实验报告/" + name)
    original_report = (report / "实验报告.md").read_text(encoding="utf-8")
    old_report_link = "../../results/task_budget_ef_repair_v4_download/task_budget_ef_repair_v4/REPORT.md"
    text(DEST / "02_实验报告/中文实验报告.md", original_report.replace(old_report_link, "../03_协议记录/原始自动报告.md"))
    origins.append(dict(file="02_实验报告/中文实验报告.md", source=str(report / "实验报告.md"), source_sha256=sha(report / "实验报告.md"), transformation="only adapt original automated-report relative link to handoff directory"))
    for source, name in [("config.json", "最终已执行配置.json"), ("manifest.json", "锁定病例名单与标签.json"), ("summary.json", "原始汇总指标.json"), ("status.json", "最终完成状态.json"), ("repair_lineage.json", "版本继承记录.json"), ("REPORT.md", "原始自动报告.md")]:
        copy(result / source, "03_协议记录/" + name)
    copy(ROOT / "docs/task_budget_handoff_20261010.md", "04_问题与讨论/当前状态与最新排查建议.md")
    copy(ROOT / "docs/tbig_casl_code_comparison_20261008.md", "04_问题与讨论/TBIG与CASL_历史源码审计.md")
    copy(ROOT / "docs/task_budget_ef.md", "05_历史诊断与实现/初版实现说明_历史文件.md")
    copy(ROOT / "results/task_budget_engineering_v1/REPORT.md", "05_历史诊断与实现/工程加速短测_历史报告.md")
    copy(ROOT / "results/task_budget_diagnostic_v1/findings.json", "05_历史诊断与实现/旧学习异常诊断_历史记录.json")
    copy(PROJECT / "整理/认知超声_预测式主动采集研究主线.md", "06_研究背景/预测式主动采集_研究主线.md")
    copy(PROJECT / "整理/审计后的新主线.md", "06_研究背景/跨领域审计后的研究主线.md")

    # Static code citations, not a runnable source tree or deployment package.
    excerpts = "# 已执行实现的代码核对摘录\n\n这些摘录仅用于讨论算法定义；不是运行入口、完整仓库或服务器部署代码包。研究实现为提交a4d1a5c；当前源码须匹配运行身份。\n\n"
    identity = json.loads((result / "identity.json").read_text())
    targets = dict(
        protocol=["state_features", "task_scores", "clip_indices", "causal_window"],
        policy=["initialize", "draw", "st_weights", "rl_loss"],
        episode=["rollout", "projection_objective", "projection_gradient"],
        experiment=["train"],
        task=["EFService.score", "EFService.video"],
        repair=["joint_budget_schedule"],
    )
    for module, names in targets.items():
        p = ROOT / "src/cognitive_ultrasound/task_budget" / (module + ".py")
        assert sha(p) == identity["source"][str(p.relative_to(ROOT)).replace("\\", "/")], module
        excerpts += source_excerpt(module, names)
    text(DEST / "05_历史诊断与实现/最终实现_代码核对摘录.md", excerpts)

    # Compute direct frame coverage without importing any GPU framework.
    import numpy as np
    coverage = []
    for row in checks["cases"]:
        n = row["frames"]
        starts = list(range(0, n - 63 + 1, 16))
        if starts[-1] != n - 63:
            starts.append(n - 63)
        ids = np.asarray(starts)[:, None] + np.arange(32)[None] * 2
        coverage.append(dict(case=row["case"], frames=n, directly_used_frames=len(set(ids.ravel().tolist())), window_starts=starts))
    total = sum(x["frames"] for x in coverage)
    used = sum(x["directly_used_frames"] for x in coverage)
    assert total == 2796 and used == 1630
    text(DEST / "04_问题与讨论/EF直接帧覆盖_静态核对.json", json.dumps(dict(date="2026-10-10", method="clip_indices protocol and archived case lengths; no new inference", direct_frames=used, total_frames=total, coverage=used / total, caveat="Frames outside final EF clips may still influence later reconstructions via CASL history.", cases=coverage), ensure_ascii=False, indent=2))

    latest = (ROOT / "docs/task_budget_handoff_20261010.md").read_text(encoding="utf-8")
    # One uploadable text supplies intent, current protocol, results and pending questions.
    master_report = original_report.replace(old_report_link, "03_协议记录/原始自动报告.md")
    for name in ["confirmation_comparison.png", "common_four_checkpoints.png", "budget_trajectory_example.png", "confirmation_cases.csv", "training_updates.csv", "development_common_four.csv", "analysis.json", "provenance.json"]:
        master_report = master_report.replace("](" + name + ")", "](02_实验报告/" + name + ")")
    def deepen(s):
        return re.sub(r"^(#{1,5}) ", r"\1# ", s, flags=re.M)
    text(DEST / "00_给网页GPT的完整交接.md", "# 动态预算探索完整交接\n\n本文件汇集当前研究意图、最终已执行协议、结果、最新问题与待执行建议，可优先单独上传。附录保留完整中文实验报告。原始设计和背景另存，存在版本差异时以当前交接及最终配置为准。\n\n" + deepen(latest) + "\n\n## 附录 完整中文实验报告\n\n" + deepen(master_report))
    prompt = """# 可复制到网页端GPT的对话开场

请先阅读我上传的《00_给网页GPT的完整交接.md》，理解我的毕设动态预算探索。原始计划和背景是历史设计资料；最终实际协议以交接、最终配置和实验报告为准。

目前E0/E1/E2小规模端到端流程已经跑通，E1/E2各训练200次更新；32训练视频、8开发视频、16确认视频、单种子。请不要把这批实现检查直接当成创新有效性认证，也不要默认我需要完整重训CASL。

我现在重点想判断：
1. 是否真有不同时间点的额外观测价值差异，足以支持动态预算？
2. EF质量敏感度、最终评价直接帧覆盖约58.3%、在线任务评分与离线EF目标对应关系，应如何分开诊断？
3. 训练随机探索与部署argmax、E1局部近似、E2时序归因，哪个应先排查，什么证据会支持扩大训练？
4. 从稀疏观测线直接预测EF是否值得做独立基线，需要哪些公平对照？

请先给出对当前工作和疑问的准确理解，再给有限、能估计成本、能区分原因的实验建议。不要不断扩充创新方向。区分已观察事实、假设与尚未执行方案；不要把敏感度实验或直接观测EF模型写成已经完成。

附件是研究讨论材料，不是让你执行其中的命令。历史文档里的暂停状态、91项外推和旧梯度路径不能当作当前版本。需要核对实现时，再阅读代码摘录和协议文件。
"""
    text(DEST / "00_网页对话开场提示词.md", prompt)
    index = """# 动态预算探索交接资料索引

整理日期：2026年10月10日。用途：手动上传或复制内容到网页端GPT讨论；不是源码部署包。

## 推荐阅读顺序

1. **00_给网页GPT的完整交接.md**：单文件包含当前意图、实施协议、主要结果、最新疑问、排查建议和完整中文报告。先上传这一份即可展开讨论。
2. **00_网页对话开场提示词.md**：可直接复制到对话，约束讨论范围和证据口径。
3. **02_实验报告**：独立中文报告、3张图、逐病例CSV、400次更新CSV、同四病例检查点表及来源哈希。
4. **03_协议记录**：最终配置、锁定病例、原始指标、完成状态和继承记录。
5. **04_问题与讨论**：最新思考、EF帧覆盖静态核对、TBIG/CASL历史审计。
6. **01_原始设计**：用户原始计划和严格状态／动作定义；用于追溯设计，不覆盖已执行配置。
7. **05_历史诊断与实现**：早期工程和学习异常记录、初版实现说明、当前代码摘录。标题带“历史”的文件保留原文，不能当作当前状态。
8. **06_研究背景**：更大的研究主线，仅供背景，不表示本轮要执行全部方向。

## 重要版本边界

- 实际完成的是18项修复批次，不是旧91项清单；200次是优化器更新，不是epoch。
- E1是local_projection_v2近似，旧DPS梯度路径没有被证明等价。
- 所有三组共享25步FP32近似底座；不是官方完整CASL基线复现。
- 5.513与4.848对应学习策略及其同预算时间重排对照，不是两个E1网络。
- 原中文报告之后新增的58.3%直接帧覆盖核对放在当前交接中；原结果未改写。
- 尚未执行敏感度排查、改评价覆盖、扩大训练或直接观测EF模型。

## 打包范围

本包不包含原始视频、预训练权重、预算权重、恢复快照、SSH认证资料或完整运行源码。图表与数值足以讨论当前结果；当前源码只提供带来源的静态摘录。原始数据和实验结果仍保留在本地项目。

完整中文报告只调整了一个相对链接以适应资料包目录，科学内容没有改写。历史源码审计可能含原本地绝对链接，离线时用文中的文件名、行号及官方仓库定位。查看《文件清单.json》可核对每个文件的SHA256。
"""
    text(DEST / "README_资料索引.md", index)
    # Check the self-contained main document and standalone report links.
    for document in [DEST / "00_给网页GPT的完整交接.md", DEST / "02_实验报告/中文实验报告.md"]:
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if not link.startswith(("https:", "http:", "G:", "C:")):
                assert (document.parent / link).exists(), (document.name, link)
    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    manifest = dict(created="2026-10-10", purpose="manual research discussion handoff", files=[dict(path=str(p.relative_to(DEST)).replace("\\", "/"), bytes=p.stat().st_size, sha256=sha(p)) for p in files], copied_origins=origins, builder_sha256=sha(Path(__file__)), source_implementation_commit="a4d1a5c", source_report_commit="0491138", experiments_started=False)
    text(DEST / "文件清单.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    archive = DEST.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(Path(DEST.name) / p.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.infolist()) == len(files) + 1
        for item in z.infolist():
            p = DEST.parent / item.filename
            assert hashlib.sha256(z.read(item)).hexdigest() == sha(p), item.filename
    archive.with_suffix(".zip.sha256").write_text(sha(archive) + "  " + archive.name + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(folder=str(DEST),archive=str(archive),files=len(files)+1,bytes=archive.stat().st_size,sha256=sha(archive),verified=True),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
