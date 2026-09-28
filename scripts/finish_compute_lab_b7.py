"""User-requested scope amendment: finish budget 7, then the existing foundation.

Deploy this Git-versioned launcher to .cache/ of the PINNED running checkout.
It waits for the old coordinator to release its lock; no running source is edited.
Original science/source identity and completed evidence remain untouched.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from cognitive_ultrasound.compute_lab.protocol import profiles
from cognitive_ultrasound.compute_lab.suite import Suite, owned_alive
from cognitive_ultrasound.preparation.common import atomic_json, read_json, source_identity
from cognitive_ultrasound.preparation.suite import run_lock
from cognitive_ultrasound.provenance import sha256


def validate_source_with_capacity_repair(root):
    """Accept only the exact result-dictionary repair, never a general identity waiver."""
    from cognitive_ultrasound.config import ROOT

    old = read_json(root / "identity.json")["source"]
    current = source_identity()
    if current == old:
        return
    file = "src/cognitive_ultrasound/compute_lab/suite.py"
    changed = {p for p in set(old) | set(current) if old.get(p) != current.get(p)}
    if changed != {file}:
        raise RuntimeError("Pinned source changed beyond reviewed capacity result merge")
    before = subprocess.check_output(["git", "show", f"ae83eaf:{file}"], cwd=ROOT)
    if hashlib.sha256(before).hexdigest() != old[file]:
        raise RuntimeError("Original source is not the reviewed frozen coordinator")
    source = b"capacity_results.append(dict(batch=batch, **value))"
    replacement = b"capacity_results.append(dict(value, batch=batch))"
    if before.count(source) != 1:
        raise RuntimeError("Capacity repair anchor is not unique")
    expected = hashlib.sha256(before.replace(source, replacement)).hexdigest()
    if current[file] != expected:
        raise RuntimeError("Coordinator differs beyond the exact one-line metadata repair")
    atomic_json(
        root / "source_repair_capacity_merge.json",
        dict(
            file=file,
            before_sha256=old[file],
            after_sha256=current[file],
            original_identity_untouched=True,
            scope="Result dictionary merge only; inference/training/selection arithmetic unchanged",
        ),
    )


def confirmation_names(selection):
    return ["confirmation_" + name + "_b7" for name in ["official", *selection["selected"]]]


def check_ready(root, selection):
    """No missing/partial evidence is accepted; candidate failures remain failures."""
    values = {}
    for name in confirmation_names(selection):
        file = root / "jobs" / name / "result.json"
        if not file.exists():
            raise RuntimeError(f"Budget 7 is not finished: {name}")
        result = read_json(file)
        if result.get("status") not in ("completed", "failed"):
            raise RuntimeError(f"Nonterminal result: {name}")
        if name == "confirmation_official_b7" and result["status"] != "completed":
            raise RuntimeError("Official reference failed; dependent conclusions are blocked")
        if result["status"] == "completed" and (
            result.get("budget") != 7 or result.get("cohort") != "confirmation"
        ):
            raise RuntimeError(f"Wrong evidence scope: {name}")
        values[name] = sha256(file)
    return values


def wait_for_boundary(root, selection, state_file):
    last = confirmation_names(selection)[-1]
    while True:
        if (root / "STOP").exists():
            raise RuntimeError("User STOP is present; continuation will not clear it")
        state = read_json(root / "status.json")
        coordinator = owned_alive(state.get("pid"), state.get("created"))
        worker = owned_alive(state.get("worker_pid"), state.get("worker_created"))
        atomic_json(state_file, dict(status="waiting_for_budget7", target=last, pid=os.getpid()))
        if not coordinator and not worker:
            if state["status"] not in ("paused_at_boundary", "finished_with_gaps", "completed"):
                raise RuntimeError(f"Coordinator stopped unexpectedly: {state['status']}")
            check_ready(root, selection)
            return
        marker = root / "PAUSE_AFTER_JOB"
        if not marker.exists() or marker.read_text().strip() != last:
            raise RuntimeError(
                "Requested budget-7 boundary changed; refusing implicit continuation"
            )
        time.sleep(10)


def write_summary(root, selection):
    lines = [
        "# 本轮加速探索结论：7 条线确认",
        "",
        "用户将本轮正式确认范围缩减至 7 条线；14/28 条线确认取消，不标为通过或失败。",
        "已有 14 条线开发组用于初筛；不能与 7 条线确认组混合统计，也不能视为 14 条线完整确认。",
        "确认对象为已锁定的32病例、3种子、完整视频。质量门槛及病例级 bootstrap 保持不变。",
        "本次结果用于选择研究工具；后续真正采用其他预算时再检查，不将其设为当前前置任务。",
        "",
        "|版本|类别|质量门槛|严格等价|热帧核心 ms|热帧闭环 ms|完整任务 min|",
        "|---|---|---|---|---:|---:|---:|",
    ]
    records = []
    for name in confirmation_names(selection):
        r = read_json(root / "jobs" / name / "result.json")
        if r["status"] != "completed":
            lines.append(f"|{name}|—|运行失败|未验证|—|—|—|")
            continue
        warm = [x for x in r["rows"] if not x["cold"] and not x["warm_signature_first"]]
        core = sum(x["core_s"] for x in warm) / len(warm)
        closed = sum(x["closed_loop_s"] for x in warm) / len(warm)
        p = r["profile"]
        q = r["quality"]
        passed = q.get("passed", False)
        lines.append(
            f"|{p['name']}|{p['category']}|{'通过' if passed else '不满足/证据不足'}|"
            f"{'通过' if r.get('equivalence_passed') else '未通过'}|{core * 1000:.2f}|"
            f"{closed * 1000:.2f}|{r['process_wall_s'] / 60:.2f}|"
        )
        records.append(
            dict(
                name=p["name"],
                category=p["category"],
                core=core,
                closed=closed,
                quality=passed,
                equivalent=r.get("equivalence_passed", False),
            )
        )
    lines += ["", "## 质量变化与范围", ""]
    for name in confirmation_names(selection):
        r = read_json(root / "jobs" / name / "result.json")
        if r["status"] != "completed":
            continue
        q = r["quality"]
        lines.append(
            f"- {r['profile']['name']}：病例平均损失 {q.get('mean_loss')}；"
            f"单侧95%上界 {q.get('upper95')}；最差病例 {q.get('worst_loss')}。"
        )
    lines += [
        "",
        "三列依次为 PSNR 下降(dB)、SSIM 下降、MAE 相对增加。负数表示相应指标改善。",
        "质量接近不等于逐位/状态等价。时序误差、共同未观测误差、选线和实际线数保留在 matrix.json 与病例逐帧记录中。",
        "core 是同步核心计时；闭环包含传输和状态导出；完整任务包含初始化、读取和输出，三者不可混用。",
        "",
        "## 使用建议",
        "",
    ]
    ref = next(x for x in records if x["name"] == "official")
    for r in records:
        if r["name"] == "official":
            continue
        gain = ref["closed"] / r["closed"]
        eligible = r["quality"] and gain > 1
        statement = "通过本预算质量筛选且闭环更快" if eligible else "未同时显示质量通过与闭环收益"
        if r["category"] == "A" and not r["equivalent"]:
            statement += "；工程修改未通过严格等价，不能作为无损替代"
        lines.append(f"- {r['name']}：闭环倍率 {gain:.3f}×，{statement}。")
    lines += [
        "- 以上为证据与候选建议，不自动采用近似版本。正式 baseline 仍保留 official。",
        "",
        "## Torch 与历史证据",
        "",
        "Torch 定向诊断定位到上采样梯度索引累加和布局转换；固定输入从约203ms改善至75ms，",
        "同输入JAX约78ms。但新原型未通过跨框架状态等价，不能宣传无损或32FPS，未混入这批确认结果。",
        "四个旧Torch低精度短测在compiled/eager内部核对失败；不是已完成质量评估后的否定结论。",
        "详见 torch_root_cause_report.md 和 torch_root_cause_evidence/；诊断微基准不与完整确认计时混算。",
        "",
        "## 其他实验基座",
        "",
        "training_report.json 保存CASL、codec、prior、filter的200步短训及100+100续训对照。",
        "未产生该文件表示训练尚未完成；出现失败/缺失必须保留，不把200步短测等同完整训练。",
        "closure_completion.json 记录原探索收尾，已有完整结果复用；TBIG官方资产缺失仍明确登记。",
        "REPORT.md、excel_facts.csv 为完整证据索引与事实导入，不修改用户预测或核心认知。",
    ]
    (root / "ACCELERATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    remaining = root / "remaining_tasks.md"
    backup = root / "remaining_tasks_before_scope_change.md"
    if remaining.exists() and not backup.exists():
        shutil.copy2(remaining, backup)
    remaining.write_text(
        "# 用户缩减后的剩余任务\n\n"
        "7条线五版本确认 → 加速综合分析 → 已有单帧profiling → 历史收尾 → 四负载短训/续训验证 → 报告及校验包。\n\n"
        "14/28条线确认取消，不再新增候选、病例、种子或优化搜索；原清单仅作历史留档。\n"
        "实际进度看status.json；细项成败看jobs、training_report.json和closure_completion.json。\n",
        encoding="utf-8",
    )


def finish(root, selection, state_file):
    from cognitive_ultrasound.compute_lab.closure import finish as finish_closure
    from cognitive_ultrasound.compute_lab.environment import probe
    from cognitive_ultrasound.compute_lab.report import archive, report
    from cognitive_ultrasound.compute_lab.suite import Stopped

    with run_lock(root):
        state = read_json(root / "status.json")
        if owned_alive(state.get("pid"), state.get("created")) or owned_alive(
            state.get("worker_pid"), state.get("worker_created")
        ):
            raise RuntimeError("Existing coordinator/worker still alive")
        if (root / "STOP").exists():
            raise RuntimeError("STOP retained; no automatic resume")
        identity = read_json(root / "identity.json")
        validate_source_with_capacity_repair(root)
        evidence = check_ready(root, selection)
        cfg = read_json(root / "config.json")
        env = probe(cfg, root / "scope_probe")
        if env["fingerprint"] != identity["environment"] or not env["passed"]:
            raise RuntimeError("Hardware/environment changed or probe failed")
        marker = root / "PAUSE_AFTER_JOB"
        if marker.exists():
            if marker.read_text().strip() != confirmation_names(selection)[-1]:
                raise RuntimeError("Unexpected pause marker; keep it intact")
            marker.unlink()
        plan = dict(
            requested_by="user",
            confirmation_budgets=[7],
            omitted_confirmation_budgets=[14, 28],
            reason="Speed and quality preparation; finish existing b7, do not expand to other budgets",
            selected=selection["selected"],
            evidence=evidence,
            launcher_sha256=sha256(Path(__file__)),
            original_identity_sha256=sha256(root / "identity.json"),
            scientific_thresholds_unchanged=True,
            automatic_adoption=False,
        )
        plan_file = root / "scope_amendment.json"
        if plan_file.exists():
            plan_file = root / "scope_restarts" / f"{os.getpid()}.json"
        atomic_json(plan_file, plan)
        suite = Suite(cfg, root)
        suite.state.update(completed=state.get("completed", []), failed=state.get("failed", []))
        try:
            suite.save(stage="budget7_analysis")
            atomic_json(state_file, dict(status="running_foundation", pid=os.getpid()))
            report(cfg, root)
            write_summary(root, selection)
            # Existing single-frame debug instrumentation only; no b14/b28 cohort runs.
            candidates = profiles(
                selection.get("torch_mode"), selection.get("combined_components", [])
            )
            for name in ["official", *selection["selected"]]:
                p = candidates[name]
                suite.job("profile_" + name, dict(kind="profile", profile=asdict(p)), p.backend)
            finish_closure(suite)
            suite.train()
            suite.save(
                status="finished_with_gaps" if suite.state["failed"] else "completed",
                stage="budget7_and_foundation_finished",
                worker_pid=None,
                worker_created=None,
            )
        except Stopped:
            suite.save(status="stopped")
            raise
        except BaseException as exc:
            suite.save(status="failed", error=str(exc))
            raise
        finally:
            report(cfg, root)
            write_summary(root, selection)
            with (root / "REPORT.md").open("a", encoding="utf-8") as f:
                f.write(
                    "\n## 用户缩减后的执行范围\n\n仅完成7条线确认；14/28条线确认取消。详见 scope_amendment.json 与 [加速综合分析](ACCELERATION_SUMMARY.md)。\n"
                )
        try:
            receipt = archive(root, root.with_name(root.name + ".results.tar.gz"))
            atomic_json(root.with_name(root.name + ".bundle.json"), receipt)
        except Exception as exc:
            suite.save(status="archive_failed", error=str(exc), experiments_finished=True)
            raise
        atomic_json(state_file, dict(status=suite.state["status"], bundle_verified=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path)
    parser.add_argument("--resume-foundation", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    lock = root / ".scope_finish"
    lock.mkdir(exist_ok=True)
    state_file = root.with_name(root.name + ".finish7.status.json")
    with run_lock(lock):
        try:
            selection = read_json(root / "selection.json")
            if args.diagnosis:
                dest = root / "torch_root_cause_evidence"
                dest.mkdir(exist_ok=True)
                for name in ("summary.json", "provenance.json", "evidence.tar.gz"):
                    src = args.diagnosis / name
                    shutil.copy2(src, dest / name)
                    if sha256(src) != sha256(dest / name):
                        raise RuntimeError("Diagnosis evidence copy checksum mismatch")
            if args.resume_foundation:
                if read_json(root / "scope_amendment.json")["confirmation_budgets"] != [7]:
                    raise RuntimeError(
                        "Explicit foundation resume requires the existing b7 amendment"
                    )
            else:
                wait_for_boundary(root, selection, state_file)
            finish(root, selection, state_file)
        except BaseException as exc:
            atomic_json(
                state_file, dict(status="failed_or_stopped", error=str(exc), pid=os.getpid())
            )
            raise


if __name__ == "__main__":
    main()
