"""Export an append-only repair addendum; retain the verified original result bundle."""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from cognitive_ultrasound.compute_lab.protocol import quality
from cognitive_ultrasound.compute_lab.report import archive
from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.preparation.suite import run_lock
from cognitive_ultrasound.provenance import sha256


def finalize(root, attempt="gap_repair"):
    if attempt not in ("gap_repair","gap_repair_classification"):
        raise ValueError("Unknown bounded repair attempt")
    repair = root / attempt
    with run_lock(root):
        state = read_json(repair / "status.json")
        if state["status"] not in ("completed", "completed_with_rejections"):
            raise RuntimeError("Repair jobs are not terminal")
        base = read_json(root.with_name(root.name + ".bundle.json"))
        base_path = Path(base["path"])
        if sha256(base_path) != base["sha256"]:
            raise ValueError("Original archive checksum changed")
        plan = read_json(repair / "plan.json")
        for name, value in plan["original_results"].items():
            if sha256(root / "jobs" / name / "result.json") != value:
                raise ValueError("Original failed evidence changed")
        for name, key in (("identity.json", "original_identity_sha256"), ("selection.json", "original_selection_sha256"), ("config.json", "scientific_config_sha256")):
            if sha256(root / name) != plan[key]:
                raise ValueError("Original science or selection changed")
        reference = read_json(root / "jobs/short_official_b14/result.json")
        records = []
        for name in plan["jobs"]:
            r = read_json(repair / "jobs" / name / "result.json")
            warm = [x for x in r.get("rows", []) if not x["cold"] and not x.get("warm_signature_first")]
            checks = r.get("operator_checks") or {}
            entry = dict(job=name, status=r["status"], error=r.get("error"),
                         strict_internal_passed=bool(checks.get("internal_correctness")),
                         runtime_integrity=bool(checks.get("runtime_integrity")),
                         compiled_eager_checks=checks.get("compiled_eager_checks"),
                         cross_official_equivalent=r.get("equivalence_passed", False),
                         compiler_option=r.get("compiler_option"),
                         warm_core_ms=float(np.mean([x["core_s"] for x in warm])*1000) if warm else None,
                         warm_closed_loop_ms=float(np.mean([x["closed_loop_s"] for x in warm])*1000) if warm else None,
                         elapsed_s=r.get("process_wall_s"),
                         scope="original 2 debug cases / 4 frames, not confirmation",
                         quality=None, automatic_adoption=False)
            if r["status"] == "completed":
                entry["quality"] = quality(reference["rows"], r["rows"], confirmation=False)
            records.append(entry)
        atomic_json(repair / "repair_matrix.json", records)
        training = read_json(root / "training_report.json")
        training_rows = []
        for workload, value in training.items():
            for mode in ("eager", "graph"):
                v = value[mode]
                training_rows.append(dict(workload=workload, mode=mode,
                    resume_bitwise=v["recovery"]["bitwise"] and v["recovery_loss"]["bitwise"],
                    resume_passed=v["recovery"]["passed"] and v["recovery_loss"]["passed"],
                    process_seconds=v["elapsed_s"],
                    graph_eager_passed=value["engineering_comparison"]["passed"] and value["engineering_loss"]["passed"],
                    recommendation=value["recommended"]))
        atomic_json(repair / "training_summary.json", training_rows)
        lines = ["# 加速与计算基座：失败排查及最终导出", "",
                 "本附录追加于原始结果；原 status.json 的 finished_with_gaps 和四个失败结果保留为历史记录。",
                 "本轮仅补查并定向重跑四个 Torch 低精度短测。五组7线确认、已完成训练和历史收尾均未重复运行；14/28线确认仍取消。", "",
                 "## 失败原因与修复范围", "",
                 "旧短测在 Torch compiled/eager 检查处停止，尚未进入完整质量评估。第一项定位尝试保留低精度中间舍入，但四项仍未通过，因此否定了这一单一修复假设，原记录一并保留。",
                 "后续修复的是流水线分类：B类的有限浮点差异仍记为2e-4数值不等价，不再等同于程序崩溃。非有限值、形状/类型变化、重复执行不稳定、梯度丢失、图捕获与未捕获执行不一致仍然中止。该重跑使用原编译设置，不宣称消除了误差，也不将内部核对改为通过。",
                 "运行成功、内部一致、跨框架等价和质量通过是不同判据。短测质量只作诊断，不能替代开发或确认；不因修复成功自动改变候选选择或默认 baseline。", "",
                 "|短测|运行|严格内部核对|与官方闭环等价|热核心 ms|热闭环 ms|", "|---|---|---|---|---:|---:|"]
        for r in records:
            fmt = lambda x: "未测得" if x is None else f"{x:.2f}"
            lines.append(f"|{r['job']}|{r['status']}|{r['strict_internal_passed']}|{r['cross_official_equivalent']}|{fmt(r['warm_core_ms'])}|{fmt(r['warm_closed_loop_ms'])}|")
            if r["error"]:
                lines += ["", f"该项拒绝原因：`{r['error'].splitlines()[0]}`。原始日志见 gap_repair/jobs。", ""]
        lines += ["", "### 短测质量（仅2例，不能作为确认结论）", "",
                  "|版本|PSNR下降 dB|SSIM下降|MAE相对增加|debug门槛|", "|---|---:|---:|---:|---|"]
        for r in records:
            q=r.get("quality") or {}
            loss=q.get("mean_loss")
            if loss:
                lines.append(f"|{r['job']}|{loss[0]:.4f}|{loss[1]:.5f}|{loss[2]:.2%}|{q['passed']}|")
            else:
                lines.append(f"|{r['job']}|未评估|未评估|未评估|未判定|")
        lines += ["", "负数表示对应指标改善。这些debug病例只用于定位，不参加确认集筛选，也不能用两例结果宣称总体质量等价。"]
        lines += ["", "## 训练与断点恢复", "",
                  "|负载|eager 200步完整进程秒|graph 200步完整进程秒|两模式各自续训逐位一致|graph/eager 容差等价|当前建议|", "|---|---:|---:|---|---|---|"]
        for work, v in training.items():
            own = all(v[m]["recovery"]["bitwise"] and v[m]["recovery_loss"]["bitwise"] for m in ("eager", "graph"))
            same = v["engineering_comparison"]["passed"] and v["engineering_loss"]["passed"]
            lines.append(f"|{work}|{v['eager']['elapsed_s']:.2f}|{v['graph']['elapsed_s']:.2f}|{own}|{same}|{v['recommended']}|")
        lines += ["", "每模式比较连续200步与100步退出后恢复100步，包括参数、梯度、优化器、EMA（适用时）和损失轨迹。200步不能证明完整训练收敛等价。",
                  "CASL 图模式未通过与 eager 的严格比较，这是保留的实验结果，不能放宽门槛后标为工程等价；已验证的各自精确续训能力仍成立。", "",
                  "## 7线确认与使用建议", "",
                  (root / "ACCELERATION_SUMMARY.md").read_text(encoding="utf-8"), "",
                  "## 交付与未覆盖项", "",
                  "- 原结果包包含全部旧结果、逐帧指标、版本矩阵、质量/速度图、训练快照和 closure 完整归档。",
                  "- 本修复附录包包含新短测、原失败的映射、编译配置、训练摘要与本报告；两包共同组成最终交付，避免重复传输原1 GiB结果。",
                  "- TBIG 官方资产缺失、LPIPS/任务评价协议未核实、取消的14/28线确认明确保持未评估。",
                  "- 正式参照仍为官方版本；任何近似版本需要用户选择，并在后续相同科学配置上重建对应 baseline。",
                  "- 不再自动增加实验或优化搜索；数值拒绝属于候选结论，不伪装成已修复通过。"]
        (repair / "FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        with (repair / "excel_facts_append.csv").open("w",encoding="utf-8-sig",newline="") as f:
            fields = ["job","status","strict_internal_passed","cross_official_equivalent","warm_core_ms","warm_closed_loop_ms","elapsed_s","scope","automatic_adoption"]
            w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(records)
        # Original files are never overwritten. This receipt binds the base and addendum.
        atomic_json(repair / "base_bundle_reference.json",dict(name=base_path.name,sha256=base["sha256"],size=base_path.stat().st_size))
        if attempt == "gap_repair_classification":
            prior=root/"gap_repair"
            previous=repair/"prior_cast_attempt"
            previous.mkdir(exist_ok=True)
            for name in ("plan.json","summary.json","status.json"):
                shutil.copy2(prior/name,previous/name)
            for source in (prior/"jobs").glob("*/*"):
                if source.name in ("console.log","result.json","task.json"):
                    dest=previous/"jobs"/source.parent.name/source.name
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        receipt=archive(repair,root.with_name(root.name+"."+attempt+".tar.gz"),exclude_cache=True)
        atomic_json(root.with_name(root.name+"."+attempt+".bundle.json"),receipt)
        # Small default deliverable: no model/optimizer snapshots, frame arrays,
        # raw profiler traces, nested archives or duplicated per-frame JSON.
        analysis=root/"analysis_delivery"
        analysis.mkdir(exist_ok=True)
        for name in ("FINAL_REPORT.md","repair_matrix.json","training_summary.json","excel_facts_append.csv","plan.json","base_bundle_reference.json"):
            shutil.copy2(repair/name,analysis/name)
        for name in ("ACCELERATION_SUMMARY.md","REPORT.md","excel_facts.csv","scope_amendment.json","selection.json","config.json","closure_completion.json","torch_root_cause_report.md","figure_provenance.json"):
            source=root/name
            if source.exists():shutil.copy2(source,analysis/name)
        for source in root.glob("tradeoff_*"):
            if source.suffix in (".png",".pdf"):shutil.copy2(source,analysis/source.name)
        matrix=root/"matrix.json"
        if matrix.exists():
            fields=("path","status","profile","cohort","budget","verdict","quality","equivalence_passed","bitwise","summary","timing_distributions","vram_peak_bytes","process_ram_peak_bytes")
            atomic_json(analysis/"version_summary.json",[{k:r[k] for k in fields if k in r} for r in read_json(matrix)])
        (analysis/"README.md").write_text("# 轻量分析包\n\n先读 FINAL_REPORT.md。本包包含结论、病例聚合质量、速度、图表和事实CSV；不含模型快照、逐帧数组、原始trace或完整历史归档。完整证据与修复附录的SHA-256见delivery_manifest.json。\n",encoding="utf-8")
        atomic_json(analysis/"delivery_manifest.json",dict(base=dict(name=base_path.name,sha256=base["sha256"]),addendum=dict(name=Path(receipt["path"]).name,sha256=receipt["sha256"])))
        light=archive(analysis,root.with_name(root.name+".analysis.tar.gz"))
        atomic_json(root.with_name(root.name+".analysis.bundle.json"),light)
        final=dict(status="completed" if all(r["status"]=="completed" for r in records) else "completed_with_rejected_candidates",
                   experiments_terminal=True,bundles_verified=True,automatic_adoption=False,
                   original_status_retained=True,base=dict(path=str(base_path),sha256=base["sha256"]),
                   addendum=dict(path=receipt["path"],sha256=receipt["sha256"]),
                   analysis=dict(path=light["path"],sha256=light["sha256"]),
                   failed_after_retry=[r["job"] for r in records if r["status"]!="completed"],
                   scientific_non_equivalence=["CASL graph training vs eager; retain eager reference"])
        atomic_json(root.with_name(root.name+".final_export.json"),final)
        print(final,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path)
    p.add_argument("--attempt",choices=("gap_repair","gap_repair_classification"),default="gap_repair")
    args=p.parse_args();finalize(args.output.resolve(),args.attempt)
