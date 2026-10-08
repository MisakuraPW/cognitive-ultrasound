"""Report measured components separately from extrapolated full-batch wall time."""

import argparse
import json
from pathlib import Path

import numpy as np

from cognitive_ultrasound.preparation.common import atomic_json, read_json


def report(root):
    result=read_json(root/"result.json")
    source=Path(read_json(root/"identity.json")["source"])
    cfg=read_json(source/"config.json")
    manifest=read_json(source/"manifest.json")
    screens=result["screens"]
    base=next(x for x in screens if x["name"]=="baseline")
    accepted=[x for x in screens if x.get("verification",{}).get("passed") and x.get("selection_identical")]
    best=min(accepted,key=lambda x:x["mean_s"])
    serial=result["baseline"]
    options=[("baseline",serial)]
    if result.get("fast_verification",{}).get("passed") and result.get("fast_verification",{}).get("actions_identical"):
        options.append(("fast_serial",result["fast"]))
    if result.get("parallel_verification",{}).get("passed") and result.get("parallel_verification",{}).get("actions_identical"):
        options.append(("fast_parallel2",result["parallel"]))
    valid=[(name,x) for name,x in options if x.get("status")=="completed"]
    selected_name,selected=min(valid,key=lambda p:p[1]["process_work_s"])
    full_file=root/"full_video_verification.json"
    full=read_json(full_file) if full_file.exists() else None
    if full and full["verification"].get("passed") and full["verification"].get("actions_identical"):
        label,item=min((("full_fast_serial",full["serial"]),("full_fast_parallel2",full["parallel"])),key=lambda p:p[1]["process_work_s"])
        selected_name,selected=label,item
    eval_frames=sum(manifest["files"][n]["frames"] for cohort in ("development","confirmation") for n in manifest["cohorts"][cohort])
    # Every seed:6 fixed videos and2 methods*3 lambdas*(dynamic+matched) videos.
    multiplier=len(cfg["seeds"])*(len(cfg["budgets"]["fixed_sweep"])+4*len(cfg["training"]["lambdas"]))
    completed=sum(read_json(p)["frames"]*(2 if "matched" in read_json(p) else 1) for p in source.glob("jobs/*/*/complete.json"))
    remaining=max(0,eval_frames*multiplier-completed)
    # Measured two-case task includes setup; extrapolation retains this conservatively.
    historical=[read_json(p) for p in source.glob("jobs/*/*/complete.json")]
    observed_old_rate=sum(x["seconds"] for x in historical)/sum(x["frames"] for x in historical)
    old_eval_h=remaining*observed_old_rate/3600
    new_eval_h=remaining*selected["process_work_s"]/selected["frames"]/3600
    training={}
    training_checks=read_json(root/"training_equivalence.json") if (root/"training_equivalence.json").exists() else {}
    models=len(cfg["seeds"])*len(cfg["training"]["lambdas"])
    remaining_updates=models*cfg["training"]["updates"]
    for method in ("E1","E2"):
        control=training_checks.get(method,{})
        value=control.get("optimized",result["training"][method])
        records=value.get("measured_updates",[])
        committed=sum(len(list(d.glob("updates/*.npz"))) for d in source.glob(f"jobs/{method}*_train"))
        todo=max(0,remaining_updates-committed)
        if records:
            # The first update may compile additional branches; bracket observed real updates.
            times=[r["seconds"] for r in records]
            training[method]=dict(remaining_updates=todo,measured_update_s=times,training_verification=control.get("passed"),
                                  projected_hours_low=todo*min(times)/3600,
                                  projected_hours_high=todo*max(times)/3600,
                                  limitation="two real updates; different budgets/data/GS branches can cost more")
        else:
            training[method]=dict(remaining_updates=todo,status="not_measured",reason=value.get("error",value.get("reason")))
    if all("projected_hours_low" in x for x in training.values()):
        low=new_eval_h+sum(x["projected_hours_low"] for x in training.values())
        high=new_eval_h+sum(x["projected_hours_high"] for x in training.values())
        projection=dict(measured_component_sum_hours=[low,high],planning_range_hours=[low*.85,high*1.3],
                        uncertainty_margin="heuristic planning margin, NOT confidence interval")
    else:projection=dict(status="incomplete_training_estimate")
    data=dict(ef_input_gradient_speedup=base["mean_s"]/best["mean_s"],
              loop_speedup=(serial["process_work_s"]/serial["frames"])/(selected["process_work_s"]/selected["frames"]),selected_loop=selected_name,
              old_remaining_evaluation_hours=old_eval_h,new_remaining_evaluation_hours=new_eval_h,
              remaining_evaluation_frame_equivalents=remaining,training=training,projection=projection,
              validation=dict(fast=result["fast_verification"],parallel=result["parallel_verification"],gs=result["gs"],
                              full_video=full,training=training_checks),
              decision="USER_REQUIRED; original research batch stays stopped",adopted=False,
              limitations=["EF screen is fixed-input;64-frame loops plus2 full videos are not full dataset quality certification",
                           "Two updates do not establish convergence or exact long-training equivalence",
                           "Per-case latency under concurrency is not single-video FPS",
                           "Compile/transport/thread classification is separate from numerical verification",
                           "Historical case scientific records can be referenced only after lineage audit; old timings retain old runtime"])
    atomic_json(root/"report.json",data)
    lines=["# EF 工程加速短测报告","", "正式实验已暂停。本报告只说明工程短测，不自动恢复91项清单。", "",
           "|候选|EF请求均值 ms|数值通过|选线一致|", "|---|---:|---|---|"]
    for x in screens:
        lines.append(f"|{x['name']}|{x.get('mean_s',0)*1000:.2f}|{x.get('verification',{}).get('passed',False)}|{x.get('selection_identical',False)}|")
    lines.extend(["",f"EF固定输入请求提速：{data['ef_input_gradient_speedup']:.2f}倍。",
                  f"任务单位帧时间比：{data['loop_speedup']:.2f}倍，选择 {selected_name}；不同长度含加载/编译的比值仅作外推。",
                  "此任务时间含加载、JIT、同步推理、参考评估和输出；不是纯核心FPS。", "",
                  f"剩余评估：原配置外推 {old_eval_h:.1f}小时，候选配置外推 {new_eval_h:.1f}小时。", "",
                  "## 真实短训练与全量外推", "", "```json",json.dumps(dict(training=training,projection=projection),ensure_ascii=False,indent=2),"```", "",
                  "## 使用边界", "",*['- '+x for x in data['limitations']], "",
                  "原实验44个已完成病例和9次E2更新未被修改；校准在独立目录执行。",
                  "最终由用户决定继续全量还是缩小，recommendation.json中的adopted保持false。"])
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    report(p.parse_args().output)
