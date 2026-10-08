"""A first developmental subset that commits into the canonical full-batch case store."""

from pathlib import Path

from ..preparation.common import atomic_json, read_json


def pilot_ids(cfg):
    p = cfg.get("execution")
    if not p:
        return []
    seed, weight = p["pilot_seed"], p["pilot_lambda"]
    if seed not in cfg["seeds"] or weight not in cfg["training"]["lambdas"]:
        raise ValueError("Pilot must use an existing full-batch scientific point")
    count = p["pilot_cases"]
    if not 1 <= count <= cfg["cohorts"]["development"]:
        raise ValueError("Pilot cases must be a developmental subset")
    pair = p["pilot_fixed"]
    if pair not in cfg["budgets"]["fixed_sweep"]:
        raise ValueError("Pilot reference must already exist in the full fixed sweep")
    index = cfg["training"]["lambdas"].index(weight)
    return ["probe", f"E0_{pair[0]}_{pair[1]}_s{seed}_development",
            f"E1_l{index}_s{seed}_train", f"E1_l{index}_s{seed}_development",
            f"E2_l{index}_s{seed}_train", f"E2_l{index}_s{seed}_development"]


def execution_schedule(plan, cfg, manifest):
    ids = pilot_ids(cfg)
    if not ids:
        return [(s, False) for s in plan]
    by_id = {s["id"]: s for s in plan}
    first = []
    for name in ids:
        s = dict(by_id[name])
        if s["kind"] == "evaluate":
            s.update(_pilot=True, _case_names=manifest["cohorts"]["development"][:cfg["execution"]["pilot_cases"]])
        first.append((s, True))
    # Full plan keeps all91 original IDs. Duplicate execution visits reuse committed work.
    return first + [(None, True)] + [(s, False) for s in plan]


def pilot_report(root, cfg, manifest):
    from .report import summarize

    ids = pilot_ids(cfg)
    names = manifest["cohorts"]["development"][:cfg["execution"]["pilot_cases"]]
    rows = []
    failed = []
    for name in ids:
        directory = root/"jobs"/name
        if name.endswith("_train") or name == "probe":
            value = read_json(directory/"result.json") if (directory/"result.json").exists() else {}
            if value.get("status") != "completed":
                failed.append(name)
            continue
        records = []
        for case in names:
            file = directory/Path(case).stem/"complete.json"
            if file.exists():
                records.append(read_json(file))
        if len(records) != len(names):
            failed.append(name)
        summary = summarize(records)
        summary.update(job=name, method=name[:2])
        rows.append(summary)
    value = dict(status="completed" if not failed else "failed", jobs=ids, cases=names,
                 summaries=rows, failed=failed, subset_of_full_batch=True,
                 interpretation="Single-seed developmental result; no convergence/multi-seed/confirmation claim",
                 continuation="Automatically continue unchanged full plan; reuse these same case commits")
    atomic_json(root/"pilot_summary.json", value)
    lines = ["# 首批子集：E0 / E1 / E2 初步对照", "",
             "种子42、λ=2、固定参照10+4，开发集固定前4病例。科学配置及50次训练更新与全量完全相同。",
             "这些病例写入全量的同一结果目录，后续直接跳过，不重复推理。",
             "此为单种子的开发集初步结果，不证明收敛或独立确认效果。确认集仍留到最后。", "",
             "|方法|病例|EF MAE|平均线数|视频闭环秒|动态减匹配固定对照 MAE|",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["cases"]:
            delta=r.get("matched",{}).get("candidate_minus_control_mae")
            lines.append(f"|{r['method']}|{r['cases']}|{r['absolute_error']:.4f}|{r['mean_lines']:.3f}|{r['seconds']:.2f}|{delta if delta is not None else '—'}|")
    if failed:
        lines += ["", "未完成项目："+", ".join(failed)]
    lines += ["", "历史病例的耗时保留原运行配置；工程优化后的新病例另保留来源，不将混合耗时解释为纯硬件测速。"]
    (root/"PILOT_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return value
