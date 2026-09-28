"""Append-only interpretation, Excel fact import and verified result bundles."""

import csv
import hashlib
import shutil
import tarfile
from pathlib import Path

import numpy as np

from ..preparation.common import atomic_json, read_json
from ..provenance import sha256


def archive(folder, destination, exclude_cache=True):
    """Verify each member against source before publishing a transport checksum."""
    files = [
        f
        for f in sorted(folder.rglob("*"))
        if f.is_file()
        and not f.is_symlink()
        and f.name not in (".run.lock", "run.lock", "STOP")
        and not f.name.endswith((".tmp", ".partial"))
        and not (exclude_cache and ".cache" in f.relative_to(folder).parts)
        and f.resolve() != destination.resolve()
    ]
    hashes = {str(f.relative_to(folder)).replace("\\", "/"): sha256(f) for f in files}
    if (
        shutil.disk_usage(destination.parent).free
        < sum(f.stat().st_size for f in files) * 1.02 + 2 * 1024**3
    ):
        raise RuntimeError("Insufficient space for verified result archive plus 2 GiB reserve")
    temporary = destination.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(folder)), recursive=False)
    with tarfile.open(temporary, "r:gz") as tar:
        seen = set()
        for member in tar:
            stream = tar.extractfile(member)
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
            if member.name not in hashes or h.hexdigest() != hashes[member.name]:
                raise ValueError(f"Archive verification failed: {member.name}")
            seen.add(member.name)
        if seen != set(hashes):
            raise ValueError("Archive member inventory differs")
    temporary.replace(destination)
    destination.with_name(destination.name + ".sha256").write_text(
        sha256(destination) + "  " + destination.name + "\n", encoding="ascii"
    )
    return dict(
        path=str(destination),
        sha256=sha256(destination),
        files=hashes,
        reference_cache_excluded=exclude_cache,
    )


def closure_audit(cfg, root):
    source = Path(cfg["closure_root"])
    if not source.exists():
        result = dict(
            status="blocked_missing_source",
            source=str(source),
            tasks=[],
            reason="Full closure-v4 results unavailable; summary alone is not verification",
        )
    else:
        tasks = []
        for task_file in sorted((source / "jobs").glob("*/task.json")):
            directory = task_file.parent
            result_file = directory / "result.json"
            value = read_json(result_file) if result_file.exists() else {}
            tasks.append(
                dict(
                    job=directory.name,
                    task=read_json(task_file),
                    status=value.get("status", "missing_result"),
                    result=str(result_file),
                    result_sha256=sha256(result_file) if result_file.exists() else None,
                    rerun=False,
                )
            )
        result = dict(
            status="audited" if tasks else "missing_tasks",
            source=str(source),
            tasks=tasks,
            tbig="blocked_official_assets; nonblocking",
            repairs="Only original missing tasks; no new cases/parameters",
            required_groups=[
                "budget_data",
                "prediction controls",
                "fair closed loop",
                "BF history",
            ],
        )
        receipt = root / "closure_bundle_receipt.json"
        if not receipt.exists():
            from ..preparation.suite import run_lock

            with run_lock(source):
                packed = archive(
                    source, root / "closure-v4.full-results.tar.gz", exclude_cache=False
                )
                atomic_json(receipt, packed)
        else:
            previous = read_json(receipt)
            if sha256(Path(previous["path"])) != previous["sha256"]:
                raise ValueError("Saved closure archive checksum mismatch")
        result["full_bundle_verified"] = read_json(receipt)
    atomic_json(root / "closure_audit.json", result)
    return result


def report(cfg, root):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for f in sorted((root / "jobs").glob("*/result.json")):
        r = read_json(f)
        if "profile" in r:
            rows = r.get("rows", [])
            warm = [x["core_s"] for x in rows if x["frame"] >= 2]
            r["summary"] = dict(
                mean_core_s=float(np.mean(warm)) if warm else None,
                p50_core_s=float(np.quantile(warm, 0.5)) if warm else None,
                p95_core_s=float(np.quantile(warm, 0.95)) if warm else None,
                cold_s=sum(x["core_s"] for x in rows if x["cold"]),
                load_s=r.get("model_load_s"),
                full_task_s=r.get("process_wall_s", r.get("task_wall_s")),
            )
            r["timing_distributions"] = {}
            for field in ("core_s", "closed_loop_s", "io_s", "output_enqueue_s"):
                values = [x[field] for x in rows if x["frame"] >= 2 and field in x]
                if values:
                    r["timing_distributions"][field] = dict(
                        mean=float(np.mean(values)),
                        p50=float(np.quantile(values, 0.5)),
                        p95=float(np.quantile(values, 0.95)),
                    )
            r["path"] = str(f)
            records.append(r)
        elif r.get("status") == "failed":
            records.append(dict(path=str(f), **r))
    atomic_json(root / "matrix.json", records)
    scheduling = read_json(root / "scheduling.json") if (root / "scheduling.json").exists() else {}
    with (root / "remaining_tasks.md").open("w", encoding="utf-8") as stream:
        stream.write("# 有界任务进度与成本决策\n\n")
        stream.write(
            "完成只表示执行完成；数值、质量、成本是独立结论。原始失败日志保留，不扩大搜索。\n\n"
        )
        stream.write("|阶段|范围|状态依据|\n|---|---|---|\n")
        for stage, scope, evidence in [
            ("JAX", "固定开发组", "jobs/development_jax_*"),
            ("Torch修复", "compile/graph短测及通过后的FP32开发", "jobs/short_torch_*"),
            (
                "Torch扩展",
                "只采用比官方DEV快且内部校验通过的compile/graph",
                "scheduling.json: torch_extension",
            ),
            ("确认", "开发选择至多4版本、7/14/28预算", "selection.json; jobs/confirmation_*"),
            ("历史收尾", "仅closure原设计缺失任务", "closure_completion.json"),
            (
                "训练恢复",
                "4负载×2模式×400真实更新，最多3200，另有短测/容量",
                "training_report.json",
            ),
            ("归档", "校验包、事实CSV、报告", "相邻 *.bundle.json"),
        ]:
            stream.write(f"|{stage}|{scope}|{evidence}|\n")
        stream.write("\n## 当前筛选\n\n")
        for name, value in scheduling.items():
            stream.write(f"- {name}: {value}\n")
        if (root / "estimates.json").exists():
            stream.write("\n## 短测估时（不保证，未含所有编译和I/O）\n\n")
            for r in read_json(root / "estimates.json")["candidates"]:
                if r.get("development_hours") is not None:
                    stream.write(
                        f"- {r['name']}: DEV {r['development_hours']:.3f} h；3预算确认 {r['confirmation_hours']:.3f} h。\n"
                    )
    fields = [
        "experiment",
        "cohort",
        "budget",
        "category",
        "status",
        "verdict",
        "quality_passed",
        "core_mean_s",
        "core_p50_s",
        "core_p95_s",
        "closed_loop_s",
        "full_task_s",
        "vram_peak_bytes",
        "ram_peak_bytes",
        "report_link",
    ]
    with (root / "excel_facts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            p, s = r.get("profile", {}), r.get("summary", {})
            writer.writerow(
                dict(
                    experiment=p.get("name", Path(r["path"]).parent.name),
                    cohort=r.get("cohort"),
                    budget=r.get("budget"),
                    category=p.get("category"),
                    status=r["status"],
                    verdict=r.get("verdict", "unassessed"),
                    quality_passed=r.get("quality", {}).get("passed"),
                    core_mean_s=s.get("mean_core_s"),
                    core_p50_s=s.get("p50_core_s"),
                    core_p95_s=s.get("p95_core_s"),
                    closed_loop_s=r.get("closed_loop_s"),
                    full_task_s=s.get("full_task_s"),
                    vram_peak_bytes=r.get("vram_peak_bytes"),
                    ram_peak_bytes=r.get("process_ram_peak_bytes", r.get("ram_peak_bytes")),
                    report_link=r["path"],
                )
            )
    lines = [
        "# 加速与计算基座审计",
        "",
        "默认 baseline：官方 FP32 / 50 步 / 逐步 DPS。没有自动采用近似版本。",
        "修复批次的继承记录见 inheritance.json，有限成本筛选见 scheduling.json，剩余任务见 remaining_tasks.md。",
        "A 表示修改意图，等价性必须另看数值、离散动作和完整闭环检查；B 必须独立比较。",
        "",
        "## 历史解释追加（不覆盖原结论）",
        "",
        "- official25_fp16 同时改变步数和精度，历史收益不能分别归因；旧宽松筛选不能代替本轮严格门槛。",
        "- 学长所述 4090 上 32 FPS 是待核实线索，并带有精度下降说明。已有 Torch 约 20 FPS 不是框架上限。",
        "- JAX 图内外 RNG 边界即使显式噪声一致，也可能改变融合路径；共同输入重放与真实闭环分别报告。",
        "- 原粒子路径已经使用 vmap，本轮没有把它包装成新增优化。",
        "",
        "## 本轮",
        "",
        f"已生成 {len(records)} 条任务记录；缺失结果不视为通过。",
        "",
        "|版本/任务|集合/预算|A/B|质量|等价性|",
        "|---|---|---|---|---|",
    ]
    for r in records:
        p = r.get("profile", {})
        lines.append(
            f"|{p.get('name', Path(r['path']).parent.name)}|{r.get('cohort', '')}/{r.get('budget', '')}|{p.get('category', '')}|{r.get('quality', {}).get('status', r['status'])}|{r.get('verdict', '未判定')}|"
        )
    lines += [
        "",
        "确认标准：病例均值 PSNR 下降≤0.1 dB、SSIM≤0.002、MAE增幅≤1%；单侧95%病例 bootstrap 界限同样须通过。最差病例分别≤0.5 dB/0.01/5%。各预算单独判定。",
        "未验证 LPIPS/临床任务协议时标为未评估。分割一致性不等于真值 Dice。",
        "计时含义：core 同步采样核心；closed_loop 包含传输和状态导出；process wall 包含进程、读取、诊断及输出，不能互换。短测微基准预热3次、计时20次。",
        "training_report.json 单独列出恢复组件、梯度/权重/优化器/EMA检查；200步结果不证明收敛等价。",
        "excel_facts.csv 仅为事实导入表，不修改 Prediction Lock、预测或核心认知。",
        "结果包排除 .cache 全量参照状态缓存；缓存留在服务器以供续跑。closure-v4.full-results.tar.gz 是完整历史结果包。",
        "",
        "数值风险依据：[PyTorch](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)、[JAX](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)。",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(records, root)


def plot(records, root):
    usable = [
        r
        for r in records
        if r.get("cohort") in ("development", "confirmation")
        and r.get("summary", {}).get("mean_core_s")
        and r.get("quality", {}).get("mean_loss")
    ]
    if not usable:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Separate sets/budgets; never connect across missing observations or different protocols.
    for cohort, budget in sorted({(r["cohort"], r["budget"]) for r in usable}):
        group = [r for r in usable if (r["cohort"], r["budget"]) == (cohort, budget)]
        fig, panels = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        axes = panels.ravel()
        for r in group:
            fps = 1 / r["summary"]["mean_core_s"]
            right = fps >= np.median([1 / x["summary"]["mean_core_s"] for x in group])
            passed = r["quality"]["passed"]
            style = dict(marker="o" if passed else "x", color="#0072B2" if passed else "#D55E00")
            for i in range(3):
                value = r["quality"]["mean_loss"][i]
                axes[i].scatter(fps, value, **style)
                axes[i].annotate(
                    r["profile"]["name"],
                    (fps, value),
                    fontsize=6,
                    xytext=(-4 if right else 4, 3),
                    textcoords="offset points",
                    ha="right" if right else "left",
                )
                upper = r["quality"].get("upper95")
                if upper is not None:
                    axes[i].plot([fps, fps], [value, upper[i]], color=style["color"], linewidth=0.7)
            if r.get("vram_peak_bytes") is not None:
                axes[3].scatter(fps, r["vram_peak_bytes"] / 1024**3, **style)
        for i, (threshold, label) in enumerate(
            zip((0.1, 0.002, 0.01), ("PSNR drop (dB)", "SSIM drop", "MAE relative increase"))
        ):
            axes[i].axhline(threshold, linestyle="--", color="black", label="Threshold")
            axes[i].set(
                xlabel="Synchronized warm core FPS (not end-to-end)", ylabel="Patient mean " + label
            )
        axes[3].set(
            xlabel="Synchronized warm core FPS", ylabel="Framework peak allocated VRAM (GiB)"
        )
        axes[0].legend()
        fig.suptitle(f"{cohort}, budget {budget}; circle=quality pass, cross=insufficient")
        file = root / f"tradeoff_{cohort}_{budget}"
        fig.savefig(file.with_suffix(".png"), dpi=180)
        fig.savefig(file.with_suffix(".pdf"))
        plt.close(fig)
    atomic_json(
        root / "figure_provenance.json",
        dict(
            source="matrix.json",
            source_sha256=sha256(root / "matrix.json"),
            estimand="patient mean PSNR drop; per-budget separate panels",
            uncertainty="confirmation: segment to one-sided 95% patient-bootstrap upper bound; development: point estimates",
            missing="omitted from scatter, retained in matrix.csv/JSON as missing",
            publisher="not specified",
        ),
    )
