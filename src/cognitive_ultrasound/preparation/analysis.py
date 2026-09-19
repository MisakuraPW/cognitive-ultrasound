"""Patient-grouped comparisons and train-only fitted, causal risk probes."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .common import FEATURES, atomic_json, read_json, rows_from


def grouped_mean(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        v = row.get(metric)
        if v is not None and np.isfinite(v):
            groups[row["case"]].append(v)
    return {k: float(np.mean(v)) for k, v in groups.items()}


def association(rows, signal="uncertainty_mean", target="unobserved_mae"):
    groups = defaultdict(list)
    for row in rows:
        if row.get(target) is not None:
            groups[row["case"]].append((row[signal], row[target]))
    correlations = []
    for pairs in groups.values():
        a, b = np.asarray(pairs).T
        if len(pairs) >= 4 and np.ptp(a) > 1e-10 and np.ptp(b) > 1e-10:
            correlations.append(float(spearmanr(a, b).statistic))
    return float(np.mean(correlations)) if correlations else None


def paired_comparison(reference, candidate, gate):
    def key(r):
        return (r["case"], r["seed"], r["frame"])

    a, b = {key(r): r for r in reference}, {key(r): r for r in candidate}
    if not a or a.keys() != b.keys():
        return dict(passed=False, reasons=["Incomplete or unmatched case/frame/seed coverage"])
    patient = defaultdict(lambda: defaultdict(list))
    for k in a:
        for metric in ("psnr", "ssim", "unobserved_mae"):
            if a[k][metric] is None or b[k][metric] is None:
                return dict(passed=False, reasons=["Missing error measurement"])
            patient[k[0]][metric].append(b[k][metric] - a[k][metric])
    psnr = [np.mean(v["psnr"]) for v in patient.values()]
    ssim = [np.mean(v["ssim"]) for v in patient.values()]
    ar = np.mean(list(grouped_mean(reference, "unobserved_mae").values()))
    br = np.mean(list(grouped_mean(candidate, "unobserved_mae").values()))
    steady = [k for k in a if a[k]["timing_valid"] and b[k]["timing_valid"]]
    if not steady:
        return dict(passed=False, reasons=["No synchronized steady timing"])
    speedup = np.median([a[k]["algorithm_s"] for k in steady]) / np.median(
        [b[k]["algorithm_s"] for k in steady]
    )
    wall_speedup = np.median([a[k]["adapter_wall_s"] for k in steady]) / np.median(
        [b[k]["adapter_wall_s"] for k in steady]
    )
    corr_a, corr_b = association(reference), association(candidate)
    reasons = []
    checks = {
        "steady speedup": speedup >= gate["speedup"],
        "adapter speedup": wall_speedup > 1,
        "mean PSNR": np.mean(psnr) >= -gate["psnr_drop"],
        "mean SSIM": np.mean(ssim) >= -gate["ssim_drop"],
        "unobserved MAE": br <= ar * gate["mae_ratio"],
        "worst-case PSNR": min(psnr) >= -gate["worst_psnr_drop"],
        "uncertainty association": corr_a is not None
        and corr_b is not None
        and corr_b >= corr_a - gate["correlation_drop"],
    }
    reasons.extend(k for k, ok in checks.items() if not ok)
    return dict(
        passed=not reasons,
        reasons=reasons,
        speedup=float(speedup),
        adapter_speedup=float(wall_speedup),
        psnr_delta=float(np.mean(psnr)),
        ssim_delta=float(np.mean(ssim)),
        mae_ratio=float(br / max(ar, 1e-12)),
        worst_case_psnr_delta=float(min(psnr)),
        reference_correlation=corr_a,
        candidate_correlation=corr_b,
        cases=len(patient),
        frame_seed_pairs=len(a),
        caveat="Development screen; not a formal noninferiority or calibration guarantee",
    )


def select_variant(root, cfg, confirmation=False):
    root = Path(root)
    if confirmation:
        provisional = read_json(root / "screening.json")["variant"]
        candidates = [] if provisional == "reference" else [provisional]
        prefix = "confirm_"
    else:
        candidates, prefix = cfg["candidates"], "debug_"
    reference = rows_from(root / "jobs" / (prefix + "reference"))
    reference_result = root / "jobs" / (prefix + "reference") / "result.json"
    reference_complete = (
        reference_result.exists() and read_json(reference_result).get("status") == "completed"
    )
    results = {}
    for name in candidates:
        directory = root / "jobs" / (prefix + name)
        result = directory / "result.json"
        if (
            not reference_complete
            or not result.exists()
            or read_json(result).get("status") != "completed"
        ):
            results[name] = dict(passed=False, reasons=["Task not complete"])
        else:
            results[name] = paired_comparison(reference, rows_from(directory), cfg["gate"])
            fixed = root / "jobs" / ("fixed_" + name) / "result.json"
            if (
                not fixed.exists()
                or read_json(fixed).get("status") != "completed"
                or not read_json(fixed).get("passed")
            ):
                results[name]["passed"] = False
                results[name]["reasons"].append("Common-history sampler check missing or failed")
    eligible = [k for k, v in results.items() if v["passed"]]
    chosen = max(eligible, key=lambda k: results[k]["speedup"]) if eligible else "reference"
    record = dict(
        variant=chosen,
        results=results,
        scope="independent confirmation subset" if confirmation else "debug subset only",
        accelerated_default=bool(eligible) and confirmation,
        statistical_guarantee=False,
    )
    atomic_json(root / ("selection.json" if confirmation else "screening.json"), record)
    return record


def pairs(rows):
    """x_t uses observables only; y is error at t+1 under this recorded policy."""
    indexed = {(r["case"], r["seed"], r["frame"]): r for r in rows}
    result = []
    for (case, seed, frame), row in indexed.items():
        nxt = indexed.get((case, seed, frame + 1))
        if nxt and nxt["unobserved_mae"] is not None:
            result.append(
                dict(
                    case=case,
                    seed=seed,
                    frame=frame,
                    x=[row[n] for n in FEATURES],
                    y=nxt["unobserved_mae"],
                )
            )
    return result


def fit_ridge(x, y, ridge):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=coef.tolist())


def predict(model, x):
    x = np.atleast_2d(np.asarray(x, float))
    return np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]]) @ model["coef"]


def patient_mae(p, predictions):
    groups = defaultdict(list)
    for item, prediction in zip(p, predictions):
        groups[item["case"]].append(abs(item["y"] - prediction))
    return float(np.mean([np.mean(v) for v in groups.values()]))


def risk_probe(root, cfg, output):
    for cohort in ("train", "development", "confirmation"):
        completion = Path(root) / "jobs" / ("risk_" + cohort) / "result.json"
        if not completion.exists() or read_json(completion).get("status") != "completed":
            atomic_json(
                output / "result.json",
                dict(status="skipped", reason="Risk cohort incomplete: " + cohort),
            )
            return
    source = {
        k: pairs(rows_from(Path(root) / "jobs" / ("risk_" + k)))
        for k in ("train", "development", "confirmation")
    }
    minimum = cfg["risk"]
    training = source["train"]
    if (
        len(training) < minimum["minimum_rows"]
        or len({r["case"] for r in training}) < minimum["minimum_cases"]
    ):
        atomic_json(
            output / "result.json",
            dict(status="skipped", reason="Insufficient training trajectories"),
        )
        return
    if not source["development"] or not source["confirmation"]:
        atomic_json(
            output / "result.json",
            dict(status="skipped", reason="Missing independent risk cohorts"),
        )
        return
    groups = [{r["case"] for r in source[k]} for k in source]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
        raise ValueError("Patient leakage in risk probe")
    x, y = np.array([r["x"] for r in training]), np.array([r["y"] for r in training])
    specs = {
        "constant": [],
        "current_uncertainty": [0],
        "observed_residual": [3],
        "change": [4],
        "linear_all": list(range(len(FEATURES))),
    }
    models, scores = {}, {}
    for name, indices in specs.items():
        model = fit_ridge(x[:, indices], y, minimum["ridge"])
        models[name] = dict(model=model, indices=indices)
        scores[name] = {}
        for cohort in ("development", "confirmation"):
            data = source[cohort]
            estimates = predict(model, np.array([r["x"] for r in data])[:, indices])
            scores[name][cohort] = patient_mae(data, estimates)
    baseline = min((n for n in specs if n != "linear_all"), key=lambda n: scores[n]["development"])
    gain = 1 - scores["linear_all"]["confirmation"] / max(scores[baseline]["confirmation"], 1e-12)
    chosen = models["linear_all"]
    estimates = predict(chosen["model"], x)
    atomic_json(
        output / "model.json",
        dict(
            **chosen,
            features=FEATURES,
            thresholds=np.quantile(estimates, [1 / 3, 2 / 3]).tolist(),
            train_min=x.min(0).tolist(),
            train_max=x.max(0).tolist(),
            target="next-frame unobserved MAE under frozen CASL policy",
            variant=read_json(Path(root) / "selection.json")["variant"],
        ),
    )
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            scores=scores,
            strongest_baseline_selected_on_development=baseline,
            confirmation_relative_gain=gain,
            eligible=gain >= minimum["minimum_relative_gain"],
            pairs={k: len(v) for k, v in source.items()},
            caveat="Small linear probe, not calibrated clinical risk; no action-value model trained",
        ),
    )


def parity(root, output):
    a = Path(root) / "jobs/debug_reference"
    b = Path(root) / "jobs/debug_wrapper"
    differences = []
    for f in sorted(a.glob("*/*/frame_*.npz")):
        other = b / f.relative_to(a)
        if not other.exists():
            atomic_json(
                output / "result.json", dict(status="skipped", reason="Unmatched wrapper coverage")
            )
            return
        with np.load(f, allow_pickle=False) as x, np.load(other, allow_pickle=False) as y:
            differences.append(
                dict(
                    error=float(np.max(abs(x["prediction"] - y["prediction"]))),
                    same_action=bool(np.array_equal(x["next_action"], y["next_action"])),
                )
            )
    atomic_json(
        output / "result.json",
        dict(
            status="completed" if differences else "skipped",
            equivalent=bool(differences)
            and all(d["error"] <= 1e-4 and d["same_action"] for d in differences),
            max_error=max((d["error"] for d in differences), default=None),
            frames=len(differences),
        ),
    )


def report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    diagnostic_artifacts(root)
    ledger = read_json(root / "status.json") if (root / "status.json").exists() else {}
    text = [
        "# 准备性实验报告",
        "",
        "这是有限开发实验，不是完整复现或正式新颖性验证。",
        "",
        "| 任务 | 状态 | 本任务累计秒数 |",
        "|---|---|---:|",
    ]
    for name, value in ledger.get("jobs", {}).items():
        text.append(f"| {name} | {value['status']} | {value.get('elapsed_s', 0):.1f} |")
        if value.get("reason"):
            text.append(f"\n{name}：{value['reason']}\n")
    text += ["", "## 加速选择", ""]
    if (root / "selection.json").exists():
        selected = read_json(root / "selection.json")
        text += [
            f"日常候选：`{selected['variant']}`；加速准入：{selected['accelerated_default']}。",
            "",
            "详细门槛见 selection.json。通过只适用于本次病例、预算与时序，不是正式非劣结论。",
        ]
    else:
        text.append("尚无独立确认的加速结论。")
    text += [
        "",
        "## 机制与风险诊断",
        "",
        "当前未测误差、下一帧误差、动作边际收益分别记录，不能互相代替。",
    ]
    all_rows = []
    for folder in sorted((root / "jobs").glob("*")):
        rows = rows_from(folder)
        if not rows:
            continue
        all_rows.extend(dict(job=folder.name, **r) for r in rows)
        if folder.name.startswith(("debug_", "confirm_")):
            means = grouped_mean(rows, "psnr")
            future = pairs(rows)
            corr = association(rows)
            future_rows = [
                dict(case=p["case"], uncertainty_mean=p["x"][0], unobserved_mae=p["y"])
                for p in future
            ]
            text.append(
                f"\n{folder.name}：{len(means)} 病例，{len(rows)} 帧次；病例均值 PSNR={np.mean(list(means.values())):.3f}。当前/未来误差关联={corr}/{association(future_rows)}。不完整任务仅作中间结果。"
            )
            times = [r["algorithm_s"] for r in rows if r.get("timing_valid")]
            cold = [r["algorithm_s"] for r in rows if r.get("cold")]
            text.append(
                f"\n同步计时：有效热态中位秒数={float(np.median(times)) if times else None}；冷启动秒数={cold}。各病例首次热态调用不计入热态速度。"
            )
            if (folder / "examples.png").exists():
                text.append(f"\n![固定位置的定性示例](jobs/{folder.name}/examples.png)")
    if all_rows:
        keys = sorted(set().union(*(r.keys() for r in all_rows)))
        with (root / "frames_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        reference = [r for r in all_rows if r["job"] == "debug_reference"]
        if reference:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for case in sorted({r["case"] for r in reference}):
                rows = sorted([r for r in reference if r["case"] == case], key=lambda r: r["frame"])
                axes[0].plot(
                    [r["frame"] for r in rows], [r["unobserved_mae"] for r in rows], label=case[:12]
                )
                axes[1].scatter(
                    [r["uncertainty_mean"] for r in rows], [r["unobserved_mae"] for r in rows], s=12
                )
            axes[0].set(xlabel="frame", ylabel="unobserved MAE")
            axes[0].legend()
            axes[1].set(xlabel="current uncertainty", ylabel="current unobserved MAE")
            fig.tight_layout()
            fig.savefig(root / "risk_diagnostics.png", dpi=160)
            plt.close(fig)
            text += ["", "![风险诊断](risk_diagnostics.png)"]
    for folder in sorted((root / "jobs").glob("*")):
        result = folder / "result.json"
        if result.exists() and folder.name.startswith(
            ("bf_", "branches", "risk_probe", "tbig", "closed_loop", "parity")
        ):
            text += [
                "",
                f"## {folder.name}",
                "",
                f"[结构化结果](jobs/{folder.name}/result.json)",
                "",
            ]
            value = read_json(result)
            for key in ("status", "reason", "eligible", "equivalent", "interpretation", "caveat"):
                if key in value:
                    text.append(f"- {key}：{value[key]}")
            if (folder / "examples.png").exists():
                text.append(f"\n![阶段对比](jobs/{folder.name}/examples.png)")
    if (root / "branch_diagnostics.json").exists():
        text += [
            "",
            "## 同历史分支",
            "",
            "[分支汇总](branch_diagnostics.json)",
            "",
            "![预算诊断](budget_diagnostics.png)",
        ]
    text += [
        "",
        "## 结论边界",
        "",
        "- skipped / blocked / timed_out 不是成功，也不是方法被证伪。",
        "- 原版与加速版保留独立结果；未通过确认不得声称等效。",
        "- 分支结果使用未来真值，只说明受限候选下的事后机会；不是可实现规划。",
        "- BF 是无学长权重的论文启发实现；阶段完成不等于达到论文质量。",
        "- 开发结果按病例汇总；没有使用 test 调参，没有临床风险保证。",
    ]
    (root / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def diagnostic_artifacts(root):
    """Readable figures plus restricted hindsight summaries; no new inference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .common import frame_files

    for folder in (root / "jobs").glob("*"):
        if not folder.name.startswith(("debug_", "confirm_")):
            continue
        files = frame_files(folder)
        if not files:
            continue
        # Deterministic first cold/early/late examples, never choose by best score.
        chosen = sorted(set([0, min(2, len(files) - 1), len(files) - 1]))
        fig, axes = plt.subplots(len(chosen), 5, figsize=(12, 2.5 * len(chosen)), squeeze=False)
        for row, index in enumerate(chosen):
            with np.load(files[index], allow_pickle=False) as data:
                gt, pred, mask, u = (
                    data[k].squeeze() for k in ("target", "prediction", "mask", "uncertainty")
                )
                pictures = [gt, np.where(mask, gt, -1), pred, np.abs(gt - pred), u]
                for col, (title, picture) in enumerate(
                    zip(
                        ["target", "observed", "reconstruction", "absolute error", "uncertainty"],
                        pictures,
                    )
                ):
                    kwargs = (
                        dict(vmin=-1, vmax=1, cmap="gray")
                        if col < 3
                        else dict(cmap="magma", vmin=0, vmax=2 if col == 3 else None)
                    )
                    im = axes[row, col].imshow(picture, **kwargs)
                    axes[row, col].set_title(title)
                    axes[row, col].axis("off")
                    if col >= 3:
                        fig.colorbar(im, ax=axes[row, col], shrink=0.7)
        fig.tight_layout()
        fig.savefig(folder / "examples.png", dpi=140)
        plt.close(fig)
    for folder in (root / "jobs").glob("bf_*"):
        files = sorted(folder.glob("*/frame_*.npz"))
        if not files:
            continue
        with np.load(files[0], allow_pickle=False) as data:
            labels = ["target", "model"] + [
                k for k in data.files if k not in ("target", "mask", "model")
            ]
            fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 3), squeeze=False)
            for ax, label in zip(axes[0], labels):
                ax.imshow(data[label].squeeze(), vmin=-1, vmax=1, cmap="gray")
                ax.set_title(label.replace("_", " "), fontsize=9)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(folder / "examples.png", dpi=140)
            plt.close(fig)
    branch = root / "jobs/branches/result.json"
    if not branch.exists() or read_json(branch).get("status") != "completed":
        return
    records = read_json(branch)["records"]
    budget = defaultdict(list)
    actions = defaultdict(dict)
    for r in records:
        if r["kind"] == "budget":
            budget[int(r["action"])].append(r["one"]["unobserved_mae"])
        else:
            actions[(r["case"], r["frame"], r["seed"])][r["action"]] = (
                r["one"]["unobserved_mae"] + r["two"]["unobserved_mae"]
            ) / 2
    regret = [v["greedy"] - min(v.values()) for v in actions.values() if "greedy" in v]
    summary = dict(
        budget_mean_unobserved_mae={k: float(np.mean(v)) for k, v in budget.items()},
        mean_restricted_two_frame_greedy_regret=float(np.mean(regret)) if regret else None,
        branch_states=len(actions),
        warning="Exploratory state/seed averages, not patient-independent confidence intervals. Hindsight labels are not online features.",
    )
    atomic_json(root / "branch_diagnostics.json", summary)
    if budget:
        keys = sorted(budget)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(keys, [np.mean(budget[k]) for k in keys], marker="o")
        ax.set(xlabel="acquired lines (same history)", ylabel="unobserved MAE")
        fig.tight_layout()
        fig.savefig(root / "budget_diagnostics.png", dpi=150)
        plt.close(fig)
