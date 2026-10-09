"""Video-level paired reports; sparse/absent results stay explicit, no adoption decision."""

import csv
import shutil
from collections import defaultdict

import numpy as np

from ..compute_lab.report import archive
from ..preparation.common import atomic_json, read_json
from ..provenance import sha256


def summarize(records):
    cases = defaultdict(list)
    for r in records:
        cases[r["case"]].append(r)
    result = dict(cases=len(cases), observations=len(records))
    if not cases:
        return result

    def case_means(key):
        return np.array([np.mean([r[key] for r in rows]) for rows in cases.values()])

    for field in (
        "absolute_error",
        "mean_lines",
        "seconds",
        "mean_psnr",
        "mean_ssim",
        "mean_mae",
        "prediction_preservation_error",
        "perception_calls",
        "full_input_absolute_error",
    ):
        result[field] = float(case_means(field).mean())
    result["rmse"] = float(
        np.sqrt(
            np.mean([np.mean([r["absolute_error"] ** 2 for r in rows]) for rows in cases.values()])
        )
    )
    errors = case_means("absolute_error")
    boot = np.random.default_rng(20261008).integers(len(cases), size=(5000, len(cases)))
    result["ef_mae_ci95"] = np.quantile(errors[boot].mean(1), [0.025, 0.975]).tolist()
    if all("matched" in r for r in records):
        delta = np.array(
            [
                np.mean([r["absolute_error"] - r["matched"]["absolute_error"] for r in rows])
                for rows in cases.values()
            ]
        )
        mismatch = max(r["matched"]["mean_line_mismatch"] for r in records)
        result["matched"] = dict(
            ef_mae=float(
                np.mean(
                    [
                        np.mean([r["matched"]["absolute_error"] for r in rows])
                        for rows in cases.values()
                    ]
                )
            ),
            mean_lines=float(
                np.mean(
                    [np.mean([r["matched"]["mean_lines"] for r in rows]) for rows in cases.values()]
                )
            ),
            candidate_minus_control_mae=float(delta.mean()),
            paired_video_bootstrap_ci95=np.quantile(delta[boot].mean(1), [0.025, 0.975]).tolist(),
            maximum_mean_line_mismatch=mismatch,
            resource_match_qualified=mismatch <= 0.1,
            interpretation="descriptive paired evidence, not automatic efficacy certification",
        )
    return result


def report(root):
    cfg = read_json(root / "config.json")
    manifest = read_json(root / "manifest.json")
    groups = defaultdict(list)
    ledger = []
    expected = read_json(root / "plan.json") if (root / "plan.json").exists() else []
    for job in expected:
        file = root / "jobs" / job["id"] / "result.json"
        value = read_json(file) if file.exists() else dict(status="missing")
        ledger.append(
            dict(
                job=job["id"],
                status=value["status"],
                reason=value.get("reason"),
                returncode=value.get("returncode"),
            )
        )
        if job["kind"] == "evaluate" and value["status"] == "completed":
            label = (
                "fixed" + str(job["fixed"]) if job["method"] == "E0" else f"lambda={job['lambda']}"
            )
            if "checkpoint_update" in job:
                label+=f",updates={job['checkpoint_update']}"
            groups[(job["method"], label, job["cohort"])].extend(value["records"])
    summaries = []
    for (method, label, cohort), records in sorted(groups.items()):
        summary = summarize(records)
        expected_cases=len(manifest["cohorts"][cohort])
        if cfg.get("execution",{}).get("repair_milestones") and cohort=="development" and "updates=" in label:
            update=int(label.split("updates=")[-1])
            if update<cfg["training"]["updates"]:expected_cases=4
        summary.update(
            method=method,
            working_point=label,
            cohort=cohort,
            complete=summary["cases"] == expected_cases
            and summary["observations"] == expected_cases * len(cfg["seeds"]),
        )
        summaries.append(summary)
    atomic_json(
        root / "summary.json",
        dict(
            points=summaries,
            ledger=ledger,
            statistics="video is replication unit; equal seeds within video; 5000 paired-video bootstrap, seed20261008",
            default_baseline="E0 on shared explicitly selected perception/EF/task-selector settings",
            multitask_training=False,
            segmentation_evaluated=False,
            gs_scope=cfg["training"]["gs_gradient"],
        ),
    )
    with (root / "facts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "working_point",
                "cohort",
                "complete",
                "cases",
                "ef_mae",
                "ef_rmse",
                "mean_lines",
                "video_seconds",
                "Prediction",
                "Prediction Lock",
                "Belief Update",
            ],
        )
        writer.writeheader()
        for r in summaries:
            writer.writerow(
                dict(
                    method=r["method"],
                    working_point=r["working_point"],
                    cohort=r["cohort"],
                    complete=r["complete"],
                    cases=r["cases"],
                    ef_mae=r["absolute_error"],
                    ef_rmse=r["rmse"],
                    mean_lines=r["mean_lines"],
                    video_seconds=r["seconds"],
                )
            )
    lines = [
        "# EF two-stage dynamic budget: observed batch report",
        "",
        "E0 fixed, E1 straight-through Gumbel-Softmax, E2 REINFORCE. Frozen CASL and EF weights.",
        "Only completed records are summarized; missing/failed jobs remain in the ledger.",
        "No multitask training or segmentation evaluation in this batch.",
        "GS estimator: "+cfg["training"]["gs_gradient"]+"; approximate estimator, not a convergence guarantee.",
        "For K2=0, GS uses a direct-projection backward surrogate and no extra DPS.",
        "Speed: preloaded video acquisition plus EF inference; full-input reference evaluation excluded.",
        "Normalization/scanning conversion and pretrained split assumptions are in manifest/identity/EF runtime.",
        "",
        "|Method|Working point|Set|Complete|Videos|EF MAE (pp)|RMSE (pp)|Lines/frame|Video compute s|",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    if cfg.get("functional_fixture", False):
        lines.insert(2, "FUNCTIONAL SYNTHETIC FIXTURE — no scientific outcome.")
    if cfg.get("execution",{}).get("repair_milestones"):
        lines.insert(2,"Repair version: frozen TRAIN-only input scales; E1 bounded local-projection GS surrogate; E2 action-independent full-input reward reference and fixed score scaling. Historical E1 weights were not resumed.")
        lines.insert(3,"Compatible E0 quality records may be inherited. Their original timings retain their original runtime; mixed-version times cannot establish a budget-policy speedup.")
    for r in summaries:
        lines.append(
            f"|{r['method']}|{r['working_point']}|{r['cohort']}|{r['complete']}|{r['cases']}|"
            f"{r['absolute_error']:.4f}|{r['rmse']:.4f}|{r['mean_lines']:.3f}|{r['seconds']:.2f}|"
        )
    lines += ["", "## Missing/failed jobs", ""]
    lines += [
        f"- {r['job']}: {r['status']} — {r.get('reason') or ''}"
        for r in ledger
        if r["status"] != "completed"
    ]
    if not summaries:
        lines += [
            "",
            "No scientific outcome yet. Probe/calibration does not establish policy efficacy.",
        ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summaries:
        figures(root, summaries, synthetic=cfg.get("functional_fixture", False))
    return summaries


def figures(root, points, synthetic=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exports = {}
    for cohort in ("development", "confirmation"):
        subset = [r for r in points if r["cohort"] == cohort]
        if not subset:
            continue
        with plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
            for method, color, marker in (
                ("E0", "#333333", "o"),
                ("E1", "#0072B2", "s"),
                ("E2", "#D55E00", "^"),
            ):
                rows = [r for r in subset if r["method"] == method]
                for i, r in enumerate(rows):
                    lo, hi = r["ef_mae_ci95"]
                    for ax, key, label in zip(
                        axes,
                        ("mean_lines", "seconds"),
                        ("Mean scan lines / frame", "Full-video compute (s)"),
                    ):
                        ax.errorbar(
                            r[key],
                            r["absolute_error"],
                            yerr=[[r["absolute_error"] - lo], [hi - r["absolute_error"]]],
                            fmt=marker,
                            color=color,
                            markerfacecolor=color if r["complete"] else "white",
                            label=method if i == 0 else None,
                            capsize=3,
                        )
                        ax.set(xlabel=label, ylabel="Video-level EF MAE (percentage points)")
                        ax.grid(alpha=0.2)
            for ax in axes:
                ax.legend()
            prefix = "FUNCTIONAL SYNTHETIC / " if synthetic else ""
            fig.suptitle(prefix + f"{cohort}: observed working points (95% video bootstrap CI)")
            # Scatter only: no invented interpolation/Pareto frontier between observed points.
            for suffix in ("png", "pdf"):
                file = root / f"ef_resource_{cohort}.{suffix}"
                fig.savefig(file, dpi=200, facecolor="white")
                exports[file.name] = sha256(file)
            plt.close(fig)
    atomic_json(
        root / "figure_provenance.json",
        dict(
            source="summary.json",
            source_sha256=sha256(root / "summary.json"),
            outputs=exports,
            transform="equal-video/equal-seed aggregation, 95% video bootstrap CI; no smoothing/interpolation",
            missing="ledger retained; partial points hollow",
            publisher="not specified; general provisional figures",
            software_guidance="Scientific Agent Skills, Kassis et al. (2026), doi:10.48550/arXiv.2609.00065",
        ),
    )


def bundle(root):
    """Small analysis-only package. Checkpoints/recovery traces remain separately on server."""
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    names = [
        "REPORT.md",
        "facts.csv",
        "summary.json",
        "manifest.json",
        "identity.json",
        "config.json",
        "plan.json",
        "status.json",
        "figure_provenance.json",
    ]
    names += [p.name for p in root.glob("ef_resource_*")]
    for name in names:
        source = root / name
        if source.exists():
            shutil.copy2(source, analysis / name)
    file = root.with_name(root.name + ".analysis.tar.gz")
    result = archive(analysis, file)
    atomic_json(file.with_name(file.name + ".bundle.json"), result)
    return result
