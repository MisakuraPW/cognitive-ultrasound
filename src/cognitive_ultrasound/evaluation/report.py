import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def generate_report(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["identity"]["config"]
    patient_rows = []
    for method in cfg["methods"]:
        for budget in cfg["budgets"]:
            for name in manifest["identity"]["cases"]:
                folder = output / method / f"lines_{budget:03d}" / Path(name).stem
                if not (folder / "complete.json").exists():
                    continue
                done = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
                with open(folder / "frames.csv", encoding="utf-8") as f:
                    frames = list(csv.DictReader(f))
                values = {
                    "method": method,
                    "budget": budget,
                    "case": Path(name).stem,
                    "frames": len(frames),
                    "budget_mismatch_frames": done["budget_mismatch_frames"],
                }
                for key in [
                    *cfg["metrics"],
                    "dice_agreement",
                    "total_s",
                    "diffusion_s",
                    "entropy_s",
                    "action_s",
                    "projection_s",
                    "overhead_s",
                ]:
                    rows = frames[1:] if key.endswith("_s") else frames
                    if (
                        rows
                        and key in rows[0]
                        and not (key == "dice_agreement" and done["segmentation_excluded"])
                    ):
                        values[key] = float(np.mean([float(r[key]) for r in rows]))
                values["first_frame_s"] = float(frames[0]["total_s"])
                patient_rows.append(values)
    if not patient_rows:
        raise ValueError("No completed patient cases; no results to report")
    fields = sorted(set().union(*(r.keys() for r in patient_rows)))
    with open(output / "patients.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(patient_rows)
    groups = defaultdict(list)
    for row in patient_rows:
        groups[(row["method"], row["budget"])].append(row)
    summary = []
    for (method, budget), cases in sorted(groups.items()):
        row = {"method": method, "budget": budget, "patients": len(cases)}
        for key in fields:
            values = [
                float(c[key]) for c in cases if key in c and key not in ("method", "case", "budget")
            ]
            if values:
                row[key] = float(np.mean(values))
        summary.append(row)
    numeric = sorted(set().union(*(r.keys() for r in summary)))
    with open(output / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=numeric)
        writer.writeheader()
        writer.writerows(summary)

    def table(keys):
        keys = ["method", "budget", "patients", *keys]
        lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for row in summary:
            lines.append(
                "| "
                + " | ".join(
                    f"{row[k]:.5g}" if isinstance(row.get(k), float) else str(row.get(k, "未计算"))
                    for k in keys
                )
                + " |"
            )
        return "\n".join(lines)

    expected = len(cfg["methods"]) * len(cfg["budgets"]) * len(manifest["identity"]["cases"])
    mismatches = sum(r["budget_mismatch_frames"] for r in patient_rows)
    lines = [
        "# CASL reproduction report",
        "",
        "SYNTHETIC TEST ONLY; not research results."
        if cfg.get("synthetic_input")
        else "EchoNet evaluation (verify provenance in manifest).",
        f"Run status: {manifest['status']}",
        f"Completed patient/method/budget cases: {len(patient_rows)}/{expected}",
        "",
        "## Environment",
        "",
        "Actual hardware, software, commits, checkpoint hashes: manifest.json.",
        "",
        "## Dataset",
        "",
        f"Split: {cfg['split']}; requested frames: {cfg['frames']}.",
        "Short videos use all available frames without padding/repetition, matching upstream slicing.",
        "Frames are averaged within each patient, then patients receive equal weight.",
        "",
        "## Method",
        "",
        "Official CASL temporal DPS / SeqDiff; next-frame action; polar simulated lines.",
        "Uniform means the official rolling equispaced baseline. No raw channel-data claim.",
        "",
        "## Reconstruction results",
        "",
        table(cfg["metrics"]),
        "",
        "## Sampling efficiency",
        "",
        "See budget_quality.png and summary.csv.",
        "",
        "## Downstream",
        "",
        table(["dice_agreement"]) if cfg["segmentation"] else "Not run.",
        "Dice compares segmentation of reconstructed and fully observed images, not human annotation.",
        "Reference failures are excluded only from Dice, identically across policies.",
        "",
        "## Runtime",
        "",
        table(["first_frame_s", "total_s"]),
        "",
        "See runtime_analysis.md. Warm-up, metrics, disk export and visualization are outside loop timing.",
        "",
        "## Reproduction gap analysis",
        "",
        f"Actual budget mismatch frames: {mismatches}. Inspect trajectories before comparing nominal budgets.",
        "Paper reference: mean PSNR 23.2 dB at 7 lines, 500 test patients. This is a reference, not our result.",
        "Require full test coverage, pinned data/checkpoints, LPIPS, and parameter review before claiming reproduction.",
        "This report does not automatically declare paper-level reproduction complete.",
    ]
    (output / "CASL_reproduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "runtime_analysis.md").write_text(
        "# Runtime analysis\n\n"
        + (
            "Profile mode: posterior JIT, eager action/recovery with synchronization. "
            "These component timings cannot be compared directly to fully compiled throughput.\n\n"
            if cfg["profile"]
            else "Fully compiled recover: only synchronized total/projection measured. "
            "Component timings require a separate --profile run; missing values are not zero.\n\n"
        )
        + table(["diffusion_s", "entropy_s", "action_s", "projection_s", "overhead_s", "total_s"])
        + "\n\nTotal = diffusion + entropy + action + projection + overhead in profile mode. "
        "First frame uses full diffusion; steady values exclude frame zero for every patient. "
        "I/O, metrics, compilation and diagnostic entropy export are excluded.\n",
        encoding="utf-8",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(cfg["metrics"]), figsize=(5 * len(cfg["metrics"]), 4), squeeze=False
    )
    for ax, metric in zip(axes[0], cfg["metrics"]):
        for method in cfg["methods"]:
            rows = sorted([r for r in summary if r["method"] == method], key=lambda r: r["budget"])
            ax.plot(
                [r["budget"] for r in rows],
                [r.get(metric, np.nan) for r in rows],
                "o-",
                label=method,
            )
        ax.set(xlabel="Requested lines per frame", ylabel=metric.upper())
        ax.legend()
        ax.grid(alpha=0.25)
    if cfg.get("synthetic_input"):
        fig.suptitle("SYNTHETIC INTEGRATION TEST - NOT RESEARCH RESULTS", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "budget_quality.png", dpi=160)
    plt.close(fig)
    return output / "CASL_reproduction_report.md"
