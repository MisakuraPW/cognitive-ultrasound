"""CPU-only audit of a downloaded follow-up; never changes experiment outputs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return float(np.mean(values))


def predict(model, x):
    x = np.asarray(x)
    return np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]]) @ model["coef"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.input.resolve(), args.output.resolve()
    if root == out or root in out.parents:
        raise ValueError("Analysis must be outside immutable experiment outputs")
    out.mkdir(parents=True, exist_ok=True)
    status, selection = read(root / "status.json"), read(root / "selection.json")
    assert status["status"] == "completed"
    assert all(j["status"] == "completed" for j in status["jobs"].values())
    with (root / "frames_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    jobs = defaultdict(list)
    for r in rows:
        jobs[r["job"]].append(r)
    paired = []
    for case in sorted({r["case"] for r in jobs["confirm_reference"]}):
        means = {}
        for mode in ("reference", "official25_fp16"):
            rr = [r for r in jobs["confirm_" + mode] if r["case"] == case]
            means[mode] = {
                k: mean([float(r[k]) for r in rr]) for k in ("psnr", "ssim", "unobserved_mae")
            }
        paired.append(
            dict(
                case=case,
                **means,
                psnr_delta=means["official25_fp16"]["psnr"] - means["reference"]["psnr"],
                mae_ratio=means["official25_fp16"]["unobserved_mae"]
                / means["reference"]["unobserved_mae"],
            )
        )

    probe = read(root / "jobs/risk_probe/result.json")
    fitted = read(root / "jobs/risk_probe/model.json")
    features = fitted["features"]
    samples = {}
    for cohort in ("train", "development", "confirmation"):
        indexed = {(r["case"], r["seed"], int(r["frame"])): r for r in jobs["risk_" + cohort]}
        samples[cohort] = []
        for (case, seed, frame), r in indexed.items():
            nxt = indexed.get((case, seed, frame + 1))
            if nxt:
                samples[cohort].append(
                    dict(
                        case=case, x=[float(r[k]) for k in features], y=float(nxt["unobserved_mae"])
                    )
                )
    # Reconstruct the declared simple baseline from the archived training rows only.
    baseline = probe["strongest_baseline_selected_on_development"]
    indices = {"constant": [], "current_uncertainty": [0], "observed_residual": [3], "change": [4]}[
        baseline
    ]
    x = np.array([r["x"] for r in samples["train"]])[:, indices]
    y = np.array([r["y"] for r in samples["train"]])
    mu, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1)
    design = np.column_stack([np.ones(len(x)), (x - mu) / scale])
    penalty = np.eye(design.shape[1]) * read(root / "config.json")["risk"]["ridge"]
    penalty[0, 0] = 0
    simple = dict(
        mean=mu, scale=scale, coef=np.linalg.solve(design.T @ design + penalty, design.T @ y)
    )
    risk_cases = {}
    for cohort, data in samples.items():
        xx = np.array([r["x"] for r in data])
        pred = predict(fitted["model"], xx[:, fitted["indices"]])
        base = predict(simple, xx[:, indices])
        grouped = defaultdict(lambda: {"model": [], "baseline": []})
        for r, a, b in zip(data, pred, base):
            grouped[r["case"]]["model"].append(abs(a - r["y"]))
            grouped[r["case"]]["baseline"].append(abs(b - r["y"]))
        risk_cases[cohort] = [
            dict(
                case=k,
                model_mae=mean(v["model"]),
                baseline_mae=mean(v["baseline"]),
                relative_gain=1 - mean(v["model"]) / mean(v["baseline"]),
            )
            for k, v in sorted(grouped.items())
        ]
        if cohort != "train":
            assert np.isclose(
                mean([v["model_mae"] for v in risk_cases[cohort]]),
                probe["scores"]["linear_all"][cohort],
                atol=1e-10,
            )
            assert np.isclose(
                mean([v["baseline_mae"] for v in risk_cases[cohort]]),
                probe["scores"][baseline][cohort],
                atol=1e-10,
            )
    bf = read(root / "jobs/bf_diagnosis/result.json")
    bf_phases = {
        phase: {k: mean([r[phase][k] for r in bf["cases"]]) for k in bf["means"]}
        for phase in ("early", "late")
    }

    loop = read(root / "jobs/closed_loop/result.json")
    loop_audit = []
    for record in loop["records"]:
        d = root / "jobs/closed_loop" / str(record["seed"]) / Path(record["case"]).stem
        metrics = {p: {"all_pixel_mae": [], "common_unobserved_mae": []} for p in record["scores"]}
        previous, causes = None, []
        for index in range(len(record["schedules"]["adaptive"])):
            arrays = {}
            for policy in metrics:
                with np.load(d / policy / f"frame_{index:04d}.npz", allow_pickle=False) as data:
                    arrays[policy] = {
                        k: data[k].copy() for k in ("target", "prediction", "mask", "row")
                    }
            common = np.logical_and.reduce([a["mask"] == 0 for a in arrays.values()])
            target = arrays["adaptive"]["target"]
            for policy, a in arrays.items():
                assert np.array_equal(target, a["target"])
                err = abs(a["prediction"] - target)
                metrics[policy]["all_pixel_mae"].append(float(err.mean()))
                metrics[policy]["common_unobserved_mae"].append(float(err[common].mean()))
            row = json.loads(str(arrays["adaptive"]["row"]))
            if previous:
                bad = [
                    k
                    for k, lo, hi in zip(features, fitted["train_min"], fitted["train_max"])
                    if not lo <= previous[k] <= hi
                ]
                assert bool(bad) == row["out_of_training_range"]
                if bad:
                    causes.append(dict(frame=index, features=bad))
            previous = row
        loop_audit.append(
            dict(
                case=record["case"],
                seed=record["seed"],
                scores=record["scores"],
                additional_metrics={
                    p: {k: mean(v) for k, v in mm.items()} for p, mm in metrics.items()
                },
                fallback_causes=causes,
            )
        )

    branches = read(root / "jobs/branches/result.json")
    groups = defaultdict(dict)
    for r in branches["records"]:
        if r["kind"] == "action":
            groups[(r["case"], r["frame"], r["seed"])][r["action"]] = (
                r["one"]["unobserved_mae"] + r["two"]["unobserved_mae"]
            ) / 2
    action_states = [
        dict(
            case=k[0],
            frame=k[1],
            seed=k[2],
            candidates=v,
            greedy_regret=v["greedy"] - min(v.values()),
        )
        for k, v in groups.items()
        if "greedy" in v
    ]
    result = dict(
        source=str(root),
        completed_jobs=len(status["jobs"]),
        task_seconds=sum(j["elapsed_s"] for j in status["jobs"].values()),
        selection=selection,
        acceleration_cases=paired,
        risk_probe=probe,
        risk_cases=risk_cases,
        bf_means=bf["means"],
        bf_phases=bf_phases,
        loop_audit=loop_audit,
        branch_summary=read(root / "branch_diagnostics.json"),
        action_states=action_states,
        note="Offline descriptive analysis; no new inference, no independent confirmation, no retuned thresholds.",
    )
    (out / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "risk_per_case.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cohort", "case", "model_mae", "baseline_mae", "relative_gain"]
        )
        writer.writeheader()
        writer.writerows(dict(cohort=c, **r) for c, rr in risk_cases.items() for r in rr)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(np.arange(len(paired)) + 1, [r["psnr_delta"] for r in paired], color="#247ba0")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(
        xlabel="Confirmation case (sorted ID)",
        ylabel="PSNR change (dB)",
        title="25 warm steps + mixed FP16 vs reference",
    )
    rr = risk_cases["confirmation"]
    axes[1].bar(
        np.arange(len(rr)) + 1,
        [100 * r["relative_gain"] for r in rr],
        color=["#247ba0" if r["relative_gain"] > 0 else "#cb4b16" for r in rr],
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(
        xlabel="Confirmation case (sorted ID)",
        ylabel="Prediction MAE improvement (%)",
        title="Future-error probe: gains vary by case",
    )
    fig.tight_layout()
    fig.savefig(out / "casewise_results.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = {
        "online": "Recursive filter",
        "reset_every_3": "Reset every 3 frames",
        "privileged_previous_truth": "Previous truth (offline)",
        "interpolation": "Spatial interpolation",
        "codec_of_interpolation": "Codec of interpolation",
        "full_input_codec": "Full-input codec (offline)",
    }
    for k, label in names.items():
        curve = np.mean([r["per_frame_unobserved_mae"][k] for r in bf["cases"]], axis=0)
        ax.plot(np.arange(32), curve, label=label, linestyle="--" if "offline" in label else "-")
    ax.set(
        xlabel="Frame",
        ylabel="Unobserved MAE (normalized [-1, 1])",
        title="Frozen BF: four-case mean, same 14-line budget",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "bf_temporal_diagnosis.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                completed_jobs=result["completed_jobs"],
                risk_cases_improved=sum(r["relative_gain"] > 0 for r in rr),
                fallback_count=sum(len(r["fallback_causes"]) for r in loop_audit),
                output=str(out),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
