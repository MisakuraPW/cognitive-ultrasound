"""P0..P5 use shared frozen kernels, explicit interventions and atomic case units."""

import copy
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..preparation.common import atomic_json, digest, read_json
from ..provenance import sha256
from .data import read_episode
from .protocol import causal_window, greedy_order
from .value_protocol import (
    LEVELS,
    anchors,
    blocks,
    candidates,
    common_unobserved,
    hybrid,
    mixed,
    periodic,
    predictability,
    quality,
    reverse,
    uniform,
    value_labels,
)
from .value_runtime import Runtime, trajectory
from .value_storage import commit, committed, compare_trees, load_tree


def reference(rt, name):
    directory = rt.root / "references" / Path(name).stem
    unit = dict(batch=rt.identity, case=name, source=rt.manifest["files"][name]["sha256"])
    if not rt.cfg["value_diagnostics"]["functional_fixture"]:
        raw = Path(rt.cfg["raw_videos"]) / (Path(name).stem + ".avi")
        unit["raw_sha"] = sha256(raw)
    old = committed(directory, unit)
    if old:
        return old
    target = read_episode(rt.cfg, rt.manifest, name)
    truth = rt.manifest["files"][name]["ef"]
    polar = rt.readout(target, truth, condition="full_polar_reference")
    if rt.cfg["value_diagnostics"]["functional_fixture"]:
        original = copy.deepcopy(polar)
    else:
        import cv2

        cap = cv2.VideoCapture(str(raw))
        images = []
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if len(images) != len(target) or any(x.shape != (112, 112, 3) for x in images):
            raise ValueError("Original and polar time indexes/geometry differ")
        original = rt.readout(
            np.stack(images), truth, domain="cartesian_rgb", condition="original_reference"
        )
    return commit(
        directory,
        unit,
        dict(
            polar=polar,
            original=original,
            scope="full input task reference, not a clinical truth bound",
        ),
    )


def evaluate(rt, name, images, directory, condition):
    ref = reference(rt, name)
    truth = rt.manifest["files"][name]["ef"]
    unit = dict(
        batch=rt.identity,
        images_sha=sha256(directory / "images.npz"),
        truth=truth,
        reference=digest(ref),
    )
    old = committed(directory / "evaluation", unit)
    if old:
        return old
    ef = rt.readout(images, truth, ref["polar"], condition=condition)
    return commit(directory / "evaluation", unit, ef)


def fixed(rt, name, level, seed=42):
    n = rt.manifest["files"][name]["frames"]
    args = {}
    captures = anchors(n)
    # Promote a verified debug prefix only when its exact M schedule matches.
    p = rt.root / "jobs/P0/result.json"
    if level == "M" and seed == 42 and p.exists():
        prefix = read_json(p).get("prefixes", {}).get(name)
        if prefix:
            d = Path(prefix["directory"])
            if sha256(d / "complete.json") != prefix["receipt_sha"]:
                raise ValueError("P0 prefix receipt changed")
            with np.load(d / "images.npz", allow_pickle=False) as z:
                imgs, masks = z["images"].copy(), z["masks"].copy()
            rows = read_json(d / "rows.json")
            state = load_tree(sorted((d / "chunks").glob("*/chunk.npz"))[-1])["state"]
            args = dict(initial=state, prefix=(imgs, masks, rows))
    value = trajectory(
        rt,
        name,
        seed,
        uniform(n, LEVELS[level]),
        tag=f"fixed_{level}",
        capture_indices=captures,
        **args,
    )
    if args:
        import shutil

        for p in d.glob("*_*.npz"):
            dest = value[4] / p.name
            if not dest.exists():
                shutil.copyfile(p, dest)
        # Prefix anchor files are part of the full M receipt after promotion.
        c = read_json(value[4] / "complete.json")
        for p in value[4].glob("*_*.npz"):
            c["files"][p.name] = sha256(p)
        atomic_json(value[4] / "complete.json", c)
    return value


def p0(rt):
    records = []
    prefixes = {}
    for name in rt.manifest["cohorts"]["development"][:2]:
        n = min(64, rt.manifest["files"][name]["frames"])
        caps = anchors(rt.manifest["files"][name]["frames"])
        baseline = trajectory(
            rt,
            name,
            42,
            uniform(n, LEVELS["M"]),
            frames_limit=n,
            capture_indices=caps,
            tag="debug_M",
        )
        images, masks, rows, _, d = baseline
        at = anchors(n)[1]
        for point in ["before", "middle"]:
            state = load_tree(d / f"{point}_{at:05d}.npz")
            b = trajectory(
                rt,
                name,
                42,
                uniform(n, LEVELS["M"]),
                initial=state,
                prefix=(images[:at], masks[:at], rows[:at]),
                frames_limit=n,
                tag=f"recovery_{point}",
            )
            np.testing.assert_allclose(images, b[0], atol=2e-4, rtol=2e-4)
            np.testing.assert_array_equal(masks, b[1])
            for arow, brow in zip(rows, b[2]):
                for k in ["k1", "k2", "lines1", "lines2"]:
                    assert arow[k] == brow[k]
            end_a = load_tree(sorted((d / "chunks").glob("*/chunk.npz"))[-1])["state"]
            end_b = load_tree(sorted((b[4] / "chunks").glob("*/chunk.npz"))[-1])["state"]
            keys = [
                "seed",
                "next_frame",
                "history",
                "masks",
                "previous",
                "past_images",
                "last_budget",
                "last_image",
                "rng_state",
            ]
            compare_trees({k: end_a[k] for k in keys}, {k: end_b[k] for k in keys})
        low = trajectory(rt, name, 42, uniform(n, LEVELS["L"]), frames_limit=n, tag="debug_L")
        locked = [{k: x[k] for k in ["lines1", "lines2"]} for x in low[2]]
        forced = trajectory(
            rt,
            name,
            42,
            uniform(n, LEVELS["L"]),
            frames_limit=n,
            force_second=True,
            fixed_lines=locked,
            tag="debug_no_new_observations",
        )
        np.testing.assert_array_equal(low[1], forced[1])
        records.append(
            dict(
                case=name,
                recovery_passed=True,
                frames=n,
                extra_dps_mse_difference=forced[3]["quality"]["mse"] - low[3]["quality"]["mse"],
                observed_lines_unchanged=low[3]["total_lines"] == forced[3]["total_lines"],
                scope="compute-only intervention; not a free observation improvement",
            )
        )
        prefixes[name] = dict(directory=str(d), receipt_sha=sha256(d / "complete.json"))
    video_frames = float(np.mean([x["frames"] for x in rt.manifest["files"].values()]))
    warm = float(np.median([x["frame_wall_s"] for x in rows[2:]]))
    count = len(rt.manifest["files"])
    estimates = dict(
        measured_warm_frame_s=warm,
        mean_video_frames=video_frames,
        fixed_inference_minutes=3 * count * video_frames * warm / 60,
        maximum_new_p4_inference_minutes=16 * count * video_frames * warm / 60,
        limitations="Debug timing is hardware specific; excludes all-start readout, loading, state IO, branches and failures; not a guarantee.",
    )
    return dict(
        status="completed",
        records=records,
        prefixes=prefixes,
        production_gpu=not rt.cfg["value_diagnostics"]["functional_fixture"],
        estimates=estimates,
    )


def p1(rt):
    records = []
    common = []
    for name in rt.manifest["cohorts"]["development"]:
        variants = []
        for level in LEVELS:
            images, masks, rows, value, d = fixed(rt, name, level)
            ef = evaluate(rt, name, images, d, f"fixed_{level}")
            ref = reference(rt, name)
            if len(images) >= 64 and not ef["all_starts_v1"]["coverage"]["complete"]:
                raise AssertionError("All-start coverage incomplete")
            records.append(
                dict(
                    value,
                    level=level,
                    trajectory=str(d),
                    ef=ef,
                    reference=ref,
                    first_stage_budget=7,
                    changed_calls_vs_M=level == "L",
                )
            )
            variants.append((images, masks))
        target = read_episode(rt.cfg, rt.manifest, name)
        reference_threshold = next(x for x in records if x["case"] == name and x["level"] == "M")[
            "quality"
        ]["frame_mse_p90"]
        for record in [x for x in records if x["case"] == name]:
            streak = longest = 0
            for error in record["quality"]["frame_mse"]:
                streak = streak + 1 if error > reference_threshold else 0
                longest = max(longest, streak)
            record["quality"]["longest_above_M_p90"] = longest
            record["quality"]["M_p90_threshold"] = reference_threshold
        common.append(
            dict(
                case=name,
                levels=list(LEVELS),
                **common_unobserved(target, [v[0] for v in variants], [v[1] for v in variants]),
            )
        )
    return dict(status="completed", records=records, common_unobserved=common)


def p2(rt):
    records = []
    selections = {}
    backgrounds = {}
    for name in rt.manifest["cohorts"]["development"]:
        directory = rt.root / "cached_sensitivity" / Path(name).stem
        unit = dict(batch=rt.identity, case=name)
        old = committed(directory, unit)
        if old:
            records.extend(old["records"])
            selections[name] = old["selection"]
            backgrounds[name] = old["backgrounds"]
            continue
        cached = {}
        for level in LEVELS:
            a = fixed(rt, name, level)
            cached[level] = a[0]
        target = read_episode(rt.cfg, rt.manifest, name)
        n = len(target)
        truth = rt.manifest["files"][name]["ef"]
        ref = reference(rt, name)["polar"]
        background = {}
        for label, high in [("repair", []), ("damage", [0, 1, 2, 3])]:
            images = hybrid(cached["L"], cached["M"], cached["H"], high)
            background[label] = dict(
                ef=rt.readout(images, truth, ref, condition=f"cached_{label}_background"),
                quality=quality(target, images),
            )
        options = []
        for i, (a, b) in enumerate(blocks(n)[0]):
            for label, base, replace in [
                ("repair", cached["L"], cached["H"]),
                ("damage", cached["H"], cached["L"]),
            ]:
                images = base.copy()
                images[0] = cached["M"][0]
                ta, tb = blocks(n)[1]
                images[ta:tb] = cached["M"][ta:tb]
                images[a:b] = replace[a:b]
                options.append((label, [i], images))
        options.extend(
            (arm, high, hybrid(cached["L"], cached["M"], cached["H"], high))
            for arm, high, _ in candidates(n)
        )
        local = []
        for arm, high, images in options:
            ef = rt.readout(images, truth, ref, condition=f"cached_{arm}_{high}")
            local.append(
                dict(
                    case=name,
                    arm=arm,
                    high=high,
                    ef=ef,
                    quality=quality(target, images),
                    physical_closed_loop=False,
                    budget_claim_allowed=False,
                    ef_gain_from_background=background[arm]["ef"]["all_starts_v1"]["absolute_error"]
                    - ef["all_starts_v1"]["absolute_error"]
                    if arm in background
                    else None,
                )
            )
        selection = {}
        for arm in ["same", "saving"]:
            pool = [x for x in local if x["arm"] == arm]
            winners = {}
            for objective, key in [("EF", "ef"), ("reconstruction", "quality")]:
                chosen = min(
                    pool,
                    key=lambda x: (
                        x["ef"]["all_starts_v1"]["absolute_error"]
                        if key == "ef"
                        else x["quality"]["mse"]
                    ),
                )
                winners[objective] = chosen["high"]
            selection[arm] = winners
        commit(
            directory,
            unit,
            dict(
                records=local,
                selection=selection,
                backgrounds=background,
                scope="posthoc cached-image diagnostics; not a deployment strategy or upper bound",
            ),
        )
        records.extend(local)
        selections[name] = selection
        backgrounds[name] = background
    return dict(status="completed", records=records, selections=selections, backgrounds=backgrounds)


def _observable(snapshot, stage):
    middle = snapshot["middle"]
    index = 0 if stage == "first" else 1
    state = middle[f"state{index}"]
    scores = middle[f"scores{index}"]
    info = middle[f"score_diagnostics{index}"]
    order = greedy_order(scores, 14, middle["lines1"] if index else ())
    extra = order[7:14]
    return np.r_[
        state,
        float(np.sum(scores[extra])),
        float(np.sum(info.get("variance_lines", np.zeros(112))[extra])),
        float(np.sum(info.get("sensitivity_lines", np.zeros(112))[extra])),
        float(info.get("gradient_cancellation_ratio", 0)),
    ].tolist()


def p3(rt):
    records = []
    for name in rt.manifest["cohorts"]["development"]:
        base = fixed(rt, name, "M")
        images, masks, rows, _, d = base
        target = read_episode(rt.cfg, rt.manifest, name)
        base_eval = evaluate(rt, name, images, d, "M_reference")
        for at in anchors(len(images)):
            middle = load_tree(d / f"middle_{at:05d}.npz")
            for stage in ["first", "second"]:
                store = rt.root / "marginal" / Path(name).stem / f"{stage}_{at:05d}"
                unit = dict(
                    batch=rt.identity,
                    case=name,
                    stage=stage,
                    at=at,
                    base_receipt=sha256(d / "complete.json"),
                )
                old = committed(store, unit)
                if old:
                    records.append(old)
                    continue
                state = load_tree(d / f"before_{at:05d}.npz") if stage == "first" else middle
                schedule = uniform(len(images), LEVELS["M"])
                schedule[at] = (14, 7) if stage == "first" else (7, 14)
                branch = trajectory(
                    rt,
                    name,
                    42,
                    schedule,
                    initial=state,
                    prefix=(images[:at], masks[:at], rows[:at]),
                    tag=f"marginal_{stage}_{at}",
                )
                new_eval = evaluate(rt, name, branch[0], branch[4], f"marginal_{stage}_{at}")
                assert branch[3]["total_lines"] - base[3]["total_lines"] == 7
                measurements = {}
                for label, folder in [("A", d), ("B", branch[4])]:
                    end = load_tree(folder / f"after_{at:05d}.npz")
                    particles = np.where(
                        end["final_mask"][None].astype(bool),
                        target[at][None],
                        end["previous"][..., -1],
                    )
                    clips = np.stack([causal_window(middle["past_images"], p) for p in particles])
                    # Separate diagnostic readout, no input gradient or extra DPS.
                    predictions = rt.task_only().request(clips, False)["predictions"]
                    measurements[label] = dict(
                        image_variance=float(np.var(particles, axis=0).mean()),
                        task_predictions=np.asarray(predictions).tolist(),
                        task_std=float(np.std(predictions)),
                    )
                value = dict(
                    case=name,
                    stage=stage,
                    at=at,
                    extra_lines=7,
                    features=_observable(middle, stage),
                    gain=value_labels(target, images, branch[0], at, base_eval, new_eval),
                    trajectory=str(branch[4]),
                    prefix_reused=True,
                    true_suffix_recomputed=True,
                    posterior_measurements=measurements,
                    uncertainty_change=measurements["B"]["image_variance"]
                    - measurements["A"]["image_variance"],
                )
                commit(store, unit, value)
                records.append(value)
    return dict(
        status="completed",
        records=records,
        scope="realized +7 observation effect at fixed states, not conditional expectation",
    )


def gate(rt):
    a = read_json(rt.root / "jobs/P1/result.json")["records"]
    b = read_json(rt.root / "jobs/P2/result.json")["records"]
    c = read_json(rt.root / "jobs/P3/result.json")["records"]
    d = rt.cfg["value_diagnostics"]
    responses = []
    for name in rt.manifest["cohorts"]["development"]:
        values = [x for x in a if x["case"] == name]
        m = next(x for x in values if x["level"] == "M")
        h = next(x for x in values if x["level"] == "H")
        ranges = [x["ef"]["all_starts_v1"]["prediction"] for x in b if x["case"] == name]
        responses.append(
            dict(
                case=name,
                mse_gain=m["quality"]["mse"] - h["quality"]["mse"],
                ef_response=abs(
                    m["ef"]["all_starts_v1"]["prediction"] - h["ef"]["all_starts_v1"]["prediction"]
                ),
                cached_range=max(ranges) - min(ranges),
            )
        )
    passed = any(
        x["mse_gain"] > d["gate_mse_abs"]
        or x["ef_response"] > d["gate_ef_pp"]
        or x["cached_range"] > d["gate_ef_pp"]
        for x in responses
    )
    passed = passed or any(
        x["gain"]["mse_gain"] > d["gate_mse_abs"] or abs(x["gain"]["ef_gain"]) > d["gate_ef_pp"]
        for x in c
    )
    return dict(
        passed=passed,
        responses=responses,
        thresholds={k: d[k] for k in ["gate_mse_abs", "gate_ef_pp"]},
        verdict="permission for bounded diagnostics, not efficacy or calibration",
    )


def p4(rt):
    decision = gate(rt)
    atomic_json(rt.root / "p4_gate.json", decision)
    if not decision["passed"]:
        return dict(
            status="skipped",
            gate=decision,
            reason="No predeclared response signal; no expanded search",
        )
    selections = read_json(rt.root / "jobs/P2/result.json")["selections"]
    records = []
    for name in rt.manifest["cohorts"]["development"]:
        n = rt.manifest["files"][name]["frames"]
        for seed in [42, 31415]:
            specs = [
                ("baseline_M", "baseline", uniform(n, LEVELS["M"])),
                ("baseline_L", "baseline", uniform(n, LEVELS["L"])),
            ]
            for arm, count in [("same", 2), ("saving", 1)]:
                specs.append((f"{arm}_periodic", arm, periodic(n, count)))
                for objective, high in selections[name][arm].items():
                    sequence = mixed(n, high)
                    specs.append((f"{arm}_{objective}", arm, sequence))
                    if seed == 42 and arm == "same":
                        specs.append((f"{arm}_{objective}_reverse", arm, reverse(sequence)))
            seen = {}
            by_arm = {}
            for alias, arm, schedule in specs:
                key = digest(schedule.tolist())
                if key in seen:
                    seen[key]["aliases"].append(alias)
                    continue
                if alias == "baseline_M":
                    value = fixed(rt, name, "M", seed)
                elif alias == "baseline_L":
                    value = fixed(rt, name, "L", seed)
                else:
                    value = trajectory(rt, name, seed, schedule, tag=alias)
                ef = evaluate(rt, name, value[0], value[4], alias)
                item = dict(
                    case=name,
                    seed=seed,
                    arm=arm,
                    aliases=[alias],
                    schedule_hash=key,
                    total_lines=value[3]["total_lines"],
                    mean_lines=value[3]["mean_lines"],
                    quality=value[3]["quality"],
                    ef=ef,
                    perception_calls=value[3]["perception_calls"],
                    inference_seconds=value[3]["inference_seconds"],
                    selection_seed=42,
                    trajectory=str(value[4]),
                    posthoc_diagnostic=True,
                )
                seen[key] = item
                if arm != "baseline":
                    histogram = Counter(map(tuple, schedule[1:]))
                    expected = Counter(map(tuple, periodic(n, 2 if arm == "same" else 1)[1:]))
                    if histogram != expected:
                        raise AssertionError("Candidate/control joint pair counts changed")
                    by_arm.setdefault(arm, []).append(item)
            for item in seen.values():
                records.append(item)
            # Includes baselines; never exceed the predeclared per-video list.
            cap = 10 if seed == 42 else 8
            if len(seen) > cap:
                raise AssertionError("Candidate list expanded beyond locked cap")
    return dict(
        status="completed",
        gate=decision,
        records=records,
        scope="posthoc bounded physical schedules, seed31415 never reselects windows",
    )


def p5(rt):
    rows = read_json(rt.root / "jobs/P3/result.json")["records"]
    return dict(
        status="completed",
        predictability=predictability(rows),
        budget_controller_trained=False,
        scope="small observable-state cross-validation; no online efficacy certification",
    )


def worker(phase, cfg, manifest, root):
    runtime = Runtime(cfg, manifest, root, phase)
    start = time.perf_counter()
    try:
        value = {"P0": p0, "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5}[phase](runtime)
        value.update(
            phase=phase,
            process_wall_s=time.perf_counter() - start,
            functional_fixture=cfg["value_diagnostics"]["functional_fixture"],
            new_ef_clips=runtime.cache_calls,
            reused_ef_clips=runtime.cache_hits,
        )
        atomic_json(Path(root) / "jobs" / phase / "result.json", value)
        return value
    finally:
        runtime.close()
