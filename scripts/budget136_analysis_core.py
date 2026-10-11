"""CPU-only, cache-only diagnostics. No sampler, task service or optimizer is imported."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import psutil

ARMS = ("T1", "Q2", "T2")
CHECKPOINTS = (1, 16, 72, 136)
LEVELS = {
    "T1": np.array([4, 7, 10, 14]),
    "Q2": np.array([0, 2, 4, 7, 14]),
    "T2": np.array([0, 2, 4, 7, 14]),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def tree_load(path):
    """Read the existing pickle-free state format without importing model libraries."""
    with np.load(path, allow_pickle=False) as z:

        def decode(x):
            if isinstance(x, dict):
                if set(x) == {"__array__"}:
                    return z[x["__array__"]].copy()
                return {k: decode(v) for k, v in x.items()}
            return [decode(v) for v in x] if isinstance(x, list) else x

        return decode(json.loads(z["metadata"].tobytes()))


def tree_hash(tree):
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                h.update(key.encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(str(len(x)).encode())
            for v in x:
                visit(v)
        else:
            h.update(json.dumps(x, allow_nan=False, sort_keys=True).encode())

    visit(tree)
    return h.hexdigest()


def softmax(x, legal=None):
    x = np.asarray(x, np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite replay logits")
    if legal is not None:
        legal = np.asarray(legal, bool)
        if legal.shape != x.shape or not legal.any(-1).all():
            raise ValueError("Invalid legal mask")
        x = np.where(legal, x, np.float32(-1e9))
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def forward(params, sequence, valid):
    """Exactly the declared GRU formula; CPU float32, checked against saved probabilities."""
    x = np.asarray(sequence, np.float32)
    v = np.asarray(valid, bool)
    if x.ndim != 3 or v.shape != x.shape[:2] or x.shape[2] != 18 or not np.isfinite(x).all():
        raise ValueError("Invalid replay input")
    p = {k: np.asarray(a, np.float32) for k, a in params.items()}
    hsize = p["wh"].shape[0]
    h = np.zeros((len(x), hsize), np.float32)

    def sigmoid(a):
        return 1 / (1 + np.exp(-np.clip(a, -80, 80)))

    for i in range(x.shape[1]):
        xx = x[:, i] @ p["wx"] + p["b"]
        hh = h @ p["wh"]
        z = sigmoid(xx[:, :hsize] + hh[:, :hsize])
        r = sigmoid(xx[:, hsize : 2 * hsize] + hh[:, hsize : 2 * hsize])
        candidate = np.tanh(xx[:, 2 * hsize :] + (r * h) @ p["wh"][:, 2 * hsize :])
        h = np.where(v[:, i, None], z * h + (1 - z) * candidate, h)
    return h @ p["wo"] + p["bo"], h


def entropy(p):
    return -(p * np.log(np.maximum(p, 1e-20))).sum(-1)


def js(a, b):
    m = (a + b) / 2
    return 0.5 * (a * np.log(np.maximum(a, 1e-20) / np.maximum(m, 1e-20))).sum(-1) + 0.5 * (
        b * np.log(np.maximum(b, 1e-20) / np.maximum(m, 1e-20))
    ).sum(-1)


def weighted_pca(hidden, case_ids):
    h = np.asarray(hidden, np.float64)
    cases = np.asarray(case_ids)
    unique = np.unique(cases)
    weights = np.array([1 / (len(unique) * (cases == c).sum()) for c in cases])
    center = (h * weights[:, None]).sum(0)
    cov = (h - center).T @ ((h - center) * weights[:, None])
    eig, vec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]
    eig = np.maximum(eig[order], 0)
    vec = vec[:, order]
    total = eig.sum()
    return (
        (h - center) @ vec[:, :2],
        eig[:2] / total if total > 0 else np.zeros(2),
        center,
        vec[:, :2],
    )


def rank_correlation(a, b):
    def ranks(x):
        _, inverse, count = np.unique(x, return_inverse=True, return_counts=True)
        return (np.cumsum(count) - count + (count - 1) / 2)[inverse]

    x, y = ranks(a), ranks(b)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None


class Source:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}
        self.scope = self.json(self.root / "scope_closure_136.json")
        if self.scope.get("status") != "completed" or self.scope["target_updates"] != 136:
            raise ValueError("Offline analysis requires the validated136 closure")
        state = self.json(self.root / "status.json")
        for pid_key, time_key in [("pid", "created"), ("worker_pid", "worker_created")]:
            if state.get(pid_key) and state.get(time_key):
                try:
                    p = psutil.Process(state[pid_key])
                    if (
                        p.is_running()
                        and p.status() != psutil.STATUS_ZOMBIE
                        and abs(p.create_time() - state[time_key]) < 0.05
                    ):
                        raise ValueError("Training still active; offline analysis cannot start")
                except psutil.NoSuchProcess:
                    pass
        receipt = self.json(self.root / "scope_136_bundle_receipt.json")
        if not receipt["verified"] or sha(receipt["archive"]) != receipt["sha256"]:
            raise ValueError("Closure archive is not verified")
        self.config = self.json(self.root / "config.json")
        self.manifest = self.json(self.root / "manifest.json")
        self.identity = self.json(self.root / "identity.json")
        repo = Path(__file__).resolve().parents[1]
        for name, value in self.identity.get("source", {}).items():
            if sha(repo / name) != value:
                raise ValueError("Recorded model source differs from the CPU replay contract")
        self.case_names = self.manifest["cohorts"]["development"]
        self.params = {}
        self.training = {}
        self.cases = {}
        for arm in ARMS:
            self.params[arm] = {}
            for update in CHECKPOINTS:
                d = self.root / "training" / arm / "updates" / f"{update:05d}"
                self.validate(d)
                self.params[arm][update] = tree_load(self.track(d / "state.npz"))["params"]
            self.training[arm] = []
            for update in range(1, 137):
                d = self.root / "training" / arm / "updates" / f"{update:05d}"
                self.validate(d)
                self.training[arm].append(self.json(d / "record.json"))
        for phase in ["pilot_evaluate", "full_pass_1_evaluate", "full_pass_2_evaluate"]:
            d = self.root / "jobs" / phase
            self.validate(d)
            self.cases[phase] = self.json(d / "result.json")["cases"]
        rows = self.cases["full_pass_2_evaluate"]
        if len(rows) != 3 * len(self.case_names) or {(r["arm"], r["case"]) for r in rows} != {
            (a, n) for a in ARMS for n in self.case_names
        }:
            raise ValueError("136 development coverage changed")
        if any(r["update"] != 136 or r["seed"] != 42 or r["cohort"] != "development" for r in rows):
            raise ValueError("Wrong136 evaluation protocol")

    def track(self, p):
        p = Path(p).resolve()
        self.hashes[str(p)] = sha(p)
        return p

    def json(self, p):
        return read(self.track(p))

    def inside(self, p):
        p = Path(p).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Cache path leaves the frozen source directory")
        return p

    def validate(self, directory):
        d = self.inside(directory)
        c = self.json(d / "complete.json")
        # Match the project's JSON digest without importing model code.
        payload_hash = hashlib.sha256(json.dumps(c["result"], sort_keys=True).encode()).hexdigest()
        if payload_hash != c["result_digest"]:
            raise ValueError("Receipt payload changed")
        for name, value in c["files"].items():
            p = self.inside(d / name)
            if sha(self.track(p)) != value:
                raise ValueError("Committed cache was modified")
        return c["result"]

    def traces(self, arm):
        rows = {r["case"]: r for r in self.cases["full_pass_2_evaluate"] if r["arm"] == arm}
        result = []
        for name in self.case_names:
            row = rows[name]
            d = self.inside(row["trajectory"])
            self.validate(d)
            trace = tree_load(self.track(d / "controller.npz"))
            if trace["identity"]["arm"] != arm or trace["identity"]["parameters"] != tree_hash(
                self.params[arm][136]
            ):
                raise ValueError("Trajectory was produced by a different policy checkpoint")
            if len(trace["trace"]) != self.manifest["files"][name]["frames"] - 1:
                raise ValueError("Missing policy frames")
            result.append((name, row, d, trace["trace"]))
        return result


def analyze_models(source, out):
    summaries = {}
    frame_rows = []
    parameter_rows = []
    representation_rows = []
    sensitivity_rows = []
    pcas = {}
    cue_rows = []
    for arm in ARMS:
        traces = source.traces(arm)
        x = np.stack([t["sequence"] for _, _, _, ts in traces for t in ts])
        valid = np.stack([t["valid"] for _, _, _, ts in traces for t in ts])
        legal = np.stack([t["legal"] for _, _, _, ts in traces for t in ts])
        actions = np.array([t["action"] for _, _, _, ts in traces for t in ts])
        case_ids = np.array([name for name, _, _, ts in traces for _ in ts])
        frame_ids = np.array([t["frame"] for _, _, _, ts in traces for t in ts])
        forced = np.array([t["quota_forced"] for _, _, _, ts in traces for t in ts])
        constrained = np.array([t["quota_constrained"] for _, _, _, ts in traces for t in ts])
        values = LEVELS[arm]
        summary = {"checkpoints": {}, "cases": [], "early_reference_update": 1}
        per_checkpoint = {}
        for update in CHECKPOINTS:
            p = source.params[arm][update]
            logits, h = forward(p, x, valid)
            raw = softmax(logits)
            masked = softmax(logits, legal)
            bias = softmax(np.broadcast_to(p["bo"], logits.shape))
            expected = raw @ values
            per_checkpoint[update] = (raw, masked, expected, h)
            metrics = []
            for name, _, _, ts in traces:
                take = case_ids == name
                metrics.append(
                    dict(
                        case=name,
                        update=update,
                        arm=arm,
                        expected_unmasked=float(expected[take].mean()),
                        expected_unmasked_sd=float(expected[take].std()),
                        raw_entropy=float(entropy(raw[take]).mean()),
                        masked_entropy=float(entropy(masked[take]).mean()),
                        state_vs_bias_js=float(js(raw[take], bias[take]).mean()),
                        within_video_probability_js=float(
                            js(
                                raw[take], np.broadcast_to(raw[take].mean(0), raw[take].shape)
                            ).mean()
                        ),
                        masked_vs_unmasked_js=float(js(raw[take], masked[take]).mean()),
                        forced_fraction=float(forced[take].mean()),
                        constrained_fraction=float(constrained[take].mean()),
                        preferred_action_forbidden=float(
                            (~legal[take][np.arange(take.sum()), raw[take].argmax(1)]).mean()
                        ),
                    )
                )
            summary["checkpoints"][str(update)] = {
                k: float(np.mean([r[k] for r in metrics]))
                for k in metrics[0]
                if k not in ["case", "arm", "update"]
            }
            summary["cases"].extend(metrics)
            for layer, a in p.items():
                base = source.params[arm][1][layer]
                delta = float(np.linalg.norm(a - base))
                norm = float(np.linalg.norm(base))
                parameter_rows.append(
                    dict(
                        arm=arm,
                        update=update,
                        layer=layer,
                        norm=float(np.linalg.norm(a)),
                        delta_from_update1=delta,
                        relative_delta=delta / norm if norm > 0 else None,
                    )
                )
        raw, masked, expected, h = per_checkpoint[136]
        saved_max = np.array([t["max_probability"] for _, _, _, ts in traces for t in ts])
        saved_entropy = np.array([t["entropy"] for _, _, _, ts in traces for t in ts])
        if not np.allclose(masked.max(1), saved_max, atol=2e-4, rtol=2e-4) or not np.allclose(
            entropy(masked), saved_entropy, atol=2e-4, rtol=2e-4
        ):
            raise ValueError("CPU replay differs from the saved actual policy probabilities")
        summary["replay_verified_atol_rtol"] = 2e-4
        latest_only = valid.copy()
        latest_only[:, :-1] = False
        without_resources = x.copy()
        without_resources[:, :, [10, 11, 16, 17]] = 0
        resource_only = np.zeros_like(x)
        resource_only[:, :, [10, 11, 16, 17]] = x[:, :, [10, 11, 16, 17]]
        p = source.params[arm][136]
        for tag, xx, vv in [
            ("latest_only", x, latest_only),
            ("without_resource_clock", without_resources, valid),
            ("resource_clock_only", resource_only, valid),
        ]:
            prob = softmax(forward(p, xx, vv)[0])
            diff = js(raw, prob)
            summary[tag] = {
                "js_case_mean": float(
                    np.mean([diff[case_ids == name].mean() for name in source.case_names])
                ),
                "expected_budget_abs_change": float(
                    np.mean(
                        [
                            np.abs((prob @ values - expected)[case_ids == name]).mean()
                            for name in source.case_names
                        ]
                    )
                ),
                "scope": "counterfactual input replay; not a physical closed-loop task ablation",
            }
        projected, ratios, center, axes = weighted_pca(h, case_ids)
        pcas[arm] = (projected, case_ids, frame_ids, x[:, -1, 17], x[:, -1, 16])
        summary["pca_variance_ratio"] = ratios.tolist()
        for name in source.case_names:
            take = case_ids == name
            for feature in [0, 6, 7, 8, 9, 12, 13, 16, 17]:
                cue_rows.append(
                    dict(
                        arm=arm,
                        case=name,
                        feature=feature,
                        expected_budget_spearman=rank_correlation(
                            expected[take], x[take, -1, feature]
                        ),
                        pc1_spearman=rank_correlation(projected[take, 0], x[take, -1, feature]),
                        pc2_spearman=rank_correlation(projected[take, 1], x[take, -1, feature]),
                    )
                )
        summary["hidden_saturated_fraction_case_mean"] = float(
            np.mean([np.mean(np.abs(h[case_ids == name]) > 0.95) for name in source.case_names])
        )
        for i in range(len(x)):
            frame_rows.append(
                dict(
                    arm=arm,
                    case=case_ids[i],
                    frame=int(frame_ids[i]),
                    chosen_lines=int(values[actions[i]]),
                    expected_unmasked=float(expected[i]),
                    expected_masked=float(masked[i] @ values),
                    raw_entropy=float(entropy(raw[i : i + 1])[0]),
                    masked_entropy=float(entropy(masked[i : i + 1])[0]),
                    forced=bool(forced[i]),
                    constrained=bool(constrained[i]),
                    unmasked_preference=int(values[raw[i].argmax()]),
                    resource_feature=float(x[i, -1, 16]),
                    clock_feature=float(x[i, -1, 17]),
                )
            )
            representation_rows.append(
                dict(
                    arm=arm,
                    case=case_ids[i],
                    frame=int(frame_ids[i]),
                    pc1=float(projected[i, 0]),
                    pc2=float(projected[i, 1]),
                )
            )
        # Fixed six warm frames per video; only the current feature is perturbed.
        selected = np.concatenate(
            [
                np.flatnonzero(case_ids == name)[
                    np.linspace(0, (case_ids == name).sum() - 1, 6, dtype=int)
                ]
                for name in source.case_names
            ]
        )
        xx = x[selected]
        vv = valid[selected]
        delta = 0.001
        jac = []
        for f in range(18):
            plus = xx.copy()
            minus = xx.copy()
            plus[:, -1, f] += delta
            minus[:, -1, f] -= delta
            jac.append(
                (
                    softmax(forward(p, plus, vv)[0]) @ values
                    - softmax(forward(p, minus, vv)[0]) @ values
                )
                / (2 * delta)
            )
        jac = np.stack(jac, axis=1)
        weights = np.array(
            [1 / (len(source.case_names) * (case_ids == name).sum()) for name in case_ids]
        )
        mean = (x[:, -1] * weights[:, None]).sum(0)
        scale = np.sqrt(((x[:, -1] - mean) ** 2 * weights[:, None]).sum(0))
        sensitivity = (np.abs(jac) * scale).mean(0)
        summary["local_sensitivity_sd_scaled"] = sensitivity.tolist()
        for f in range(18):
            sensitivity_rows.append(
                dict(
                    arm=arm,
                    feature=f,
                    mean_abs_local_derivative=float(np.abs(jac[:, f]).mean()),
                    descriptive_dev_feature_sd=float(scale[f]),
                    sd_scaled_sensitivity=float(sensitivity[f]),
                )
            )
        summaries[arm] = summary
    write_csv(out / "policy_frames.csv", frame_rows)
    write_csv(out / "weight_changes.csv", parameter_rows)
    write_csv(out / "representation_pca.csv", representation_rows)
    write_csv(out / "local_sensitivity.csv", sensitivity_rows)
    write_csv(out / "representation_cue_correlations.csv", cue_rows)
    return summaries, pcas


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to export")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def samples(source, out):
    names = source.case_names[:3]
    samples = {}
    gaps = []
    sample_stats = []
    maps = {
        a: {r["case"]: r for r in source.cases["full_pass_2_evaluate"] if r["arm"] == a}
        for a in ARMS
    }
    for name in names:
        item = source.manifest["files"][name]
        n = item["frames"]
        q = (n - 1) // 4
        # Equivalent to value_protocol.anchors: centers of [1+iq,1+(i+1)q).
        indexes = np.array([(1 + i * q + 1 + (i + 1) * q - 1) // 2 for i in range(4)], dtype=int)
        target_file = Path(source.config["data_root"]) / item["relative"]
        if sha(source.track(target_file)) != item["sha256"]:
            raise ValueError("Selected HDF5 input changed")
        with h5py.File(target_file) as h:
            target = h["data/image"][indexes].astype(np.float32) / 30 + 1
        data = dict(indexes=indexes, target=target, models={})
        paths = {
            "M": maps["T1"][name]["reference"]["trajectory"],
            **{a: maps[a][name]["trajectory"] for a in ARMS},
        }
        for arm, path in paths.items():
            d = source.inside(path)
            source.validate(d)
            with np.load(source.track(d / "images.npz"), allow_pickle=False) as z:
                images = z["images"][indexes].copy()
                masks = z["masks"][indexes].copy()
            rows = source.json(d / "rows.json")
            variance = []
            for at in indexes:
                p = d / f"after_{at:05d}.npz"
                if p.exists():
                    state = tree_load(source.track(p))
                    variance.append(np.var(state["previous"][..., -1], axis=0))
                else:
                    variance.append(np.full((112, 112), np.nan))
                    gaps.append(f"missing particle snapshot: {name}/{arm}/{at}")
            data["models"][arm] = dict(
                images=images,
                masks=masks,
                variance=np.stack(variance),
                rows=[rows[i] for i in indexes],
            )
            for ri, at in enumerate(indexes):
                unknown = ~masks[ri].astype(bool)
                vv = variance[ri]
                sample_stats.append(
                    dict(
                        case=name,
                        arm=arm,
                        frame=int(at),
                        unobserved_mse=float(
                            np.mean((images[ri][unknown] - target[ri][unknown]) ** 2)
                        )
                        if unknown.any()
                        else None,
                        unobserved_particle_variance=float(np.mean(vv[unknown]))
                        if unknown.any() and np.isfinite(vv).all()
                        else None,
                        displayed_out_of_range_fraction=float(
                            ((images[ri] < -1) | (images[ri] > 1)).mean()
                        ),
                    )
                )
        samples[name] = data
    arrays = {}
    for ci, (name, data) in enumerate(samples.items()):
        arrays[f"case{ci}_target"] = data["target"]
        arrays[f"case{ci}_indexes"] = data["indexes"]
        for arm, model in data["models"].items():
            for key in ["images", "masks", "variance"]:
                arrays[f"case{ci}_{arm}_{key}"] = model[key]
    np.savez_compressed(out / "selected_samples.npz", **arrays)
    write_csv(out / "sample_uncertainty_error.csv", sample_stats)
    return samples, gaps
