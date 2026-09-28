"""Frozen candidates and patient-level decisions. No automatic approximation adoption."""

from dataclasses import asdict, dataclass

import numpy as np

from ..preparation.common import digest


@dataclass(frozen=True)
class Profile:
    name: str
    backend: str = "jax"
    mode: str = "official"
    precision: str = "fp32"
    steps: int = 50
    dps: int = 50
    io: str = "serial"
    constants: bool = False
    transfers: bool = False
    async_output: bool = False

    @property
    def category(self):
        return "B" if self.precision != "fp32" or self.steps != 50 or self.dps != 50 else "A"

    @property
    def family(self):
        precision = self.precision != "fp32"
        sampling = self.steps != 50 or self.dps != 50
        return (
            "combination"
            if precision and sampling
            else ("precision" if precision else "sampling" if sampling else "engineering")
        )

    def record(self):
        return dict(
            **asdict(self),
            category=self.category,
            family=self.family,
            cold_steps=500,
            cold_precision=self.precision,
            particles=2,
            window=3,
            tf32=False,
            automatic_adoption=False,
        )

    def variant(self):
        v = dict(
            initial_step=500 - self.steps,
            precision={"fp32": "float32", "fp16": "mixed_float16", "bf16": "mixed_bfloat16"}[
                self.precision
            ],
        )
        if self.dps != self.steps:
            v["guidance_steps"] = self.dps
        return v


def profiles(torch_mode=None, combined=()):
    result = [
        Profile("official"),
        Profile("jax_prefetch", io="prefetch"),
        Profile("jax_constants", constants=True),
        Profile("jax_transfers", transfers=True),
        Profile("jax_async_output", async_output=True),
    ]
    result += [
        Profile("torch_" + m, backend="torch", mode=m) for m in ("eager", "compile", "graph")
    ]
    for steps in (50, 25):
        for precision in ("fp32", "fp16", "bf16"):
            if steps == 50 and precision == "fp32":
                continue
            result.append(
                Profile(f"jax_{steps}_{precision}", precision=precision, steps=steps, dps=steps)
            )
    result += [Profile(f"jax_dps{d}", dps=d) for d in (10, 5)]
    if torch_mode:
        result += [
            Profile(f"torch_{s}_{p}", backend="torch", mode=torch_mode, steps=s, dps=s, precision=p)
            for s in (50, 25)
            for p in ("fp16", "bf16")
        ]
    if len(combined) >= 2:
        result.append(
            Profile(
                "jax_combined",
                io="prefetch" if "jax_prefetch" in combined else "serial",
                constants="jax_constants" in combined,
                transfers="jax_transfers" in combined,
                async_output="jax_async_output" in combined,
            )
        )
    return {p.name: p for p in result}


def numeric(a, b, discrete=False):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False, bitwise=False, max_abs=None, reason="shape/nonfinite")
    equal = bool(np.array_equal(a, b))
    exact = a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return dict(
        passed=equal if discrete else bool(np.allclose(a, b, atol=2e-4, rtol=2e-4)),
        bitwise=exact,
        max_abs=float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0)),
    )


def aggregate(rows):
    """Equal patient weights; equal seeds within patient; equal frames within seed."""
    groups = {}
    for r in rows:
        groups.setdefault((r["case"], r["seed"]), []).append(r)
    fields = ("psnr", "ssim", "mae")
    patients = {}
    for (case, _), group in groups.items():
        patients.setdefault(case, []).append([np.mean([r[k] for r in group]) for k in fields])
    return {k: np.mean(v, axis=0) for k, v in sorted(patients.items())}


def quality(reference, candidate, confirmation=False, repetitions=10000):
    def keys(rows):
        return {(r["case"], r["seed"], r["frame"]) for r in rows}

    if not reference or keys(reference) != keys(candidate):
        return dict(passed=False, status="incomplete_pairs")
    if len(keys(reference)) != len(reference) or len(keys(candidate)) != len(candidate):
        raise ValueError("Duplicate paired observations")
    ref, cand = aggregate(reference), aggregate(candidate)
    a, b = np.array(list(ref.values())), np.array([cand[k] for k in ref])
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a[:, 2] <= 0):
        return dict(passed=False, status="nonfinite_or_zero_reference_mae")
    losses = a - b
    losses[:, 2] = b[:, 2] / a[:, 2] - 1
    limits, worst_limits = np.array([0.1, 0.002, 0.01]), np.array([0.5, 0.01, 0.05])
    mean, worst = losses.mean(0), losses.max(0)
    upper = None
    if confirmation:
        rng = np.random.default_rng(20260928)
        bootstrap = losses[rng.integers(len(losses), size=(repetitions, len(losses)))].mean(1)
        upper = np.quantile(bootstrap, 0.95, axis=0)
    passed = bool(
        np.all(mean <= limits + 1e-12)
        and np.all(worst <= worst_limits + 1e-12)
        and (not confirmation or (len(a) >= 32 and np.all(upper <= limits + 1e-12)))
    )
    return dict(
        passed=passed,
        status="passed" if passed else "insufficient_evidence",
        cases=list(ref),
        patient_loss=losses.tolist(),
        mean_loss=mean.tolist(),
        worst_loss=worst.tolist(),
        upper95=None if upper is None else upper.tolist(),
        columns=["psnr_drop_db", "ssim_drop", "mae_relative_increase"],
        replication_unit="patient; seeds and frames aggregated before bootstrap",
    )


def select(records):
    """At most four DEV selections. Confirmation never enters this function."""
    selected = []
    for family in ("engineering", "precision", "sampling", "combination"):
        available = [
            r
            for r in records
            if r["cohort"] == "development"
            and r["profile"]["family"] == family
            and r["quality"]["passed"]
            and r["profile"]["name"] != "official"
            and (family != "engineering" or r["equivalence_passed"])
        ]
        if available:
            winner = min(available, key=lambda r: (r["closed_loop_s"], r["profile"]["name"]))
            if winner["profile"]["name"] not in selected:
                selected.append(winner["profile"]["name"])
    return selected


def science_identity(cfg, manifest):
    return digest(
        dict(
            checkpoint=cfg["checkpoint_sha256"],
            checkpoint_files=cfg.get("checkpoint_files"),
            manifest=manifest,
            particles=2,
            window=3,
            cold=500,
            warm=50,
            dps=50,
            tf32=False,
            metric="polar_uint8_psnr_ssim_float_mae_v1",
        )
    )
