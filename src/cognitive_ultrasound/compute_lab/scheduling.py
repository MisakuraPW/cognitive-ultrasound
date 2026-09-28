"""Finite cost decisions, distinct from scientific quality and numerical verdicts."""

from statistics import mean

from ..preparation.common import atomic_json, read_json


def failure_kind(error):
    text = str(error).lower()
    if any(x in text for x in ("assertionerror", "nonfinite", "not close", "graph validation")):
        return "numerical_correctness_failure"
    if any(
        x in text
        for x in ("modulenotfound", "filenotfound", "no module", "gpu unavailable", "unsupported")
    ):
        return "dependency_or_asset_missing"
    return "implementation_failure"


def steady(record):
    rows = record.get("micro") or [r for r in record.get("rows", []) if r["frame"] >= 2]
    return mean([r["closed_loop_s"] for r in rows]) if rows else None


def cost_decision(candidate, reference, limit=4.0):
    """A bounded relative investment screen, never a quality/equivalence assertion."""
    if candidate.get("status") != "completed":
        return dict(proceed=False, kind=candidate.get("failure_kind", "implementation_failure"))
    a, b = steady(candidate), steady(reference)
    if a is None or b is None or b <= 0:
        return dict(proceed=False, kind="missing_timing_evidence")
    ratio = a / b
    return dict(
        proceed=ratio <= limit,
        kind="cost_eligible" if ratio <= limit else "performance_not_worth_continuing",
        measured_ratio=ratio,
        maximum_ratio=limit,
        basis="synchronized closed-loop fixed-input short benchmark",
        scientific_parameters_changed=False,
    )


def record_decision(root, name, value):
    file = root / "scheduling.json"
    records = read_json(file) if file.exists() else {}
    records[name] = value
    atomic_json(file, records)
    print(f"DECISION {name}: {value}", flush=True)


def fastest_torch(records):
    """Never extend eager or a mode slower than the measured official DEV rollout."""
    official = next(x for x in records if x["profile"]["name"] == "official")
    eligible = [
        x
        for x in records
        if x["profile"]["backend"] == "torch"
        and x["profile"]["mode"] in ("compile", "graph")
        and x.get("status") == "completed"
        and x.get("internal_correctness", False)
        and x["closed_loop_s"] < official["closed_loop_s"]
    ]
    return min(eligible, key=lambda x: x["closed_loop_s"])["profile"]["mode"] if eligible else None
