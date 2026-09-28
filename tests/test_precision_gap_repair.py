import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("repair", Path(__file__).parents[1] / "scripts/repair_compute_lab_precision_gaps.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    return dict(kind="inference", short=True, cohort="debug", budget=14,
                profile=dict(backend="torch", mode="graph", precision="fp16", steps=50, dps=50)), dict(
                    status="failed", failure_kind="numerical_correctness_failure", error="AssertionError: Tensor-likes are not close!")


def test_fixed_failure_is_eligible():
    task, result = inputs()
    module.validate_task("short_torch_50_fp16_b14", task, result)
    assert len(module.JOBS) == 4


@pytest.mark.parametrize("change", ["completed", "cohort", "short", "profile", "failure"])
def test_no_completed_job_cohort_expansion_or_precision_fallback(change):
    task, result = inputs()
    if change == "completed": result["status"] = "completed"
    elif change == "cohort": task["cohort"] = "confirmation"
    elif change == "short": task["short"] = False
    elif change == "profile": task["profile"]["precision"] = "fp32"
    else: result["error"] = "CUDA out of memory"
    with pytest.raises(ValueError):
        module.validate_task("short_torch_50_fp16_b14", task, result)
