import csv

import pytest

from cognitive_ultrasound.evaluation.report import generate_report
from cognitive_ultrasound.provenance import write_json


def test_report_patient_weighting_and_partial_status(tmp_path):
    cfg = {
        "methods": ["casl"],
        "budgets": [7],
        "metrics": ["psnr"],
        "frames": 100,
        "split": "test",
        "profile": False,
        "segmentation": False,
        "synthetic_input": True,
    }
    write_json(
        tmp_path / "manifest.json",
        {
            "status": "running",
            "identity": {"config": cfg, "cases": ["a.hdf5", "b.hdf5", "missing.hdf5"]},
        },
    )
    for name, scores in (("a", [10.0, 10.0]), ("b", [30.0, 30.0, 30.0, 30.0])):
        folder = tmp_path / "casl/lines_007" / name
        folder.mkdir(parents=True)
        write_json(
            folder / "complete.json", {"budget_mismatch_frames": 0, "segmentation_excluded": None}
        )
        with open(folder / "frames.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["psnr", "total_s"])
            writer.writeheader()
            for score in scores:
                writer.writerow({"psnr": score, "total_s": 0.1})
    report = generate_report(tmp_path)
    with open(tmp_path / "summary.csv") as f:
        row = next(csv.DictReader(f))
    assert float(row["psnr"]) == pytest.approx(20.0)  # NOT pooled-frame mean of 23.33.
    assert "2/3" in report.read_text(encoding="utf-8")
    assert "SYNTHETIC" in report.read_text(encoding="utf-8")
    assert (tmp_path / "budget_quality.png").exists()


def test_strict_json_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"loss": float("nan")})
