import numpy as np
import pytest

from cognitive_ultrasound.preparation.followup import assert_new_holdout, interpolate_observed


def test_followup_never_reuses_old_validation_as_new_holdout():
    previous = {
        "cohorts": {"train": ["t"], "debug": ["a"], "development": ["b"], "confirmation": ["c"]}
    }
    for name in ("a", "b", "c"):
        with pytest.raises(ValueError, match="overlaps"):
            assert_new_holdout(previous, {"cohorts": {"confirmation": [name]}})
    assert_new_holdout(previous, {"cohorts": {"confirmation": ["new"]}})


def test_interpolation_uses_acquired_columns_only():
    lines = np.array([0, 55, 111])
    observed = np.full((112, 112, 1), np.nan, np.float32)
    observed[:, lines, 0] = np.array([-1, 0, 1])
    first = interpolate_observed(observed, lines)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first[:, lines], observed[:, lines])
    hidden = np.ones(112, bool)
    hidden[lines] = False
    observed[:, hidden] = 999
    np.testing.assert_array_equal(interpolate_observed(observed, lines), first)
