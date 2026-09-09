import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import griddata

from cognitive_ultrasound.config import ROOT
from cognitive_ultrasound.polar import cached_cartesian_to_polar_matrix, geometry


@pytest.fixture(scope="module")
def reference():
    # Load the actual pinned geometry functions without importing the full ML runtime.
    tree = ast.parse((ROOT / "vendor/casl/zea/zea/data/convert/echonet.py").read_text())
    selected = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"rotate_coordinates", "cartesian_to_polar_matrix"}
    ]
    scope = {"np": np, "griddata": griddata}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "official_geometry", "exec"), scope)
    return scope["cartesian_to_polar_matrix"]


@pytest.mark.parametrize("mode", ["zeros", "ones", "random", "impulse"])
def test_cached_cubic_matches_upstream_exactly(reference, mode):
    frame = np.zeros((112, 112))
    if mode == "ones":
        frame[:] = 1
    elif mode == "random":
        frame[:] = np.random.default_rng(42).uniform(size=frame.shape)
    elif mode == "impulse":
        frame[53, 70] = 1
    actual = cached_cartesian_to_polar_matrix(frame, interpolation="cubic")
    expected = reference(frame, interpolation="cubic")
    np.testing.assert_array_equal(actual, expected)
    assert geometry(112, 112, (61, 7), 107, 0.79) is geometry(112, 112, (61, 7), 107, 0.79)


def test_local_stop_refuses_other_projects():
    import runpy

    verify = runpy.run_path(str(ROOT / "scripts/stop_local_conversion.py"))["verify_target"]
    raw, output = Path("G:/raw"), Path("G:/polar")
    command = [
        "python",
        str(ROOT / "scripts/local_conversion.py"),
        "--raw",
        str(raw),
        "--output",
        str(output),
    ]
    verify(command, ROOT, raw, output)
    with pytest.raises(RuntimeError, match="unrelated"):
        verify(["python", "another_project.py"], ROOT, raw, output)
    with pytest.raises(RuntimeError, match="different paths"):
        verify(command, ROOT, Path("G:/other"), output)
