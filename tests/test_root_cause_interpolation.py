"""The diagnostic replacement must preserve the half-pixel operator and its VJP."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "root_cause", Path(__file__).parents[1] / "scripts/profile_torch_root_cause.py"
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.mark.parametrize("shape", [(2, 3, 1, 1), (2, 3, 3, 5), (2, 64, 56, 56)])
def test_forward_and_input_vjp_including_edges(shape):
    rng = torch.Generator().manual_seed(31415)
    x = torch.randn(shape, dtype=torch.float64, generator=rng, requires_grad=True)
    original = torch.nn.functional.interpolate(
        x, scale_factor=2, mode="bilinear", align_corners=False
    )
    changed = diagnostic.bilinear_2x_separable(x)
    upstream = torch.randn(original.shape, dtype=torch.float64, generator=rng)
    grad_original = torch.autograd.grad(original, x, upstream)[0]
    grad_changed = torch.autograd.grad(changed, x, upstream)[0]
    torch.testing.assert_close(changed, original, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(grad_changed, grad_original, atol=2e-14, rtol=2e-14)
