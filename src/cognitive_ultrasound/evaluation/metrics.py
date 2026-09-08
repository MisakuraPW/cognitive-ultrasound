import numpy as np
from skimage.metrics import structural_similarity


def uint8_image(image):
    """Match upstream AgentResults.to_uint8: clip, then truncate (do not round)."""
    return np.clip((np.asarray(image) + 1.0) * 127.5, 0, 255).astype(np.uint8)


def psnr(reference, prediction, data_range=255.0):
    error = np.mean((np.asarray(reference, dtype=np.float64) - prediction) ** 2)
    return float("inf") if error == 0 else float(10 * np.log10(data_range**2 / error))


def dice(reference, prediction):
    reference, prediction = np.asarray(reference, dtype=bool), np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape:
        raise ValueError("Dice masks must have matching shapes")
    denominator = reference.sum() + prediction.sum()
    return (
        1.0
        if denominator == 0
        else float(2 * np.logical_and(reference, prediction).sum() / denominator)
    )


class Metrics:
    def __init__(self, names, checkpoint_root=None):
        self.names, self.lpips = names, None
        if "lpips" in names:
            # Use the SAME zea LPIPS implementation/preset as the official benchmark.
            from ..official import activate

            activate("jax")
            from pathlib import Path
            from unittest.mock import patch

            from zea.metrics import Metrics as OfficialMetrics
            from zea.models.lpips import LPIPS

            from ..config import ROOT

            directory = Path(checkpoint_root or ROOT / "checkpoints/evaluation") / "lpips"
            if not (directory / "config.json").is_file():
                raise FileNotFoundError(
                    "Pinned LPIPS weights missing; run fetch-assets --with-evaluation"
                )
            loader = LPIPS.from_preset
            with patch.object(
                LPIPS, "from_preset", side_effect=lambda *a, **k: loader(str(directory), **k)
            ):
                self.lpips = OfficialMetrics(metrics=["lpips"], image_range=[0, 255])

    def __call__(self, reference, prediction):
        reference, prediction = uint8_image(reference).squeeze(), uint8_image(prediction).squeeze()
        result = {}
        if "psnr" in self.names:
            result["psnr"] = psnr(reference, prediction)
        if "ssim" in self.names:
            result["ssim"] = float(
                structural_similarity(
                    reference,
                    prediction,
                    data_range=255,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )
            )
        if self.lpips is not None:
            result["lpips"] = float(
                np.asarray(
                    self.lpips(
                        reference[None, ..., None].astype("float32"),
                        prediction[None, ..., None].astype("float32"),
                    )["lpips"]
                ).mean()
            )
        return result
