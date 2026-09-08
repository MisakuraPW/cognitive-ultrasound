"""Paper endpoint is model-to-model agreement, not human-label Dice."""

import numpy as np
from scipy.ndimage import label

from .metrics import dice


def segmentation_failure(masks, consecutive=5):
    run = 0
    for mask in masks:
        # Explicit 8-connectivity; retain the raw masks/flag for alternative protocols.
        count = label(np.asarray(mask).squeeze() > 0, structure=np.ones((3, 3)))[1]
        run = run + 1 if count > 1 else 0
        if run >= consecutive:
            return True
    return False


class Segmentation:
    def __init__(self, checkpoint_root):
        from ..official import activate

        activate("jax")
        from pathlib import Path
        from unittest.mock import patch

        from ulsa.downstream_task import EchoNetSegmentation
        from zea.models.echonet import EchoNetDynamic

        directory = Path(checkpoint_root) / "segmentation"
        if not (directory / "saved_model.pb").is_file():
            raise FileNotFoundError(
                "Pinned segmentation weights missing; run fetch-assets --with-evaluation"
            )
        loader = EchoNetDynamic.from_preset
        with patch.object(
            EchoNetDynamic, "from_preset", side_effect=lambda *a, **k: loader(str(directory), **k)
        ):
            self.model = EchoNetSegmentation(batch_size=1)

    def __call__(self, reference, reconstruction):
        gt = np.asarray(self.model.call_generic(reference[None]))[0]
        pred = np.asarray(self.model.call_generic(reconstruction[None]))[0]
        return gt, pred, dice(gt, pred)
