from pathlib import Path

import numpy as np
from PIL import Image

from ..evaluation.metrics import uint8_image


def save_frame(folder, state):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("ground_truth", "ground_truth"),
        ("sparse_observation", "observation"),
        ("reconstruction", "reconstruction"),
    ):
        pixels = uint8_image(state[key]).squeeze()
        if name == "sparse_observation":
            pixels = np.where(state["observation_mask"].squeeze() > 0, pixels, 0).astype("uint8")
        Image.fromarray(pixels).save(folder / f"{name}.png")
    entropy = state["entropy_map"].squeeze()
    Image.fromarray(np.uint8(255 * entropy / max(float(entropy.max()), 1e-12))).save(
        folder / "entropy.png"
    )
    # selected_action is for t+1; acquired_action belongs to the current observation.
    selected = np.broadcast_to(state["selected_action"].astype(bool)[None], (112, 112))
    Image.fromarray(selected.astype("uint8") * 255).save(folder / "selected_lines.png")
    Image.fromarray(state["observation_mask"].squeeze().astype("uint8") * 255).save(
        folder / "acquired_lines.png"
    )
