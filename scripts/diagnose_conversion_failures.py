"""Read-only AVI diagnostics; write findings only under results/local_conversion."""

import ast
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"G:\SRTP\dataset\EchoNet-Dynamic")
OUT = ROOT / "results/local_conversion/failure_diagnosis.json"


def upstream_functions():
    # Execute only these pure NumPy functions, avoiding an extra ML runtime.
    source = ROOT / "vendor/casl/zea/zea/data/convert/echonet.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    subset = ast.Module(
        body=[
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("segment", "accept_shape")
        ],
        type_ignores=[],
    )
    scope = {"np": np}
    exec(compile(subset, str(source), "exec"), scope)
    return scope["segment"], scope["accept_shape"]


def scores(frame, segment, accept):
    tensor = segment(frame.astype(np.float64)[None] / 255)[0]
    lower = np.linspace(78, 47, 21).astype(np.int32)
    upper = np.linspace(67, 47, 21).astype(np.int32)
    left = sum(np.sum(tensor[upper[i] : row, i]) for i, row in enumerate(lower))
    cols = np.linspace(70, 111, 42).astype(np.int32)
    bot = np.linspace(17, 57, 42).astype(np.int32)
    top = np.linspace(17, 80, 42).astype(np.int32)
    values = sorted(
        [float(v) for i, c in enumerate(cols) for v in tensor[bot[i] : top[i], c]], reverse=True
    )
    return {"accepted": bool(accept(tensor)), "left": float(left), "right": sum(values[100:])}


def main():
    previous = json.loads(
        (ROOT / "results/local_conversion/monitor_snapshot.json").read_text(encoding="utf-8")
    )
    cases = previous["known_failure_sources"]
    rows = {
        r["FileName"].removesuffix(".avi"): r for r in csv.DictReader((RAW / "FileList.csv").open())
    }
    import yaml

    splits = yaml.safe_load((ROOT / "configs/splits/split.yaml").read_text())
    assignment = {Path(name).stem: split for split, names in splits.items() for name in names}
    segment, accept = upstream_functions()
    cv2.setNumThreads(1)
    results = []
    for name in cases:
        path = RAW / "Videos" / name
        item = {
            "source": name,
            "split": assignment.get(path.stem, "rejected"),
            "csv_frames": int(rows[path.stem]["NumberOfFrames"]),
        }
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        item["sha256"] = digest.hexdigest()
        frame_scores = []
        shapes = set()
        try:
            with imageio.get_reader(path) as reader:
                for i, rgb in enumerate(reader):
                    shapes.add(tuple(rgb.shape))
                    gray = np.asarray(Image.fromarray(rgb).convert("L"))
                    score = scores(gray, segment, accept)
                    frame_scores.append(score)
                    if i == 0:
                        item["first_frame"] = score
                        item["first_frame_channels"] = {
                            str(c): scores(rgb[..., c], segment, accept) for c in range(3)
                        }
                        item["first_frame_channel_mean"] = scores(
                            rgb.mean(axis=-1), segment, accept
                        )
                        item["rgb_channels_identical"] = bool(
                            np.array_equal(rgb[..., 0], rgb[..., 1])
                            and np.array_equal(rgb[..., 1], rgb[..., 2])
                        )
            item.update(
                decoded_frames=len(frame_scores),
                shapes=sorted(shapes),
                accepted_frames=sum(s["accepted"] for s in frame_scores),
                first_accepted_frame=next(
                    (i for i, s in enumerate(frame_scores) if s["accepted"]), None
                ),
                right_min=min(s["right"] for s in frame_scores),
                right_max=max(s["right"] for s in frame_scores),
                frame_count_matches=len(frame_scores) == item["csv_frames"],
                decode_error=None,
            )
        except Exception as error:
            item.update(
                decoded_frames=len(frame_scores), decode_error=f"{type(error).__name__}: {error}"
            )
        # Historical upstream used VideoCapture (BGR) followed by COLOR_RGB2GRAY.
        cap = cv2.VideoCapture(str(path))
        cv_frames = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if cv_frames == 0:
                item["legacy_opencv_first_frame"] = scores(
                    cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY), segment, accept
                )
                item["correct_opencv_first_frame"] = scores(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), segment, accept
                )
            cv_frames += 1
        cap.release()
        item["opencv_decoded_frames"] = cv_frames
        results.append(item)
        OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(item), flush=True)


if __name__ == "__main__":
    main()
