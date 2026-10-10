"""Isolated models, content-addressed EF clips and resumable physical trajectories."""

import copy
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np

from ..experiments import case_seed
from ..preparation.common import atomic_json, atomic_npz, digest, emit, read_json
from .data import read_episode
from .episode import rollout
from .policy import initialize
from .protocol import clip_indices
from .value_protocol import READOUTS, anchors, coverage, quality
from .value_storage import commit, committed, load_tree, save_tree, tree_digest


class Runtime:
    def __init__(self, cfg, manifest, root, phase):
        self.cfg, self.manifest, self.root, self.phase = cfg, manifest, Path(root), phase
        self.model = self.task = None
        self.identity = digest(read_json(self.root / "identity.json"))
        self.cache_calls = 0
        self.cache_hits = 0

    def stopped(self):
        if (self.root / "STOP").exists():
            raise InterruptedError("User requested stop")
        if (
            not self.cfg["value_diagnostics"]["functional_fixture"]
            and shutil.disk_usage(self.root).free < 2 * 2**30
        ):
            raise OSError("Disk space below2GiB; preserve commits and resume after freeing space")

    def progress(self, **detail):
        self.stopped()
        value = dict(phase=self.phase, **detail)
        atomic_json(self.root / "worker_progress.json", value)
        emit("progress", **value)

    def models(self):
        if self.model is None:
            if self.cfg["value_diagnostics"]["functional_fixture"]:
                from .value_testing import SyntheticPerception

                self.model = SyntheticPerception()
                coordinates = np.indices((112, 112)).astype(np.float32)
            else:
                from .perception import CASLPerception

                self.model = CASLPerception(self.cfg)
                coordinates = self.model.coordinates
            atomic_npz(self.root / "coordinates.npz", coordinates=coordinates)
        self.task_only()
        return self.model, self.task

    def task_only(self):
        if self.task is None:
            if self.cfg["value_diagnostics"]["functional_fixture"]:
                from .value_testing import SyntheticTask

                self.task = SyntheticTask(self.cfg)
            else:
                from .task import EFService

                with np.load(self.root / "coordinates.npz", allow_pickle=False) as z:
                    coordinates = z["coordinates"]
                self.task = EFService(self.cfg, self.root / "jobs" / self.phase, coordinates)
        return self.task

    def close(self):
        if self.task:
            self.task.close()

    def readout(self, images, truth, reference=None, domain="polar", condition=""):
        images = np.asarray(images, np.float32)
        if not np.isfinite(images).all():
            raise FloatingPointError("Nonfinite EF input")
        task = self.task_only()
        results = {}
        for protocol, stride in READOUTS.items():
            ids = clip_indices(len(images), 32, 2, stride)
            predictions = []
            for i, indexes in enumerate(ids):
                self.stopped()
                array = np.ascontiguousarray(images[indexes][None])
                key = hashlib.sha256(
                    (self.identity + domain + str(array.shape)).encode() + array.tobytes()
                ).hexdigest()
                directory = self.root / "ef_cache" / key[:2]
                file = directory / (key + ".json")
                if file.exists():
                    receipt = read_json(file)
                    if (
                        receipt.get("key") != key
                        or not np.isfinite(receipt["prediction"])
                        or receipt.get("checksum")
                        != digest(dict(key=key, prediction=receipt["prediction"]))
                    ):
                        raise ValueError("EF clip cache corrupted")
                    pred = receipt["prediction"]
                    self.cache_hits += 1
                else:
                    output = task.request(array, False, domain)
                    pred = float(output["predictions"][0])
                    if not np.isfinite(pred):
                        raise FloatingPointError("EF prediction not finite")
                    atomic_json(
                        file,
                        dict(
                            key=key,
                            prediction=pred,
                            checksum=digest(dict(key=key, prediction=pred)),
                        ),
                    )
                    self.cache_calls += 1
                predictions.append(pred)
                if i % 32 == 0:
                    self.progress(
                        condition=condition, readout=protocol, clip=i + 1, total_clips=len(ids)
                    )
            value = float(np.mean(predictions))
            results[protocol] = dict(
                prediction=value,
                absolute_error=abs(value - truth),
                clips=predictions,
                coverage=coverage(ids, len(images)),
                preservation_error=abs(value - reference[protocol]["prediction"])
                if reference
                else None,
            )
        return results


def trajectory(
    runtime,
    name,
    seed,
    schedule,
    initial=None,
    prefix=None,
    force_second=False,
    tag="",
    frames_limit=None,
    capture_indices=None,
    fixed_lines=None,
):
    """Chunk receipts commit frames+state together; full receipt is written last."""
    attempt_start = time.monotonic()
    cfg, manifest, root = runtime.cfg, runtime.manifest, runtime.root
    target = read_episode(cfg, manifest, name, count=frames_limit)
    schedule = np.asarray(schedule, np.int32)
    if schedule.shape != (len(target), 2):
        raise ValueError("Budget sequence must cover the entire video")
    if schedule[0].tolist() != [10, 4]:
        raise ValueError("Common cold acquisition changed")
    logical_seed = case_seed(seed, name)
    initial = copy.deepcopy(initial)
    if initial is not None and initial["seed"] != logical_seed:
        raise ValueError("Branch seed differs from saved state")
    start = 0 if initial is None else initial["next_frame"]
    if prefix is None:
        prefix = (np.empty((0, 112, 112), np.float32), np.empty((0, 112, 112), np.uint8), [])
    if len(prefix[0]) != start or len(prefix[1]) != start or len(prefix[2]) != start:
        raise ValueError("Incomplete immutable prefix")
    tree_hash = tree_digest(initial) if initial else None
    unit = dict(
        batch=runtime.identity,
        case=name,
        data_sha=manifest["files"][name]["sha256"],
        seed=seed,
        schedule=schedule.tolist(),
        force_second=force_second,
        branch_state=tree_hash,
        locked_observations=fixed_lines,
        prefix_sha=tree_digest(
            dict(
                images=prefix[0],
                masks=prefix[1],
                rows=[{k: x[k] for k in ["k1", "k2", "lines1", "lines2"]} for x in prefix[2]],
            )
        ),
        captures=sorted(set(anchors(len(target))) | set(capture_indices or [])),
    )
    directory = root / "trajectories" / Path(name).stem / f"s{seed}" / digest(unit)[:20]
    result = committed(directory, unit)
    if result is not None:
        runtime.progress(case=name, condition=tag, operation="REUSE_TRAJECTORY")
        with np.load(directory / "images.npz", allow_pickle=False) as z:
            images, masks = z["images"].copy(), z["masks"].copy()
        return images, masks, read_json(directory / "rows.json"), result, directory
    directory.mkdir(parents=True, exist_ok=True)
    arrays = list(prefix[0])
    masks = list(prefix[1])
    rows = list(prefix[2])
    state = initial
    for chunk in sorted((directory / "chunks").glob("*")):
        if not chunk.is_dir() or not (chunk / "complete.json").exists():
            continue
        committed(chunk, dict(unit=unit, end=int(chunk.name)))
        values = load_tree(chunk / "chunk.npz")
        if values["start"] != len(arrays):
            raise ValueError("Gap/overlap in committed trajectory chunks")
        arrays.extend(values["images"])
        masks.extend(values["masks"])
        rows.extend(values["rows"])
        state = values["state"]
        if state["next_frame"] != len(arrays):
            raise ValueError("State/frame commit disagreement")
    if len(arrays) > len(target):
        raise ValueError("Too many committed frames")
    chunk_images = []
    chunk_masks = []
    chunk_rows = []
    chunk_start = len(arrays)
    selected = set(unit["captures"])

    def capture(stage, index, value):
        nonlocal chunk_start
        runtime.stopped()
        if stage in ["before", "middle"] and index in selected:
            save_tree(directory / f"{stage}_{index:05d}.npz", value)
        if stage != "after":
            return
        if index in selected:
            save_tree(directory / f"after_{index:05d}.npz", value)
        row = value["row"]
        mask = value["final_mask"].astype(np.uint8)
        image = value["prediction"]
        if len(set(row["lines1"] + row["lines2"])) != row["actual_lines"] or row[
            "actual_lines"
        ] != int(mask[0].sum()):
            raise AssertionError("Physical acquisition cardinality mismatch")
        arrays.append(image.copy())
        masks.append(mask.copy())
        rows.append(row)
        chunk_images.append(image.copy())
        chunk_masks.append(mask.copy())
        chunk_rows.append(row)
        runtime.progress(
            case=name, condition=tag, seed=seed, frame=index + 1, total_frames=len(target)
        )
        if len(chunk_images) >= cfg["value_diagnostics"]["checkpoint_frames"] or index + 1 == len(
            target
        ):
            chunk = directory / "chunks" / f"{index + 1:05d}"
            save_tree(
                chunk / "chunk.npz",
                dict(
                    start=chunk_start,
                    state=value,
                    images=np.stack(chunk_images),
                    masks=np.stack(chunk_masks),
                    rows=chunk_rows,
                ),
            )
            commit(chunk, dict(unit=unit, end=index + 1), dict(frames=index + 1), ["chunk.npz"])
            chunk_start = index + 1
            chunk_images.clear()
            chunk_masks.clear()
            chunk_rows.clear()

    if len(arrays) < len(target):
        model, task = runtime.models()
        params = initialize(42, cfg)
        rollout(
            cfg,
            model,
            task,
            target[len(arrays) :],
            params,
            "E0",
            logical_seed,
            fixed=schedule[len(arrays) :],
            initial_state=state,
            capture=capture,
            retain_contexts=False,
            force_second=force_second,
            fixed_lines=fixed_lines[len(arrays) :] if fixed_lines is not None else None,
        )
    images = np.stack(arrays)
    mask_array = np.stack(masks)
    if not np.isfinite(images).all():
        raise FloatingPointError("Nonfinite complete video")
    atomic_npz(directory / "images.npz", images=images, masks=mask_array)
    atomic_json(directory / "rows.json", rows)
    q = quality(target, images)
    result = dict(
        case=name,
        seed=seed,
        frames=len(target),
        quality=q,
        mean_lines=float(np.mean(schedule.sum(1))),
        total_lines=int(schedule.sum()),
        inference_seconds=float(sum(x["frame_wall_s"] for x in rows)),
        perception_calls=sum(x["perception_calls"] for x in rows),
        executed_calls=sum(
            int(x.get("executed_stage1", True)) + int(x["reverse_steps_stage2"] > 0) for x in rows
        ),
        suffix_executed_calls=sum(
            int(x.get("executed_stage1", True)) + int(x["reverse_steps_stage2"] > 0)
            for x in rows[start:]
        ),
        suffix_frame_seconds=sum(x["frame_wall_s"] for x in rows[start:]),
        execution_seconds_this_attempt=time.monotonic() - attempt_start,
        timing_scope="whole logical trajectory includes reused prefix; suffix counts exclude reused first-stage work; lost uncommitted work excluded",
        psnr=float(np.mean([x["psnr"] for x in rows])),
        ssim=float(np.mean([x["ssim"] for x in rows])),
        cache_scope="full physical trajectory; immutable prefix reused for branches",
        functional_fixture=cfg["value_diagnostics"]["functional_fixture"],
    )
    names = ["images.npz", "rows.json"] + [p.name for p in directory.glob("*_*.npz")]
    commit(directory, unit, result, names)
    return images, mask_array, rows, result, directory
