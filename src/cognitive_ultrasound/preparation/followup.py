"""One bounded follow-up: longer confirmation, more risk data, frozen BF diagnosis."""

import argparse
import time
from pathlib import Path

import numpy as np

from ..config import load, path
from ..provenance import sha256
from .common import atomic_json, atomic_npz, emit, read_frames, read_json, verify_manifest
from .suite import Coordinator, run


def assert_new_holdout(previous, current):
    used = set().union(*(set(v) for k, v in previous["cohorts"].items() if k != "train"))
    if used & set(current["cohorts"]["confirmation"]):
        raise ValueError("Follow-up confirmation overlaps previous development/confirmation cases")


class Followup(Coordinator):
    def initialize(self):
        source = Path(self.cfg["followup_source"]).resolve()
        if source == self.root or source in self.root.parents or self.root in source.parents:
            raise ValueError("Follow-up and source outputs must be separate sibling directories")
        old_cfg, old_manifest = (
            read_json(source / "config.json"),
            read_json(source / "manifest.json"),
        )
        verify_manifest(old_cfg, old_manifest)
        if read_json(source / "status.json")["status"] == "running":
            raise RuntimeError("Source experiment must finish before follow-up")
        chosen = read_json(source / "selection.json")
        if not chosen["accelerated_default"] or chosen["variant"] not in self.cfg["candidates"]:
            raise ValueError("Follow-up requires the previously qualified acceleration candidate")
        for key in ("budget", "checkpoint", "split_manifest", "seeds", "gate"):
            if old_cfg[key] != self.cfg[key]:
                raise ValueError("Inherited scientific settings differ: " + key)
        provenance = {"source": str(source), "files": {}}
        files = (
            ["identity.json", "selection.json", "manifest.json"]
            + [f"jobs/bf_{stage}/training/checkpoint.npz" for stage in ("codec", "prior", "filter")]
            + [f"jobs/fixed_{chosen['variant']}/result.json"]
        )
        for name in files:
            provenance["files"][name] = sha256(source / name)
        if (self.root / "source_run.json").exists() and read_json(
            self.root / "source_run.json"
        ) != provenance:
            raise ValueError("Source artifacts changed since this follow-up")
        super().initialize()
        assert_new_holdout(old_manifest, read_json(self.root / "manifest.json"))
        old_identity = read_json(source / "identity.json")
        new_identity = read_json(self.root / "identity.json")
        for key in ("weights", "casl_commit", "zea_commit"):
            if old_identity[key] != new_identity[key]:
                raise ValueError("Official assets changed: " + key)
        atomic_json(self.root / "source_run.json", provenance)
        atomic_json(
            self.root / "screening.json",
            dict(variant=chosen["variant"], inherited_from=str(source)),
        )
        fixed = f"jobs/fixed_{chosen['variant']}/result.json"
        atomic_json(
            self.root / fixed, dict(**read_json(source / fixed), inherited_from=str(source / fixed))
        )

    def execute(self):
        from .analysis import report, select_variant

        root, cfg = self.root, self.cfg
        candidate = read_json(root / "screening.json")["variant"]
        # Reuse prior sampler screening, but challenge long closed-loop quality on NEW cases.
        for variant in (candidate, "reference"):
            self.job(
                "confirm_" + variant,
                "trajectory",
                "A",
                cohort="confirmation",
                variant=variant,
                seeds=cfg["seeds"],
            )
        selected = select_variant(root, cfg, confirmation=True)
        self.job("bf_diagnosis", "bf_diagnose", "B")
        allowed = selected["accelerated_default"]
        reason = (
            None if allowed else "Long-trajectory acceleration gate failed; no expanded risk runs"
        )
        for cohort in ("train", "development"):
            self.job(
                "risk_" + cohort,
                "trajectory",
                "C",
                condition=reason,
                cohort=cohort,
                variant=candidate,
                seeds=cfg["seeds"],
            )
        self.job(
            "risk_confirmation",
            "reuse",
            "C",
            condition=reason,
            source="confirm_" + candidate,
            dependencies=("confirm_" + candidate,),
        )
        self.job(
            "risk_probe",
            "risk_probe",
            "C",
            condition=reason,
            dependencies=tuple("risk_" + c for c in ("train", "development", "confirmation")),
        )
        passed = allowed and self.eligible("risk_probe")
        self.job(
            "branches",
            "branches",
            "C",
            condition=None if passed else "Future-risk gate did not pass",
            dependencies=("confirm_" + candidate,),
        )
        self.job(
            "closed_loop",
            "closed_loop",
            "C",
            dependencies=("branches", "risk_probe"),
            condition=None if passed else "Future-risk gate did not pass",
        )
        self.state["status"] = (
            "completed"
            if all(j["status"] == "completed" for j in self.state["jobs"].values())
            else "finished_with_gaps"
        )
        self.save()
        report(root)


def diagnose_filter(task, cfg, output):
    from ..belief_filter.runner import load_checkpoint, runtime

    tf = runtime(task.get("cpu", False))
    from ..belief_filter.core import ArrayOracle, Models, initial_memory, step
    from .belief import configuration

    source = Path(cfg["followup_source"])
    original_cfg, manifest = read_json(source / "config.json"), read_json(source / "manifest.json")
    model_cfg = configuration(original_cfg, manifest)
    models = Models(model_cfg)
    checkpoint = source / "jobs/bf_filter/training/checkpoint.npz"
    before_hash = sha256(checkpoint)
    metadata = load_checkpoint(checkpoint, models)
    if metadata["stage"] != "filter" or metadata["status"] != "completed":
        raise ValueError("A completed filter checkpoint is required")
    names = manifest["cohorts"]["development"]
    sample = read_frames(original_cfg, "val", names[0], 3)
    original_update = models.update
    z = tf.TensorSpec((1, 28, 28, model_cfg["latent_channels"]), tf.float32)
    im = tf.TensorSpec((1, 112, 112, 1), tf.float32)
    compiled = tf.function(original_update, input_signature=[z, im, im], autograph=False)
    predictions, times = {}, {}
    for mode, update in (("eager", original_update), ("graph", compiled)):
        models.update = update
        memory, pictures, timings = initial_memory(models), [], []
        for index, target in enumerate(sample):
            started = time.perf_counter()
            memory, state = step(models, ArrayOracle(target), memory, index, model_cfg, "uniform")
            pictures.append(np.asarray(state["reconstruction"]))
            timings.append(time.perf_counter() - started)
        predictions[mode], times[mode] = np.stack(pictures), timings
    equivalent = bool(np.allclose(predictions["eager"], predictions["graph"], rtol=1e-4, atol=1e-5))
    faster = np.median(times["graph"][1:]) < 0.9 * np.median(times["eager"][1:])
    models.update = compiled if equivalent and faster else original_update
    atomic_json(
        output / "preflight.json",
        dict(
            equivalent=equivalent,
            times_s=times,
            execution="graph_update" if equivalent and faster else "eager",
            tf32=False,
        ),
    )
    records = []
    for name in names:
        directory = output / Path(name).stem
        if (directory / "complete.json").exists():
            records.append(read_json(directory / "complete.json"))
            continue
        frames = read_frames(original_cfg, "val", name, 32)
        lines = np.linspace(0, 111, sum(model_cfg["groups"]), dtype=int)
        mask = np.zeros((112, 112, 1), np.float32)
        mask[:, lines, :] = 1
        values = {
            label: []
            for label in (
                "online",
                "reset_every_3",
                "privileged_previous_truth",
                "interpolation",
                "codec_of_interpolation",
                "full_input_codec",
            )
        }
        memories = {label: initial_memory(models) for label in list(values)[:3]}
        for index, target in enumerate(frames):
            pictures = {}
            for label in memories:
                memory = memories[label]
                if label == "reset_every_3" and index % 3 == 0:
                    memory = initial_memory(models)
                if label == "privileged_previous_truth" and index:
                    # Offline diagnostic only. Never deployed or used to select online actions.
                    memory.latent = models.encoder(frames[index - 1 : index])
                memories[label], state = step(
                    models, ArrayOracle(target), memory, index, model_cfg, "uniform"
                )
                pictures[label] = np.asarray(state["reconstruction"])[0]
            observed = np.where(mask > 0, target, 0)
            interpolated = interpolate_observed(observed, lines)
            pictures["interpolation"] = interpolated
            pictures["codec_of_interpolation"] = np.where(
                mask > 0,
                observed,
                np.asarray(models.decoder(models.encoder(interpolated[None])))[0],
            )
            pictures["full_input_codec"] = np.asarray(models.decoder(models.encoder(target[None])))[
                0
            ]
            for label, prediction in pictures.items():
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(label)
                values[label].append(float(np.abs(prediction - target)[mask == 0].mean()))
            if index in (0, 2, 15, 31):
                atomic_npz(
                    directory / f"frame_{index:04d}.npz", target=target, mask=mask, **pictures
                )
        record = dict(
            case=name,
            per_frame_unobserved_mae=values,
            means={k: float(np.mean(v)) for k, v in values.items()},
            early={k: float(np.mean(v[:3])) for k, v in values.items()},
            late={k: float(np.mean(v[16:])) for k, v in values.items()},
        )
        atomic_json(directory / "complete.json", record)
        records.append(record)
        emit("BF_DIAGNOSE", case=name, means=record["means"])
    assert sha256(checkpoint) == before_hash
    atomic_json(
        output / "result.json",
        dict(
            status="completed",
            cases=records,
            means={
                k: float(np.mean([r["means"][k] for r in records])) for k in records[0]["means"]
            },
            checkpoint_sha256=before_hash,
            trained_updates=0,
            caveat="Frozen-model diagnosis. Full input codec and previous truth are privileged offline diagnostics, not deployable baselines.",
        ),
    )


def interpolate_observed(observed, lines):
    return np.stack([np.interp(np.arange(112), lines, observed[r, lines, 0]) for r in range(112)])[
        ..., None
    ].astype("float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preparation_followup.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load(path(args.config))
    # Keep the exact existing asset paths so reuse checks are explicit.
    source = read_json(Path(cfg["followup_source"]) / "config.json")
    for key in ("checkpoint", "split_manifest", "data_root"):
        cfg[key] = source[key]
    run(cfg, args.output, args.resume, coordinator_type=Followup)


if __name__ == "__main__":
    main()
