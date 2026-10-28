"""One bounded benchmark protocol over existing engines; no training or asset downloads."""

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import psutil
import yaml
from skimage.metrics import structural_similarity

from .compute_lab.protocol import Profile
from .preparation.common import atomic_json as atomic
from .preparation.common import read_json as read
from .preparation.suite import run_lock
from .provenance import sha256 as sha

ROOT = Path(__file__).resolve().parents[2]


def hardware():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        gpu = q.stdout.strip() if q.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        gpu = None
    return dict(gpu=gpu, cpu=platform.processor(), system=platform.system())


def fingerprint(path):
    path = Path(path)
    st = path.stat()
    with path.open("rb") as f:
        head = f.read(65536)
        f.seek(max(0, st.st_size - 65536))
        tail = f.read(65536)
    return dict(
        bytes=st.st_size,
        mtime_ns=st.st_mtime_ns,
        head_tail_sha256=hashlib.sha256(head + tail).hexdigest(),
    )


def runtime_versions(pythons):
    script = "import importlib.metadata as m,json; names=['numpy','jax','jaxlib','tensorflow','keras','torch','scikit-image']; found={d.metadata['Name'].lower():d.version for d in m.distributions()}; print(json.dumps({n:found.get(n) for n in names}))"
    results = {}
    for executable in sorted(set(pythons.values())):
        try:
            p = subprocess.run(
                [executable, "-c", script], capture_output=True, text=True, timeout=15
            )
            results[executable] = (
                json.loads(p.stdout) if p.returncode == 0 else {"unavailable": p.returncode}
            )
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            results[executable] = {"unavailable": True}
    return results


@dataclass(frozen=True)
class CoarseProfile(Profile):
    """Change integration count, retain original warm noise interval (not native25)."""

    def variant(self):
        return dict(
            initial_step=450,
            precision="float32",
            inference_steps=self.steps,
            guidance_steps=self.dps,
        )


def catalog():
    from .compute_lab.protocol import profiles

    result = {
        k: dict(kind="casl", profile=p.record()) for k, p in profiles(torch_mode="eager").items()
    }
    result["torch_25_fp32"] = dict(
        kind="casl",
        profile=Profile("torch_25_fp32", backend="torch", mode="eager", steps=25, dps=25).record(),
    )
    result["jax_full500"] = dict(
        kind="casl",
        profile=Profile("jax_full500", steps=500, dps=500).record(),
        note="opt-in cost control; not a required default run",
    )
    result["jax_async_output"] = dict(
        kind="unavailable",
        category="A",
        reason="Legacy async full-trajectory writer is intentionally excluded from this bounded-output benchmark; no comparable per-frame disk workload",
    )
    result["jax_cache"] = dict(kind="casl", profile=Profile("jax_cache", io="cache").record())
    for steps in (16, 8, 4):
        p = CoarseProfile(f"jax_coarse{steps}", steps=steps, dps=steps)
        result[p.name] = dict(kind="casl_coarse", profile=p.record(), warm_initial_step=450)
    for name, precision in [("jax_10_fp32", "fp32"), ("jax_5_fp32", "fp32")]:
        steps = int(name.split("_")[1])
        p = Profile(name, steps=steps, dps=steps, precision=precision)
        result[name] = dict(kind="casl", profile=p.record())
    result["fast_belief"] = dict(
        kind="belief",
        category="C",
        groups=[10, 4],
        requires="existing locally trained checkpoint; no automatic training",
    )
    result["interpolation"] = dict(kind="control", category="C")
    result["seqdiff_plus"] = dict(
        kind="unavailable",
        category="C",
        reason="No verified compatible ViViT weights; not silently replaced by an untrained predictor",
    )
    result["senior_original"] = dict(
        kind="unavailable",
        category="C",
        reason="Only thesis, no original code/weights; independent BF is a separate method",
    )
    return result


def config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if cfg["budget"] != 14 or cfg["cases"] < 1 or cfg["frames"] < 4:
        raise ValueError("Benchmark locks14 lines, positive cases, at least4 frames")
    if cfg["warmup"] < 1 or cfg["repeats"] < 2 or cfg["timeout_seconds"] <= 0:
        raise ValueError("Invalid timing protocol")
    if len(set(cfg["methods"])) != len(cfg["methods"]) or any(
        k not in catalog() for k in cfg["methods"]
    ):
        raise ValueError("Unknown/duplicate methods")
    if cfg["methods"][0] != "official" or cfg["modes"] != ["fixed", "closed_loop"]:
        raise ValueError("Official reference first; both observation protocols required")
    return cfg


def measurements(target, prediction, mask):
    if (
        prediction.shape != target.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise FloatingPointError("Invalid prediction; no finite-value replacement")
    if (
        mask.shape != target.shape
        or not np.isin(mask, [0, 1]).all()
        or not np.array_equal(mask, np.broadcast_to(mask[:1], mask.shape))
    ):
        raise ValueError("Mask must represent whole binary columns")
    error = prediction.astype(np.float64) - target
    mse = float(np.mean(error**2))
    return dict(
        mse=mse,
        mae=float(np.abs(error).mean()),
        psnr=float(10 * np.log10(4 / max(mse, 1e-12))),
        ssim=float(
            structural_similarity(
                target[..., 0],
                prediction[..., 0],
                data_range=2,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ),
        unobserved_mae=float(np.abs(error)[~mask.astype(bool)].mean()),
        observed_max_error=float(np.abs(error)[mask.astype(bool)].max()),
        actual_lines=int(mask[0, :, 0].sum()),
    )


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Invalid elapsed times")
    return dict(
        mean_s=float(values.mean()),
        p50_s=float(np.median(values)),
        p95_s=float(np.quantile(values, 0.95)),
        fps=float(1 / values.mean()),
    )


def force_mask(engine, kind, lines):
    if kind.startswith("casl"):
        if hasattr(engine, "adapter"):
            engine.adapter.force_lines(lines)
        else:
            import torch

            m = torch.zeros_like(engine.mask[:, -1:])
            m[..., lines] = 1
            engine.mask = torch.cat([engine.mask[:, :-1], m], 1)


class BeliefEngine:
    def __init__(self, cfg):
        import torch

        from .fast_belief.model import BeliefSystem
        from .fast_belief.runner import configure, load

        path = cfg.get("belief_checkpoint")
        if not path or not Path(path).is_file():
            raise FileNotFoundError(
                "Existing trained belief_checkpoint required; training is never started"
            )
        self.torch = torch
        self.device = configure(False, cfg["threads"])
        payload = load(path)
        model_cfg = read(Path(path).parent.parent / "config.json")
        self.model = BeliefSystem(model_cfg).to(self.device).eval()
        self.model.load_state_dict(payload["model"], strict=True)

    def reset(self, seed, initial=None):
        with self.torch.no_grad():
            self.memory = self.model.initial(1, self.device)
        self.index = 0

    def capture(self):
        return self.memory.detach(), self.index

    def reinstate(self, state):
        self.memory, self.index = state

    def step(self, target, noise=None):
        from .fast_belief.model import ArrayOracle

        t = self.torch
        with t.no_grad():
            self.memory, state = self.model.frame(
                ArrayOracle(t.from_numpy(target[..., 0].copy())[None, None].to(self.device)),
                self.memory,
                self.index,
                policy=self.policy,
            )
        self.index += 1
        t.cuda.synchronize()
        return {}, dict(
            prediction=state["image"][0].permute(1, 2, 0).cpu().numpy(),
            mask=state["mask"][0].permute(1, 2, 0).cpu().numpy(),
            next_action=state["mask"][0, 0, 0].cpu().numpy(),
        )

    def peak(self):
        return self.torch.cuda.max_memory_allocated()


def build(cfg, method, root):
    spec = catalog()[method]
    if spec["kind"] == "unavailable":
        raise FileNotFoundError(spec["reason"])
    if spec["kind"] == "control":
        return None
    if spec["kind"] == "belief":
        return BeliefEngine(cfg)
    if not Path(cfg["checkpoint"]).is_dir():
        raise FileNotFoundError("Missing existing CASL checkpoint directory")
    if method.startswith("torch") and not all(
        (Path(cfg["torch_export"]) / f).is_file() for f in ["network.json", "ema.npz", "probes.npz"]
    ):
        raise FileNotFoundError(
            "Missing verified existing Torch weight export; no retraining or downloads"
        )
    from .compute_lab.engines import JaxEngine, TorchEngine

    fields = Profile.__dataclass_fields__
    cls = CoarseProfile if spec["kind"] == "casl_coarse" else Profile
    p = cls(**{k: v for k, v in spec["profile"].items() if k in fields})
    return (JaxEngine if p.backend == "jax" else TorchEngine)(
        cfg, p, cfg["budget"], Path(cfg["torch_export"])
    )


def disk_guard(root, cfg):
    if (root / "STOP").exists():
        raise InterruptedError("Stop requested")
    if (
        sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        > cfg["max_output_gib"] * 2**30
    ):
        raise OSError("Benchmark output cap exceeded; no automatic deletion")


def worker(cfg, method, root):
    started = time.perf_counter()
    kind = catalog()[method]["kind"]
    engine = build(cfg, method, root)
    setup_seconds = time.perf_counter() - started
    atomic(
        root / f"{method}.environment.json",
        dict(hardware=hardware(), python=platform.python_version(), category=catalog()[method]),
    )
    manifest = read(root / "manifest.json")
    uniform = np.linspace(0, 111, 14).round().astype(int)
    output = root / method
    output.mkdir(exist_ok=True)
    for mode in cfg["modes"]:
        for name in manifest["cases"]:
            folder = output / mode / Path(name).stem
            folder.mkdir(parents=True, exist_ok=True)
            marker = folder / "complete.json"
            if marker.exists():
                saved = read(marker)
                for f, checksum in saved["artifacts"].items():
                    if sha(folder / f) != checksum:
                        raise ValueError("Committed benchmark artifact changed")
                continue
            case_start = time.perf_counter()
            seed = int.from_bytes(
                hashlib.sha256((str(cfg["seed"]) + name).encode()).digest()[:4], "little"
            )
            rng = np.random.default_rng(seed)
            initial_file = root / (Path(name).stem + ".initial.npz")
            if engine is not None:
                if method.startswith("torch"):
                    if sha(initial_file) != initial_file.with_suffix(".sha256").read_text().strip():
                        raise ValueError("Official initial-state checksum changed")
                    with np.load(initial_file, allow_pickle=False) as z:
                        initial = {k: z[k] for k in z.files}
                    engine.reset(seed, initial)
                else:
                    engine.reset(seed)
                if method == "official":
                    if not initial_file.exists():
                        np.savez_compressed(initial_file, **engine.snapshot())
                        initial_file.with_suffix(".sha256").write_text(
                            sha(initial_file), encoding="ascii"
                        )
                if kind == "belief":
                    engine.policy = "uniform" if mode == "fixed" else "greedy"
            rows, previews, reference_images, reference_masks, reference_actions = (
                [],
                {},
                [],
                [],
                [],
            )
            reference_file = root / "official" / mode / Path(name).stem / "reference.npz"
            ref = (
                np.load(reference_file, allow_pickle=False)
                if method != "official" and reference_file.exists()
                else None
            )
            micro = None
            peak_ram = psutil.Process().memory_info().rss
            if (
                ref is not None
                and sha(reference_file)
                != read(reference_file.parent / "complete.json")["artifacts"]["reference.npz"]
            ):
                ref.close()
                raise ValueError("Reference outputs corrupted")
            previous_prediction, previous_target = None, None
            try:
                from .compute_lab.data import frames

                file = Path(cfg["data_root"]) / "val" / name
                with h5py.File(file) as h:
                    count = min(cfg["frames"], len(h["data/image"]))
                if count < 4:
                    raise ValueError("Locked video shorter than4 frames")
                io_mode = catalog()[method].get("profile", {}).get("io", "serial")
                with contextlib.closing(frames(file, count, io_mode)) as iterator:
                    for index in range(count):
                        disk_guard(root, cfg)
                        tick = time.perf_counter()
                        target = next(iterator)
                        io_seconds = time.perf_counter() - tick
                        noise = rng.standard_normal((2, 112, 112, 3)).astype("float32")
                        if engine is not None and mode == "fixed":
                            force_mask(engine, kind, uniform)

                        def step():
                            if engine is not None:
                                return engine.step(target, noise)[1]
                            prediction = np.stack(
                                [
                                    np.interp(np.arange(112), uniform, row[uniform, 0])
                                    for row in target
                                ]
                            )[..., None].astype("float32")
                            mask = np.zeros_like(target)
                            mask[:, uniform] = 1
                            return dict(prediction=prediction, mask=mask, next_action=mask[0, :, 0])

                        if index == 2 and engine is not None:
                            state = engine.capture()
                            for _ in range(cfg["warmup"]):
                                engine.reinstate(state)
                                step()
                            timings = []
                            for _ in range(cfg["repeats"]):
                                engine.reinstate(state)
                                t0 = time.perf_counter()
                                step()
                                timings.append(time.perf_counter() - t0)
                            engine.reinstate(state)
                            micro = statistics(timings)
                            # Microbenchmark is diagnostic overhead, excluded from sequence timing.
                            tick = time.perf_counter() - io_seconds
                        t0 = time.perf_counter()
                        arrays = step()
                        wall = time.perf_counter() - t0
                        prediction, mask = arrays["prediction"], arrays["mask"]
                        score = measurements(target, prediction, mask)
                        if score["actual_lines"] != 14:
                            raise ValueError("Measured acquisition budget mismatch")
                        action = np.asarray(arrays["next_action"]).reshape(-1)
                        if action.shape != (112,):
                            raise ValueError("Unexpected action space")
                        row = dict(frame=index, io_s=io_seconds, model_wall_s=wall, **score)
                        row["temporal_error"] = (
                            None
                            if previous_prediction is None
                            else float(
                                np.abs(
                                    (prediction - previous_prediction) - (target - previous_target)
                                ).mean()
                            )
                        )
                        previous_prediction, previous_target = prediction.copy(), target.copy()
                        if ref is not None:
                            rmask = ref["mask"][index].astype(bool)
                            common = ~(rmask | mask.astype(bool))
                            row.update(
                                reference_max_abs=float(
                                    np.abs(prediction - ref["prediction"][index]).max()
                                ),
                                mask_equal=bool(np.array_equal(mask, ref["mask"][index])),
                                action_equal=bool(np.array_equal(action, ref["action"][index])),
                                common_unobserved_mae=float(
                                    np.abs(prediction - target)[common].mean()
                                )
                                if common.any()
                                else None,
                            )
                        if method == "official":
                            reference_images.append(prediction.copy())
                            reference_masks.append(mask.copy())
                            reference_actions.append(action.copy())
                        if index in (0, count // 2, count - 1):
                            previews[f"prediction_{index}"] = prediction.copy()
                            previews[f"target_{index}"] = target.copy()
                            previews[f"mask_{index}"] = mask.copy()
                        row["full_frame_s"] = time.perf_counter() - tick
                        rows.append(row)
                        peak_ram = max(peak_ram, psutil.Process().memory_info().rss)
                        print(
                            json.dumps(
                                dict(
                                    event="FRAME",
                                    method=method,
                                    mode=mode,
                                    case=name,
                                    frame=index + 1,
                                    total=count,
                                    seconds=wall,
                                    psnr=score["psnr"],
                                )
                            ),
                            flush=True,
                        )
            finally:
                if ref is not None:
                    ref.close()
            atomic(folder / "frames.json", rows)
            np.savez_compressed(folder / "samples.npz", **previews)
            if method == "official":
                np.savez_compressed(
                    reference_file,
                    prediction=reference_images,
                    mask=reference_masks,
                    action=reference_actions,
                )
            warm = rows[2:]
            value = dict(
                method=method,
                mode=mode,
                case=name,
                frames=len(rows),
                setup_seconds=setup_seconds,
                cold_frame_s=rows[0]["model_wall_s"],
                first_warm_frame_s=rows[1]["model_wall_s"],
                warm=statistics([r["model_wall_s"] for r in warm]),
                complete_frames=statistics([r["full_frame_s"] for r in rows]),
                micro=micro,
                case_wall_including_micro_s=time.perf_counter() - case_start,
                sampled_peak_ram_rss=peak_ram,
                peak_vram=engine.peak() if engine is not None else None,
                quality={
                    k: float(np.mean([r[k] for r in rows]))
                    for k in ["mse", "psnr", "ssim", "mae", "unobserved_mae"]
                },
                observed_max_error=max(r["observed_max_error"] for r in rows),
                rng_protocol="JAX-native same seed"
                if method.startswith("jax") or method == "official"
                else "explicit NumPy noise for Torch; not cross-framework matched RNG",
                numerical_verdict="not_certified; benchmark outputs do not prove full-state equivalence",
                artifacts={p.name: sha(p) for p in folder.iterdir() if p.is_file() and p != marker},
            )
            atomic(marker, value)


def report(root):
    rows = [read(p) for p in root.glob("*/*/*/complete.json")]
    summary = []
    for method in read(root / "config.json")["methods"]:
        for mode in ["fixed", "closed_loop"]:
            group = [r for r in rows if r["method"] == method and r["mode"] == mode]
            if group:
                summary.append(
                    dict(
                        method=method,
                        mode=mode,
                        cases=len(group),
                        complete=len(group) == len(read(root / "manifest.json")["cases"]),
                        category=catalog()[method]
                        .get("profile", catalog()[method])
                        .get("category"),
                        frames=sum(r["frames"] for r in group),
                        mean_case_warm_s=float(np.mean([r["warm"]["mean_s"] for r in group])),
                        **{
                            k: float(np.mean([r["quality"][k] for r in group]))
                            for k in ["psnr", "ssim", "mae"]
                        },
                    )
                )
    atomic(
        root / "summary.json",
        dict(
            rows=summary,
            status=read(root / "status.json"),
            interpretation="Case-equal quality/time; same14 lines but different update counts. No automatic baseline adoption. Fixed observations do not imply matched latent state or cross-framework noise. BF comparisons require learned asset provenance.",
        ),
    )
    if summary:
        with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for ax, mode in zip(axes, ["fixed", "closed_loop"]):
            for row in summary:
                if row["mode"] == mode and row["complete"]:
                    ax.scatter(
                        row["mean_case_warm_s"] * 1000,
                        row["psnr"],
                        marker={"A": "o", "B": "^", "C": "s"}[row["category"]],
                    )
                    ax.annotate(
                        row["method"],
                        (row["mean_case_warm_s"] * 1000, row["psnr"]),
                        fontsize=7,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )
            ax.set(
                xscale="log",
                xlabel="Mean case warm loop time (ms; lower is faster)",
                ylabel="Float polar PSNR (dB)",
                title=mode,
            )
            ax.margins(x=0.3)
        if read(root / "config.json").get("functional_test_only"):
            fig.suptitle("SYNTHETIC CPU FUNCTIONAL TEST - NOT SCIENTIFIC RESULTS")
        fig.savefig(root / "speed_quality.png", dpi=180)
        plt.close(fig)


def run(path, root, resume=False):
    cfg = config(path)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        from .compute_lab.suite import stop_orphan

        stop_orphan(root)
        if resume and (root / "STOP").exists():
            (root / "STOP").unlink()
        source = {
            str(p.relative_to(ROOT)): sha(p)
            for sub in ["compute_lab", "torch_casl", "preparation", "fast_belief"]
            for p in (ROOT / "src/cognitive_ultrasound" / sub).glob("*.py")
        }
        source["speedbench"] = sha(__file__)
        assets = {}
        for label in ["checkpoint", "torch_export", "belief_checkpoint"]:
            p = Path(cfg[label]) if cfg.get(label) else None
            if p and p.exists():
                files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
                assets[label] = {str(f): sha(f) for f in files}
        identity = dict(
            config=cfg,
            source=source,
            assets=assets,
            hardware=hardware(),
            runtimes=runtime_versions(cfg["pythons"]),
        )
        if (root / "identity.json").exists():
            if read(root / "identity.json") != identity:
                raise ValueError("Code/config/assets changed; fresh output required")
            manifest = read(root / "manifest.json")
            for n, value in manifest["files"].items():
                if fingerprint(Path(cfg["data_root"]) / "val" / n) != value:
                    raise ValueError("Benchmark data changed")
        else:
            split = yaml.safe_load(Path(cfg["split_manifest"]).read_text(encoding="utf-8-sig"))
            names = sorted(split["val"])
            if len(names) != len(set(names)) or set(names) & (
                set(split["test"]) | set(split["train"])
            ):
                raise ValueError("Split overlap")
            if len(names) < cfg["cases"]:
                raise ValueError("Insufficient validation cases")
            selected = (
                np.random.default_rng(cfg["seed"]).permutation(names).tolist()[: cfg["cases"]]
            )
            if any(Path(n).name != n or "/" in n or "\\" in n for n in selected):
                raise ValueError("Invalid video filename")
            manifest = dict(
                cases=selected,
                files={n: fingerprint(Path(cfg["data_root"]) / "val" / n) for n in selected},
                split_sha256=sha(cfg["split_manifest"]),
                test_used=False,
            )
            atomic(root / "manifest.json", manifest)
            atomic(root / "identity.json", identity)
            atomic(root / "config.json", cfg)
            atomic(root / "catalog.json", catalog())
        if sha(cfg["split_manifest"]) != manifest["split_sha256"]:
            raise ValueError("Split changed")
        state = dict(
            status="running", pid=os.getpid(), created=psutil.Process().create_time(), methods={}
        )
        if (root / "status.json").exists():
            state["methods"] = read(root / "status.json").get("methods", {})
        for method in cfg["methods"]:
            if (root / "STOP").exists():
                state.update(status="stopped", current=None)
                atomic(root / "status.json", state)
                report(root)
                return
            if state["methods"].get(method, {}).get("status") == "completed":
                markers = list((root / method).glob("*/*/complete.json"))
                if len(markers) != len(manifest["cases"]) * len(cfg["modes"]):
                    raise ValueError("Completed method missing case receipts")
                for marker in markers:
                    for file, checksum in read(marker)["artifacts"].items():
                        if sha(marker.parent / file) != checksum:
                            raise ValueError("Completed artifact changed")
                continue
            disk_guard(root, cfg)
            state["current"] = method
            atomic(root / "status.json", state)
            backend = "torch" if method.startswith("torch") or method == "fast_belief" else "jax"
            python = cfg["pythons"][backend]
            command = [
                python,
                "-m",
                "cognitive_ultrasound.speedbench",
                "worker",
                "--output",
                str(root),
                "--method",
                method,
            ]
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT / "src"),
                PYTHONUTF8="1",
                OMP_NUM_THREADS=str(cfg["threads"]),
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                NVIDIA_TF32_OVERRIDE="0",
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
            )
            log = root / f"{method}.console.log"
            begin = time.perf_counter()
            with log.open("a", encoding="utf-8") as f:
                try:
                    child = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name != "nt",
                    )
                    proc = psutil.Process(child.pid)
                    state["worker_pid"] = child.pid
                    state["worker_created"] = proc.create_time()
                    atomic(root / "status.json", state)
                    deadline = time.monotonic() + cfg["timeout_seconds"]
                    while child.poll() is None:
                        if (root / "STOP").exists() or time.monotonic() > deadline:
                            for p in proc.children(recursive=True):
                                try:
                                    p.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            proc.kill()
                            child.wait()
                            raise TimeoutError("Stopped or bounded worker timeout")
                        time.sleep(0.3)
                    status = (
                        "completed"
                        if child.returncode == 0
                        else "unavailable"
                        if child.returncode == 3
                        else "failed"
                    )
                except (OSError, TimeoutError) as error:
                    f.write(str(error) + "\n")
                    status = "failed"
            state["methods"][method] = dict(
                status=status, elapsed_s=time.perf_counter() - begin, log=str(log)
            )
            atomic(root / "status.json", state)
            print(
                json.dumps(dict(event="METHOD_END", method=method, **state["methods"][method])),
                flush=True,
            )
        state.update(
            status="completed"
            if all(v["status"] == "completed" for v in state["methods"].values())
            else "finished_with_gaps",
            current=None,
        )
        if (root / "STOP").exists():
            state["status"] = "stopped"
        atomic(root / "status.json", state)
        report(root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "action", choices=["run", "worker", "report", "catalog", "status", "stop", "resume"]
    )
    p.add_argument("--config", default="configs/speedbench.yaml")
    p.add_argument("--output", default="/root/autodl-tmp/outputs_casl/speedbench_v1")
    p.add_argument("--method")
    a = p.parse_args()
    root = Path(a.output)
    if a.action == "catalog":
        print(json.dumps(catalog(), indent=2))
        return
    if a.action == "status":
        print(json.dumps(read(root / "status.json"), indent=2))
        return
    if a.action == "stop":
        root.joinpath("STOP").touch()
        return
    if a.action == "report":
        report(root)
        return
    if a.action == "worker":
        try:
            worker(read(root / "config.json"), a.method, root)
        except FileNotFoundError as error:
            atomic(
                root / f"{a.method}.unavailable.json", dict(status="unavailable", reason=str(error))
            )
            raise SystemExit(3) from error
        return
    run(a.config, root, resume=a.action == "resume")


if __name__ == "__main__":
    main()
