"""Frozen Torch EF evaluator in a persistent, separately configured process."""

import json
import os
import subprocess
import time
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import ROOT
from ..preparation.common import atomic_json, atomic_npz
from .protocol import causal_window, clip_indices, task_scores
from .exchange import write_exchange


class EFService:
    def __init__(self, cfg, root, coordinates):
        self.cfg, self.directory = cfg, root / ".ipc"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ipc_mode = cfg["runtime"].get("ipc_mode", "compressed")
        self.exchange_directory = self.directory
        if self.ipc_mode == "tmpfs":
            if not os.path.isdir("/dev/shm"):
                raise RuntimeError("tmpfs IPC requested but /dev/shm unavailable")
            self.exchange_directory = type(self.directory)(tempfile.mkdtemp(prefix="casl-ef-", dir="/dev/shm"))
        self.config_file = self.directory / "config.json"
        atomic_json(self.config_file, cfg)
        atomic_npz(self.directory / "coordinates.npz", coordinates=coordinates)
        self.log = (root / "ef_service.log").open("a", encoding="utf-8")
        self.pool = ThreadPoolExecutor(max_workers=1)
        env = dict(
            os.environ,
            PYTHONUTF8="1",
            PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
        self.process = subprocess.Popen(
            [
                cfg["torch_python"],
                "-u",
                "-m",
                "cognitive_ultrasound.task_budget.ef_worker",
                "serve",
                "--config",
                str(self.config_file),
                "--coordinates",
                str(self.directory / "coordinates.npz"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            env=env,
        )
        try:
            ready = self._line()
            if not ready.get("ready"):
                raise RuntimeError(f"EF worker failed: {ready}")
            atomic_json(root / "ef_runtime.json", ready)
        except BaseException:
            self.close()
            raise
        self.calls = 0
        self.seconds = 0.0

    def _line(self):
        line = self.pool.submit(self.process.stdout.readline).result(
            timeout=self.cfg["runtime"]["ef_request_timeout_seconds"]
        )
        if not line:
            raise RuntimeError(f"EF subprocess exited {self.process.poll()}; see ef_service.log")
        value = json.loads(line)
        if value.get("error"):
            raise RuntimeError(value["error"])
        return value

    def request(self, clips, gradient=False, domain="polar"):
        start = time.perf_counter()
        write_exchange(self.exchange_directory / "input.npz", dict(clips=np.asarray(clips, np.float32)), self.ipc_mode)
        self.process.stdin.write(
            json.dumps(
                dict(
                    op="evaluate",
                    input=str(self.exchange_directory / "input.npz"),
                    output=str(self.exchange_directory / "output.npz"),
                    ipc_mode=self.ipc_mode,
                    gradient=gradient,
                    domain=domain,
                )
            )
            + "\n"
        )
        self.process.stdin.flush()
        self._line()
        with np.load(self.exchange_directory / "output.npz", allow_pickle=False) as data:
            result = {k: data[k].copy() for k in data.files}
        self.calls += 1
        self.seconds += time.perf_counter() - start
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError("Nonfinite EF output/input gradient")
        return result

    def score(self, particles, history):
        task = self.cfg["task"]
        clips = np.stack(
            [causal_window(history, p, task["frames"], task["period"]) for p in particles]
        )
        result = self.request(clips, gradient=True)
        # With no history, left padding repeats the CURRENT hypothesis in every
        # temporal slot: all corresponding input derivatives must be accumulated.
        current_gradient = (
            result["gradients"].sum(axis=1) if not history else result["gradients"][:, -1]
        )
        scores = task_scores(particles, current_gradient)
        return scores, result["predictions"]

    def video(self, images, gradient=False, domain="polar"):
        task = self.cfg["task"]
        indices = clip_indices(len(images), task["frames"], task["period"], task["stride"])
        predictions, grads = [], np.zeros_like(images, np.float32)
        # One clip per request bounds EF autograd memory for complete videos.
        for ids in indices:
            result = self.request(images[ids][None], gradient=gradient, domain=domain)
            predictions.append(float(result["predictions"][0]))
            if gradient:
                np.add.at(grads, ids, result["gradients"][0] / len(indices))
        return float(np.mean(predictions)), grads, predictions

    def close(self):
        if hasattr(self, "process"):
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"op":"close"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
        if hasattr(self, "pool"):
            self.pool.shutdown(wait=False, cancel_futures=True)
        if getattr(self, "ipc_mode", "") == "tmpfs":
            shutil.rmtree(self.exchange_directory, ignore_errors=True)
        if hasattr(self, "log"):
            self.log.close()
