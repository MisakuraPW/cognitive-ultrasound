"""Exercise sequence receipts, common-state replay and resume without claiming GPU evidence."""

from dataclasses import asdict

import h5py
import numpy as np
import pytest

from cognitive_ultrasound.compute_lab.protocol import Profile
from cognitive_ultrasound.preparation.common import atomic_json, read_json


class Engine:
    calls = 0

    def __init__(self, cfg, profile, budget, export):
        self.profile = profile
        self.probes = dict(passed=True)

    def operator_checks(self, output):
        return self.probes

    def reset(self, seed, initial=None):
        self.state = dict(
            resume_buffer=np.zeros((112, 112, 3), np.float32),
            resume_mask=np.ones((112, 112, 3), np.float32),
        )

    def snapshot(self):
        return {k: v.copy() for k, v in self.state.items()}

    def restore(self, arrays):
        self.state = {k: v.copy() for k, v in arrays.items() if k.startswith("resume_")}

    def capture(self):
        return self.snapshot()

    def reinstate(self, state):
        self.restore(state)

    def noise(self):
        return np.zeros((2, 112, 112, 3), np.float32)

    def step(self, target, noise):
        Engine.calls += 1
        self.state["resume_buffer"] = np.concatenate(
            (self.state["resume_buffer"][..., 1:], target), -1
        )
        self.state["resume_posterior_samples"] = np.repeat(self.state["resume_buffer"][None], 2, 0)
        result = dict(
            prediction=target * 0.9,
            mask=np.ones_like(target),
            uncertainty=np.zeros((112, 112), np.float32),
            next_action=np.ones(112, np.float32),
        )
        result.update(self.snapshot())
        return dict(core_s=0.1, closed_loop_s=0.2), result

    def peak(self):
        return 1024


def test_sequence_worker_micro_replay_and_case_boundary_resume(tmp_path, monkeypatch):
    from cognitive_ultrasound.compute_lab import engines, inference

    name = "0X123.hdf5"
    (tmp_path / "data/val").mkdir(parents=True)
    with h5py.File(tmp_path / "data/val" / name, "w") as h:
        h.create_dataset("data/image", data=np.full((4, 112, 112), -12.0, np.float32))
    atomic_json(
        tmp_path / "manifest.json",
        dict(cohorts=dict(debug=[name]), files={name: dict(frames=4)}, seeds=dict(debug=[42])),
    )
    monkeypatch.setattr(engines, "JaxEngine", Engine)

    def replay(engine, cfg, profile, budget, previous, target, noise):
        engine.restore(previous)
        result = engine.step(target, noise)[1]
        return {
            k: result[k]
            for k in ("prediction", "uncertainty", "next_action", "resume_posterior_samples")
        }

    monkeypatch.setattr(inference, "explicit_replay", replay)
    cfg = dict(data_root=str(tmp_path / "data"))

    def execute(profile, folder):
        output = tmp_path / folder
        output.mkdir(exist_ok=True)
        inference.run(
            dict(profile=asdict(profile), cohort="debug", budget=14, short=True),
            cfg,
            tmp_path,
            output,
        )
        return read_json(output / "result.json")

    official = execute(Profile("official"), "official")
    candidate = execute(Profile("jax_prefetch", io="prefetch", async_output=True), "candidate")
    assert len(official["rows"]) == len(candidate["rows"]) == 4
    assert len(candidate["micro"]) == 20 and candidate["equivalence_passed"]
    assert candidate["replay"][0]["checks"]["prediction"]["passed"]
    before = Engine.calls
    execute(Profile("jax_prefetch", io="prefetch"), "candidate")
    assert Engine.calls == before  # case receipt prevents repeated compute


def test_cuda_capture_changed_inputs_gradient_and_no_output_alias():
    import torch

    from cognitive_ultrasound.torch_casl.native import CapturedFrame

    if not torch.cuda.is_available():
        pytest.skip("Requires GPU; remains a runtime acceptance test")

    class Toy:
        def frame(self, a, b, c, d, steps=50):
            gradient = torch.func.grad(lambda x: ((x + b + c) * d).square().sum())(a)
            return gradient, a + b, c * d, a.square()

    args = tuple(torch.ones((2, 3, 8, 8), device="cuda") * i for i in (1, 2, 3, 4))
    model = Toy()
    captured = CapturedFrame(model, args, 50)
    old = captured(*args)
    owned = [x.clone() for x in old]
    changed = (args[0] * 0.3, args[1] * 0.7, args[2] * 0.2, args[3] * 0.9)
    actual, expected = captured(*changed), model.frame(*changed)
    torch.cuda.synchronize()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for a, b in zip(old, owned):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["stop", "oom_exit"])
def test_stop_reaps_worker_tree(tmp_path, monkeypatch, reason):
    import subprocess
    import sys
    import threading
    import time

    import psutil

    from cognitive_ultrasound.compute_lab import suite

    # Real process tree, simulated GPU worker: STOP must reap both parent and descendant.
    real_popen = subprocess.Popen
    pidfile = tmp_path / "child.pid"
    script = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(40)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(40)"
    if reason == "oom_exit":
        script = script.replace("time.sleep(40)", "time.sleep(40)", 1)
        script = script.rsplit("; time.sleep(40)", 1)[0] + "; time.sleep(1.5); sys.exit(137)"

    def launch(*args, **kwargs):
        return real_popen([sys.executable, "-X", "utf8", "-c", script, str(pidfile)], **kwargs)

    monkeypatch.setattr(suite.subprocess, "Popen", launch)
    root = tmp_path / "run"
    root.mkdir()
    coordinator = suite.Suite(dict(pythons=dict(jax=sys.executable), threads=1), root)

    def stop():
        for _ in range(100):
            if pidfile.exists():
                break
            time.sleep(0.05)
        (root / "STOP").touch()

    if reason == "stop":
        thread = threading.Thread(target=stop)
        thread.start()
        with pytest.raises(suite.Stopped):
            coordinator.job("running", {})
        thread.join()
    else:
        result = coordinator.job("running", {})
        assert result["status"] == "failed" and result["returncode"] == 137
    pid = int(pidfile.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
