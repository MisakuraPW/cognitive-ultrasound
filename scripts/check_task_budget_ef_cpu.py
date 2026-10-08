"""Real CASL weights / one-step CPU VJP functional check, not an EF efficacy experiment."""

import argparse
import os
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from cognitive_ultrasound.config import ROOT, load
    from cognitive_ultrasound.preparation.common import atomic_json
    from cognitive_ultrasound.provenance import sha256
    from cognitive_ultrasound.task_budget.episode import gs_gradient, gs_local_objective
    from cognitive_ultrasound.task_budget.perception import CASLPerception
    from cognitive_ultrasound.task_budget.policy import initialize
    from cognitive_ultrasound.task_budget.protocol import mask_bank

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/official"))
    parser.add_argument(
        "--output", default=str(ROOT / "outputs/task_budget_local_validation/official_cpu.json")
    )
    args = parser.parse_args()
    cfg = load(ROOT / "configs/task_budget_ef.yaml")
    cfg["checkpoint"] = str(Path(args.checkpoint).resolve())
    cfg["perception"]["warm_steps"] = 1
    before = sha256(Path(cfg["checkpoint"]) / "model.weights.h5")
    start = time.perf_counter()
    perception = CASLPerception(cfg, cpu_functional_test=True)
    key = jax.random.PRNGKey(42)
    frame = jax.random.uniform(key, (112, 112), minval=-0.5, maxval=0.5)
    mask = jnp.zeros((112, 112)).at[:, [5, 20, 40, 60]].set(1.0)
    previous = jax.random.normal(key, (2, 112, 112, 3)) * 0.1

    def objective(m):
        history = jnp.zeros((112, 112, 3)).at[..., -1].set(frame * m)
        masks = jnp.zeros_like(history).at[..., -1].set(m)
        return jnp.mean(perception.infer(history, masks, previous, key)[0, ..., -1])

    value, gradient = jax.value_and_grad(objective)(mask)
    if not np.isfinite(np.asarray(gradient)).all() or float(jnp.linalg.norm(gradient)) <= 0:
        raise AssertionError("Real CASL mask VJP failed")
    params = initialize(42, cfg)
    context = dict(
        cold=False,
        history=np.zeros((112, 112, 3), np.float32),
        masks=np.zeros((112, 112, 3), np.float32),
        previous=np.asarray(previous),
        key0=key,
        key1=jax.random.fold_in(key, 1),
        target=np.asarray(frame),
        state0=np.ones(12, np.float32),
        state1=np.ones(12, np.float32),
        legal0=np.ones(4, bool),
        legal1=np.ones(5, bool),
        action0=2,
        action1=2,
        noise0=np.zeros(4, np.float32),
        noise1=np.zeros(5, np.float32),
        bank0=mask_bank(np.arange(14), cfg["budgets"]["first"]),
        bank1=mask_bank(np.arange(14, 28), cfg["budgets"]["second"]),
        k1=10,
        k2=4,
    )
    adjoint = np.full((112, 112), 1 / (112 * 112), np.float32)
    _, pred = gs_local_objective(params, context, adjoint, perception, cfg, 1.0, 0.0, 1)
    context["prediction"] = np.asarray(pred)
    _, gate = gs_gradient(params, [context], [adjoint], perception, cfg, 1.0, 0.0)
    if not all(x > 0 for x in gate["task_gradient_norms"].values()):
        raise AssertionError("Task adjoint cannot reach both heads")
    if sha256(Path(cfg["checkpoint"]) / "model.weights.h5") != before:
        raise AssertionError("Checkpoint modified")
    record = dict(
        kind="FUNCTIONAL_CPU_ONLY_NOT_SCIENTIFIC_OR_GPU",
        official_weight_sha256=before,
        value=float(value),
        mask_gradient_norm=float(jnp.linalg.norm(gradient)),
        warm_steps=1,
        production_warm_steps=25,
        gs_two_stage_gate=gate,
        seconds=time.perf_counter() - start,
        task_adjoint="synthetic differentiable task; frozen real CASL weights; no EF quality claim",
    )
    atomic_json(Path(args.output), record)
    print(record, flush=True)


if __name__ == "__main__":
    main()
