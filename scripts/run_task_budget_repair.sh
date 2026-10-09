#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUTF8=1
export KERAS_BACKEND=jax NVIDIA_TF32_OVERRIDE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset LD_LIBRARY_PATH
output="${2:-/root/autodl-tmp/outputs_casl/task_budget_ef_repair_v4}"
source_batch="${TASK_BUDGET_SOURCE:-/root/autodl-tmp/outputs_casl/task_budget_ef_v3}"
engineering="${TASK_BUDGET_ENGINEERING:-/root/autodl-tmp/outputs_casl/task_budget_engineering_v1}"
action="${1:-start}"
if [[ "$action" == start || "$action" == resume ]]; then
  python scripts/prepare_task_budget_repair.py --source "$source_batch" --engineering "$engineering" --output "$output"
fi
TASK_BUDGET_CONFIG="$output/config.json" bash scripts/run_task_budget_ef.sh "$action" "$output"
