#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 KERAS_BACKEND=jax
export NVIDIA_TF32_OVERRIDE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
action="${1:-start}"
output="${2:-/root/autodl-tmp/outputs_casl/task_budget_ef_v1}"
config="${TASK_BUDGET_CONFIG:-configs/task_budget_ef.yaml}"
if [[ "$action" == start || "$action" == resume ]]; then
  mkdir -p "$(dirname "$output")"
  command=run
  [[ "$action" == resume ]] && command=resume
  nohup python -m cognitive_ultrasound.task_budget "$command" --config "$config" --output "$output" \
    >>"$output.console.log" 2>&1 < /dev/null &
  printf 'Coordinator PID: %s\nLog: %s.console.log\n' "$!" "$output"
else
  python -m cognitive_ultrasound.task_budget "$action" --config "$config" --output "$output"
fi
