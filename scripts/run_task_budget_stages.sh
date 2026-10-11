#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1
export KERAS_BACKEND=jax NVIDIA_TF32_OVERRIDE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset LD_LIBRARY_PATH
action="${1:-start}"
output="${2:-/root/autodl-tmp/outputs_casl/task_budget_stages_v1}"
config="${TASK_STAGES_CONFIG:-configs/task_budget_stages.yaml}"
if [[ "$action" == start || "$action" == resume ]]; then
  if [[ -f "$output/scope_closure_136.json" && "${TASK_STAGES_RESUME_ORIGINAL:-0}" != 1 ]]; then
    printf 'This batch was closed at136 by user request. Use a new output for a new experiment.\n'
    printf 'Only an explicit decision to resume the ORIGINAL264 plan may set TASK_STAGES_RESUME_ORIGINAL=1.\n'
    exit 2
  fi
fi
case "$action" in
  start|resume|probe|calibrate)
    mkdir -p "$(dirname "$output")"
    live="$(python -m cognitive_ultrasound.task_budget stage-status --output "$output")"
    if python -c 'import json,sys; sys.exit(0 if json.loads(sys.argv[1]).get("coordinator_alive") else 1)' "$live"; then
      printf 'Already running. Use status or tail %s.console.log\n' "$output"; exit 0
    fi
    command=run
    [[ "$action" != start ]] && command="$action"
    nohup python -m cognitive_ultrasound.task_budget "stage-$command" --config "$config" --output "$output" \
      >>"$output.console.log" 2>&1 < /dev/null &
    printf 'Coordinator PID: %s\nLog: %s.console.log\n' "$!" "$output"
    ;;
  status|stop|report|bundle|prepare)
    python -m cognitive_ultrasound.task_budget "stage-$action" --config "$config" --output "$output"
    ;;
  *) printf 'Use start/resume/probe/calibrate/status/stop/report/bundle/prepare\n'; exit 2 ;;
esac
