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
output="${2:-/root/autodl-tmp/outputs_casl/task_budget_value_v1}"
config="${TASK_VALUE_CONFIG:-configs/task_budget_value.yaml}"
extra=()
if [[ "${3:-}" == --expand16 ]]; then
  [[ "$output" != /root/autodl-tmp/outputs_casl/task_budget_value_v1 ]] || { printf 'Expansion requires a NEW output directory.\n'; exit 2; }
  extra+=(--expand16)
elif [[ -n "${3:-}" ]]; then
  printf 'Unsupported extra argument: %s\n' "$3"; exit 2
fi
case "$action" in
  start|resume|probe|calibrate)
    mkdir -p "$(dirname "$output")"
    live="$(python -m cognitive_ultrasound.task_budget value-status --output "$output")"
    if python -c 'import json,sys; sys.exit(0 if json.loads(sys.argv[1]).get("coordinator_alive") else 1)' "$live"; then
      printf 'Coordinator already running. Use status or tail %s.console.log\n' "$output"; exit 0
    fi
    command=run
    [[ "$action" != start ]] && command="$action"
    nohup python -m cognitive_ultrasound.task_budget "value-$command" --config "$config" --output "$output" "${extra[@]}" \
      >>"$output.console.log" 2>&1 < /dev/null &
    printf 'Coordinator PID: %s\nLog: %s.console.log\n' "$!" "$output"
    ;;
  status|stop|report|bundle|prepare)
    python -m cognitive_ultrasound.task_budget "value-$action" --config "$config" --output "$output" "${extra[@]}"
    ;;
  *) printf 'Use start/resume/probe/calibrate/status/stop/report/bundle/prepare\n'; exit 2 ;;
esac
