#!/usr/bin/env bash
# Dedicated PyTorch environment. Never activates casl/JAX or installs CUDA wheels.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action="${1:-start}"
output="${2:-/root/autodl-tmp/outputs_casl/fast_belief_v2}"
config="${FAST_BELIEF_CONFIG:-configs/fast_belief.yaml}"
env_path="${FAST_BELIEF_ENV:-/root/autodl-tmp/venvs/fast-belief}"
base="${FAST_BELIEF_BASE_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${MPLCONFIGDIR:-/root/autodl-tmp/.cache/fast-belief-matplotlib}"
setup() {
  "$base" -c 'import torch; assert torch.cuda.is_available(), "Select a CUDA PyTorch base image"; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())'
  if [[ ! -x "$env_path/bin/python" ]]; then "$base" -m venv --system-site-packages "$env_path"; fi
  "$env_path/bin/python" -m pip install -r requirements-fast-belief.txt
  "$env_path/bin/python" -c 'import torch,h5py,yaml; assert torch.cuda.is_available(); assert tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) >= (2,7); print("FAST_BELIEF_ENV_READY")'
}
if [[ "$action" == setup ]]; then setup; exit 0; fi
if [[ ! -x "$env_path/bin/python" ]]; then setup; fi
python="$env_path/bin/python"
case "$action" in
  start|resume|probe)
    mkdir -p "$(dirname "$output")"
    live="$("$python" -m cognitive_ultrasound.fast_belief status --output "$output")"
    if "$python" -c 'import json,sys;sys.exit(0 if json.loads(sys.argv[1]).get("alive") else 1)' "$live"; then
      printf 'Already running. Log: %s.console.log\n' "$output"; exit 0
    fi
    command=run
    [[ "$action" != start ]] && command="$action"
    nohup "$python" -u -m cognitive_ultrasound.fast_belief "$command" --config "$config" --output "$output" \
      >>"$output.console.log" 2>&1 </dev/null &
    printf 'Launched PID: %s\nLog: %s.console.log\nStatus: bash scripts/run_fast_belief.sh status %q\n' "$!" "$output" "$output"
    ;;
  status|stop|report)
    "$python" -m cognitive_ultrasound.fast_belief "$action" --config "$config" --output "$output"
    ;;
  *) printf 'Use setup/start/resume/probe/status/stop/report\n'; exit 2 ;;
esac
