#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/usr/local/cuda/bin:$PATH"
export NVIDIA_TF32_OVERRIDE=0 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
action="${1:-start}"
output="${2:-/root/autodl-tmp/outputs_casl/compute_lab_v1}"
config="${COMPUTE_LAB_CONFIG:-configs/compute_lab.yaml}"
phase="${COMPUTE_LAB_PHASE:-all}"
if [[ "$action" == start || "$action" == resume ]]; then
  # Reproduce the recorded runtime search path, independent of the caller's shell.
  # Identity checking remains strict; do not accumulate CUDA entries on each resume.
  probe_source="${COMPUTE_LAB_INHERIT:-$output}"
  if [[ -f "$probe_source/probe/probe.json" ]]; then
    export LD_LIBRARY_PATH="$("${TORCH_PYTHON:-/root/miniconda3/bin/python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["identity"].get("cuda_path") or "")' "$probe_source/probe/probe.json")"
  fi
  "${TORCH_PYTHON:-/root/miniconda3/bin/python}" -m cognitive_ultrasound.compute_lab.readiness check
  python -m pip check
  mkdir -p "$(dirname "$output")"
  command=run
  [[ "$action" == resume ]] && command=resume
  options=(--phase "$phase")
  [[ -z "${COMPUTE_LAB_INHERIT:-}" ]] || options+=(--inherit-from "$COMPUTE_LAB_INHERIT")
  nohup python -m cognitive_ultrasound.compute_lab "$command" --config "$config" --output "$output" "${options[@]}" >>"$output.console.log" 2>&1 < /dev/null &
  printf 'Coordinator PID: %s\nLog: %s.console.log\n' "$!" "$output"
else
  python -m cognitive_ultrasound.compute_lab "$action" --config "$config" --output "$output"
fi
