#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUTPUT="${ALL_PREPARATION_OUTPUT:-/root/autodl-tmp/outputs_casl/all_preparation_v2}"
CONFIG="${ALL_PREPARATION_CONFIG:-$PWD/configs/all_preparation.yaml}"
MODE="${1:-start}"
mkdir -p "$OUTPUT"
case "$MODE" in
    stop) touch "$OUTPUT/STOP"; echo 'Pause requested; wait for the active suite to stop.'; exit 0 ;;
    status) test -f "$OUTPUT/status.json" && cat "$OUTPUT/status.json"; exit 0 ;;
esac
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
ARGS=(--config "$CONFIG" --output "$OUTPUT")
case "$MODE" in
    plan|report) python -m cognitive_ultrasound.all_preparation "$MODE" "${ARGS[@]}" ;;
    foreground) python -m cognitive_ultrasound.all_preparation run "${ARGS[@]}" ;;
    start|resume)
        if [[ "$MODE" == resume ]]; then ARGS+=(--resume); fi
        if [[ "$MODE" == start && -f "$OUTPUT/STOP" ]]; then
            echo 'Explicitly paused; use bash scripts/run_all_preparation.sh resume'; exit 2
        fi
        nohup python -m cognitive_ultrasound.all_preparation run "${ARGS[@]}" >> "${OUTPUT}.console.log" 2>&1 < /dev/null &
        echo "Coordinator launch PID: $! (locks reject duplicate/overlapping suites)"
        echo "tail -n 60 -F '${OUTPUT}.console.log'"
        echo 'Existing completed suites are skipped. No automatic shutdown.'
        ;;
    *) echo 'Use start | resume | status | stop | plan | report | foreground'; exit 2 ;;
esac
