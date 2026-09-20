#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR="$PWD"
OLD_REPO="${CASL_EXISTING_REPO:-/root/autodl-tmp/cognitive-ultrasound}"
OUTPUT="${PREPARATION_OUTPUT:-/root/autodl-tmp/outputs_casl/preparation_auto_v2}"
CONFIG="${PREPARATION_CONFIG:-$ROOT_DIR/configs/preparation_auto.yaml}"
MODE="${1:-start}"
MODULE="${PREPARATION_MODULE:-cognitive_ultrasound.preparation}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
export PYTHONPATH="$ROOT_DIR/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
for item in vendor checkpoints; do
    if [[ ! -e "$ROOT_DIR/$item" ]]; then
        test -d "$OLD_REPO/$item" || { echo "Missing $OLD_REPO/$item"; exit 2; }
        ln -s "$OLD_REPO/$item" "$ROOT_DIR/$item"
    fi
done
LOG="${OUTPUT}.console.log"
mkdir -p "$(dirname "$OUTPUT")"
case "$MODE" in
    plan) python -m "$MODULE" plan --config "$CONFIG" --output "$OUTPUT" ;;
    stop) mkdir -p "$OUTPUT"; touch "$OUTPUT/STOP"; echo "Stop requested. Completed frames/checkpoints retained." ;;
    status) test -f "$OUTPUT/status.json" && cat "$OUTPUT/status.json" ;;
    report) python -m "$MODULE" report --output "$OUTPUT" ;;
    foreground)
        python scripts/check_gpu.py --output "${OUTPUT}.gpu.json"
        ARGS=()
        if [[ -f "$OUTPUT/identity.json" ]]; then ARGS+=(--resume); fi
        if [[ "${PREPARATION_RETRY_FAILED:-0}" == 1 ]]; then ARGS+=(--retry-failed); fi
        python -m "$MODULE" run --config "$CONFIG" --output "$OUTPUT" "${ARGS[@]}"
        ;;
    start|resume)
        if [[ -f "$OUTPUT/STOP" ]]; then
            echo "Run is explicitly paused. Remove only $OUTPUT/STOP before resuming."; exit 2
        fi
        nohup bash "$ROOT_DIR/scripts/run_preparation.sh" foreground >> "$LOG" 2>&1 < /dev/null &
        echo "Coordinator launch PID: $! (output lock prevents duplicate experiments)"
        echo "Overall: tail -n 60 -F '$LOG'"
        echo "Per task: tail -n 60 -F '$OUTPUT/jobs/<job>/console.log'"
        echo "No automatic machine shutdown; AutoDL billing continues after completion."
        ;;
    *) echo 'Use start | resume | status | stop | plan | report'; exit 2 ;;
esac
