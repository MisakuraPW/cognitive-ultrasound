#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
action="${1:-start}"
output="${2:-/root/autodl-tmp/outputs_casl/speedbench_v1}"
python="${BENCH_PYTHON:-/root/miniconda3/envs/casl/bin/python}"
config="${BENCH_CONFIG:-configs/speedbench.yaml}"
case "$action" in
 start|resume)
  mkdir -p "$(dirname "$output")"
  command=run; [[ "$action" == resume ]] && command=resume
  nohup "$python" -u -m cognitive_ultrasound.speedbench "$command" --config "$config" --output "$output" >>"$output.console.log" 2>&1 </dev/null &
  printf 'PID=%s\nLog=%s.console.log\n' "$!" "$output" ;;
 status|stop|report|catalog)
  "$python" -m cognitive_ultrasound.speedbench "$action" --output "$output" ;;
 *) exit 2 ;;
esac
