#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PREPARATION_MODULE=cognitive_ultrasound.torch_casl
export PREPARATION_CONFIG="${PREPARATION_CONFIG:-$PWD/configs/torch_casl_repair_v3.yaml}"
export PREPARATION_OUTPUT="${PREPARATION_OUTPUT:-/root/autodl-tmp/outputs_casl/torch_casl_v3}"
# Optional TORCH_PYTHON selects an existing GPU Torch environment (e.g. the image base env).
exec bash scripts/run_preparation.sh "${1:-start}"
