#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PREPARATION_MODULE=cognitive_ultrasound.preparation.closure
export PREPARATION_CONFIG="${PREPARATION_CONFIG:-$PWD/configs/preparation_closure.yaml}"
export PREPARATION_OUTPUT="${PREPARATION_OUTPUT:-/root/autodl-tmp/outputs_casl/preparation_closure_v4}"
exec bash scripts/run_preparation.sh "${1:-start}"
