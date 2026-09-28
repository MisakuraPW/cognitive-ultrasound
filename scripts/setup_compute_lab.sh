#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo >/dev/null 2>&1
fi
torch_python="${TORCH_PYTHON:-/root/miniconda3/bin/python}"
"$torch_python" -c 'import torch; assert hasattr(torch, "compile") and hasattr(torch, "func"); print(torch.__version__)'
"$torch_python" -m pip install -r requirements/compute-torch.txt
"$torch_python" -m pip check
/root/miniconda3/envs/casl/bin/python -m pip check
echo 'Existing CUDA/Torch/JAX/TensorFlow retained. Next: bash scripts/run_compute_lab.sh start'
