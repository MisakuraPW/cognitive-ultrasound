#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
rm -f .cache/compute-ready.json
if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo >/dev/null 2>&1
fi
torch_python="${TORCH_PYTHON:-/root/miniconda3/bin/python}"
"$torch_python" -c 'import torch; assert hasattr(torch, "compile") and hasattr(torch, "func"); print(torch.__version__)'
# Avoid inherited HTTP mirror/proxy failures; scoped to this installation only.
(
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  "$torch_python" -m pip install --index-url "${COMPUTE_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" -r requirements/compute-torch.txt
)
PYTHONPATH= "$torch_python" -m pip check
/root/miniconda3/envs/casl/bin/python -m pip check
"$torch_python" -m cognitive_ultrasound.compute_lab.readiness write
echo 'Existing CUDA/Torch/JAX/TensorFlow retained. Next: bash scripts/run_compute_lab.sh start'
