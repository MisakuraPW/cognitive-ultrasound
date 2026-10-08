#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1
task_python="${TORCH_PYTHON:-/root/miniconda3/bin/python}"
# Use existing Torch/CUDA. No framework or NVIDIA library replacement.
"$task_python" -c 'import torch, torchvision; from torchvision.models.video import r2plus1d_18; print(torch.__version__, torchvision.__version__)'
"$task_python" -m pip install 'numpy>=1.26,<3' 'PyYAML>=6' 'h5py>=3.11' 'scikit-image>=0.23' 'psutil>=5.9' 'opencv-python-headless>=4.8,<5'
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
python -c 'import jax, keras, h5py, cv2; print(jax.__version__)'
python -m pip check
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m cognitive_ultrasound.task_budget --help
