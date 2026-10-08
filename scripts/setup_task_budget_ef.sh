#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1
task_python="${TORCH_PYTHON:-/root/miniconda3/bin/python}"
# Use the image's existing Torch/CUDA. No framework or NVIDIA library replacement.
"$task_python" -c 'import torch, torchvision; from torchvision.models.video import r2plus1d_18; print(torch.__version__, torchvision.__version__); assert torch.cuda.is_available(), "PyTorch CUDA is unavailable"'
"$task_python" -m pip install 'numpy>=1.26,<3' 'PyYAML>=6' 'h5py>=3.11' 'scikit-image>=0.23' 'psutil>=5.9' 'opencv-python-headless>=4.8,<5'
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
# Original-video reference evaluation also decodes AVI in the JAX worker.
# Pin just the CPU OpenCV wheel; do not let this step change NumPy/CUDA packages.
python -m pip install --no-deps 'opencv-python-headless==4.12.0.88'
# JAX's pip CUDA wheels provide their own libraries.
unset LD_LIBRARY_PATH
export KERAS_BACKEND=jax XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import jax, keras, h5py, cv2; print(jax.__version__, jax.devices(), keras.backend.backend(), cv2.__version__); assert jax.default_backend() == "gpu", "JAX GPU is unavailable"'
python -m pip check
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m cognitive_ultrasound.task_budget --help
