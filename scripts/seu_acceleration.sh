#!/usr/bin/env bash
set -euo pipefail

task=$1
root=$2
data=$3
jax_python=$4
out="$root/jobs/acceleration/$task-$SLURM_JOB_ID"
mkdir -p -- "$out"
printf 'experiment_output=%s\n' "$out"
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8 JAX_PLATFORMS=cuda

torch_python=
for candidate in "${CASL_SEU_TORCH_PYTHON:-}" "$root/envs/fast-belief/bin/python" "$jax_python"; do
    [[ -n $candidate && -x $candidate ]] || continue
    if "$candidate" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.ones(16,16,device="cuda"); assert (x@x)[0,0].item()==16'; then
        torch_python=$candidate
        break
    fi
done
if [[ -z $torch_python ]]; then
    printf 'No CUDA Torch found. Set CASL_SEU_TORCH_PYTHON to a compatible user environment.\n' >&2
    module avail 2>&1 | awk '/[Tt]orch|[Cc]ontainer|[Aa]pptainer/{print}' || true
    exit 2
fi

"$jax_python" -u scripts/seu_acceleration.py configure "$task" "$out" "$data" "$jax_python" "$torch_python"
"$torch_python" -u -m cognitive_ultrasound.fast_belief run --config "$out/training.json" --output "$out/belief"
"$jax_python" -u -m cognitive_ultrasound.speedbench run --config "$out/benchmark.json" --output "$out/benchmark"
"$jax_python" -u scripts/seu_acceleration.py summarize "$out"
