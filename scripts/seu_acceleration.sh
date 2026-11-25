#!/usr/bin/env bash
set -euo pipefail

task=$1
root=$2
data=$3
jax_python=$4
# The first rollout's already submitted job wrote its full revision to this log.
# Later jobs pass the revision directly. Pin the code before loading any modules.
revision=${5:-}
if [[ -z $revision ]]; then
    revision=$(sed -n 's/^submitted_repo_rev=//p' "$root/jobs/acceleration/casl-$task-$SLURM_JOB_ID.log")
fi
[[ $revision =~ ^[a-f0-9]{40}$ ]] || { echo 'Missing full submission revision' >&2; exit 2; }
checkout="$root/runtime/acceleration-revisions/$revision"
if [[ $(pwd -P) != "$checkout" ]]; then
    repo=$(pwd -P)
    mkdir -p -- "${checkout%/*}"
    (
        flock -x 9
        if [[ ! -d $checkout ]]; then
            git worktree add --detach "$checkout" "$revision"
        fi
        [[ $(git -C "$checkout" rev-parse --verify HEAD) == "$revision" ]]
        if [[ ! -L $checkout/vendor/casl ]]; then
            if [[ -d $checkout/vendor/casl ]]; then rmdir "$checkout/vendor/casl"; fi
            ln -s "$repo/vendor/casl" "$checkout/vendor/casl"
        fi
        if [[ ! -L $checkout/checkpoints/official ]]; then
            ln -s "$repo/checkpoints/official" "$checkout/checkpoints/official"
        fi
    ) 9>"$checkout.lock"
    cd -- "$checkout"
    export PYTHONPATH="$checkout/src"
    printf 'pinned_acceleration_rev=%s\n' "$revision"
    exec bash scripts/seu_acceleration.sh "$@"
fi
out="$root/jobs/acceleration/$task-$SLURM_JOB_ID"
if [[ $task == *-resume-* ]]; then
    [[ $task =~ ^(seqdiff|seqdiff-core|bf-core)-resume-([1-9][0-9]*)$ ]]
    original=${BASH_REMATCH[1]}
    prior=${BASH_REMATCH[2]}
    if [[ $original == seqdiff ]]; then original=seqdiff-train; fi
    out="$root/jobs/acceleration/$original-$prior"
    [[ $(cat "$out/revision.txt") == "$revision" ]]
else
    mkdir -p -- "$out"
fi
exec 8>"$out/.job.lock"
flock -n 8
benchmark="$out/benchmark"
if [[ $task == *-resume-* && -d $benchmark ]]; then
    # Benchmark identity includes the physical GPU. Keep the old attempt intact.
    benchmark="$out/benchmark-$SLURM_JOB_ID"
fi
printf 'experiment_output=%s\n' "$out"
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8 JAX_PLATFORMS=cuda
export KERAS_BACKEND=jax

torch_python=$jax_python
if [[ $task == bf-* || $task == accel-compare || $task == accel-core-test ]]; then
    torch_python=
    for candidate in "${CASL_SEU_TORCH_PYTHON:-}" "$root/envs/fast-belief/bin/python" "$jax_python"; do
        [[ -n $candidate && -x $candidate ]] || continue
        if env -u LD_LIBRARY_PATH "$candidate" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.backends.cudnn.version()); x=torch.ones(16,16,device="cuda"); assert (x@x)[0,0].item()==16'; then
            torch_python=$candidate
            break
        fi
    done
    if [[ -z $torch_python ]]; then
        printf 'No CUDA Torch found. Set CASL_SEU_TORCH_PYTHON to a compatible user environment.\n' >&2
        module avail 2>&1 | awk '/[Tt]orch|[Cc]ontainer|[Aa]pptainer/{print}' || true
        exit 2
    fi
fi

if [[ $task != *-resume-* ]]; then
    printf '%s\n' "$revision" > "$out/revision.txt"
    "$jax_python" -u scripts/seu_acceleration.py configure "$task" "$out" "$data" "$jax_python" "$torch_python"
fi
case $task in
    bf-*)
        bf_action=run
        if [[ $task == *-resume-* ]]; then bf_action=resume; fi
        env -u LD_LIBRARY_PATH "$torch_python" -u -m cognitive_ultrasound.fast_belief "$bf_action" --config "$out/training.json" --output "$out/belief"
        ;;
    seqdiff-*) "$jax_python" -u scripts/seu_acceleration.py train-vivit "$out" ;;
    accel-compare|accel-core-test) ;;
esac
if [[ $task == bf-core || $task == seqdiff-core || $task == *-core-resume-* ]]; then
    if [[ $task == bf-* ]]; then stage=belief; else stage=vivit; fi
    test -s "$out/$stage/complete.json"
    printf 'SEU_CORE_TRAIN_OK %s\n' "$out"
    exit 0
fi
"$jax_python" -u -m cognitive_ultrasound.speedbench run --config "$out/benchmark.json" --output "$benchmark"
"$jax_python" -u scripts/seu_acceleration.py summarize "$out" --benchmark "$benchmark"
