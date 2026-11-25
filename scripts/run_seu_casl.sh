#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "${BASH_SOURCE[0]%/*}/.." && pwd -P)
root=${CASL_SEU_ROOT:-$HOME/casl}

usage() {
    printf '%s\n' \
        'Usage:' \
        '  bash scripts/run_seu_casl.sh submit TASK [partition]' \
        '  Tasks: data-prep cpu-check gpu-smoke bf-smoke bf-train seqdiff-smoke seqdiff-train seqdiff-resume-JOB_ID accel-compare' \
        '  bash scripts/run_seu_casl.sh status JOB_ID' \
        '  bash scripts/run_seu_casl.sh logs JOB_ID' \
        'Defaults: 6126-24C-192G for CPU jobs, gpu-L20 for the one-GPU smoke test.'
}

case ${1:-help} in
    submit)
        if (( $# < 2 || $# > 3 )); then usage >&2; exit 2; fi
        task=$2
        log_dir="$root/jobs/acceptance"
        case $task in
            data-prep)
                partition=${3:-6126-24C-192G}
                log_dir="$root/jobs/data-prep"
                resources=(--time=03:00:00 --cpus-per-task=2)
                ;;
            cpu-check)
                partition=${3:-6126-24C-192G}
                resources=(--time=00:20:00 --cpus-per-task=2)
                ;;
            gpu-smoke)
                partition=${3:-gpu-L20}
                resources=(--time=00:30:00 --nodes=1 --ntasks-per-node=1
                           --cpus-per-task=3 --gres=gpu:1)
                ;;
            bf-smoke|bf-train|seqdiff-smoke|seqdiff-train|seqdiff-resume-*|accel-compare)
                partition=${3:-gpu-L20}
                log_dir="$root/jobs/acceleration"
                wall=00:20:00
                if [[ $task == *-train || $task == seqdiff-resume-* ]]; then wall=12:00:00; fi
                if [[ $task == accel-compare ]]; then wall=03:00:00; fi
                resources=(--time="$wall" --nodes=1 --ntasks-per-node=1
                           --cpus-per-task=3 --gres=gpu:1)
                ;;
            *) usage >&2; exit 2 ;;
        esac

        # The interactive `sbatch` alias is unavailable in this Bash script.
        # Use the school's wrapper directly so its resource policies still apply.
        submitter=$(command -v sbatch_wrapper || true)
        if [[ -z $submitter ]]; then
            submitter=/usr/bin/sbatch_wrapper
        fi
        if [[ ! -x $submitter ]]; then
            printf 'School submitter not found: %s\n' "$submitter" >&2
            exit 1
        fi
        # Some GPU compute nodes have no Git. Record checkout revisions at submission.
        job_args=("$(git -C "$repo" rev-parse --verify HEAD)")
        if [[ $task == seqdiff-resume-* ]]; then
            prior=${task#seqdiff-resume-}
            [[ $prior =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
            folder="$log_dir/seqdiff-train-$prior"
            if [[ -e $folder/paired_results.tar.gz ]]; then
                printf 'ViViT job %s already exported its results\n' "$prior" >&2
                exit 2
            fi
            [[ -s $folder/revision.txt && -s $folder/vivit/training_state.json ]] || {
                printf 'No committed ViViT training checkpoint for %s\n' "$prior" >&2
                exit 2
            }
            read -r revision < "$folder/revision.txt"
            [[ $revision =~ ^[a-f0-9]{40}$ ]] || exit 2
            job_args[0]=$revision
        fi
        if [[ $task == gpu-smoke || $task == bf-* || $task == seqdiff-* || $task == accel-compare ]]; then
            # Version lookup reads shared files only. Reject an unusable runtime before queuing.
            if command -v module >/dev/null; then module load cuda-12.8; fi
            env_dir=${CASL_SEU_ENV:-$root/envs/casl}
            cudnn_dir=$("$env_dir/bin/python" "$repo/scripts/seu_casl_cudnn.py" "$root")
            cuda_dir=${CASL_SEU_CUDA_ROOT:-/seu_share/apps/cuda-12.8}
            for program in nvcc ptxas nvlink; do
                [[ -x $cuda_dir/bin/$program ]] || { echo "Missing CUDA program: $cuda_dir/bin/$program" >&2; exit 1; }
            done
            if [[ ! -r $cuda_dir/nvvm/libdevice/libdevice.10.bc || ! -r $cuda_dir/lib64/libcudart.so.12 ]]; then
                echo "Incomplete CUDA toolkit at $cuda_dir (libdevice.10.bc or libcudart.so.12 missing)" >&2
                exit 1
            fi
            printf 'Verified CUDA toolkit and XLA libdevice in %s\n' "$cuda_dir"
            # The model verifies both pristine upstream checkouts with real Git at runtime.
            shared_git=${CASL_SEU_GIT_BIN:-}
            if [[ -z $shared_git ]]; then
                for candidate in "$env_dir/bin/git" /seu_share/apps/git*/bin/git /seu_share/apps/anaconda3-2024.10-1/bin/git; do
                    if [[ -x $candidate ]]; then shared_git=$candidate; break; fi
                done
            fi
            if [[ -z $shared_git ]]; then
                login_git=$(type -P git)
                digest=$(sha256sum -- "$login_git")
                git_dir="$root/runtime/git/${digest%% *}/bin"
                mkdir -p -- "$git_dir"
                shared_git="$git_dir/git"
                if ! cmp -s -- "$login_git" "$shared_git"; then
                    staging=$(mktemp "$git_dir/.git.XXXXXX")
                    cp -- "$login_git" "$staging" && chmod 755 "$staging" && mv -f -- "$staging" "$shared_git"
                fi
            fi
            "$shared_git" --version
            job_args+=("$cudnn_dir" "$cuda_dir" "$shared_git")
            if [[ $task != gpu-smoke ]]; then
                sinfo -h -p "$partition" -o '%P %a %l %G %t %D'
            fi
        elif [[ $task == cpu-check ]]; then
            for dependency in vendor/casl vendor/casl/zea; do
                job_args+=("$(git -C "$repo/$dependency" rev-parse --verify HEAD)")
            done
        fi
        mkdir -p -- "$log_dir"
        printf 'Log directory: %s\n' "$log_dir"
        exec "$submitter" --job-name="casl-$task" --partition="$partition" \
            "${resources[@]}" --output="$log_dir/%x-%j.log" \
            "$repo/scripts/seu_casl_job.sh" "$task" "$repo" "${job_args[@]}"
        ;;
    status|logs)
        if (( $# != 2 )) || [[ ! $2 =~ ^[0-9]+$ ]]; then usage >&2; exit 2; fi
        job_id=$2
        if [[ $1 == status ]]; then
            if command -v sjob >/dev/null; then
                exec sjob -j "$job_id"
            fi
            exec sacct -j "$job_id" --format=JobID,Partition,State,ExitCode,Elapsed
        fi
        shopt -s nullglob
        logs=("$root/jobs/data-prep"/casl-*-"$job_id".log
              "$root/jobs/acceptance"/casl-*-"$job_id".log
              "$root/jobs/acceleration"/casl-*-"$job_id".log)
        if (( ${#logs[@]} == 1 )); then
            exec tail -n 50 -F -- "${logs[0]}"
        fi
        if command -v speek >/dev/null; then
            exec speek -f "$job_id"
        fi
        printf 'No unique local log for job %s. Check the job status first.\n' "$job_id" >&2
        exit 1
        ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
