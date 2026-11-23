#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "${BASH_SOURCE[0]%/*}/.." && pwd -P)
root=${CASL_SEU_ROOT:-$HOME/casl}

usage() {
    printf '%s\n' \
        'Usage:' \
        '  bash scripts/run_seu_casl.sh submit data-prep|cpu-check|gpu-smoke [partition]' \
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
        mkdir -p -- "$log_dir"
        printf 'Log directory: %s\n' "$log_dir"
        exec "$submitter" --job-name="casl-$task" --partition="$partition" \
            "${resources[@]}" --output="$log_dir/%x-%j.log" \
            "$repo/scripts/seu_casl_job.sh" "$task" "$repo"
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
              "$root/jobs/acceptance"/casl-*-"$job_id".log)
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
