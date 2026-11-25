#!/bin/bash -l
set -eo pipefail

: "${SLURM_JOB_ID:?Submit this job with scripts/run_seu_casl.sh}"
task=$1
cd -- "$2"
root=${CASL_SEU_ROOT:-$HOME/casl}
nvme_root=${CASL_SEU_NVME_ROOT:-/seu_nvme/home/chenchuan/$(id -un)/casl}
env_dir=${CASL_SEU_ENV:-$root/envs/casl}
dataset=CASL-EchoNet-polar
share="$root/datasets"
nvme="$nvme_root/datasets"
source_data="$share/$dataset"
target_data="$nvme/$dataset"

hostname
if [[ -n ${3:-} ]]; then
    printf 'submitted_repo_rev=%s\n' "$3"
fi
if command -v git >/dev/null; then
    printf 'compute_repo_rev=%s\n' "$(git rev-parse --verify HEAD)"
fi

check_layout() {
    local directory=$1 split expected actual
    for split in train:6985 val:500 test:500 rejected:2045; do
        expected=${split#*:}
        split=${split%:*}
        actual=$(find "$directory/$split" -type f -name '*.hdf5' -printf '.' | wc -c)
        printf '%s %s %s\n' "$directory" "$split" "$actual"
        if [[ $actual != "$expected" ]]; then
            printf 'Expected %s HDF5 files, found %s.\n' "$expected" "$actual" >&2
            return 1
        fi
    done
}

case $task in
    data-prep)
        archive="$share/$dataset.tar"
        copy_stage="$nvme/.$dataset.copying"
        exec 9>"$share/.$dataset.prepare.lock"
        if ! flock -n 9; then
            printf 'Cannot acquire the data-prep lock in %s. Check for an existing job.\n' "$share" >&2
            exit 1
        fi

        echo '[1/4] Verify archive'
        archive_bytes=$(stat -c '%s' -- "$archive")
        if [[ $archive_bytes != 73705502720 ]]; then
            printf 'Expected 73705502720 bytes, found %s in %s.\n' "$archive_bytes" "$archive" >&2
            exit 1
        fi
        printf '%s  %s\n' 82f96f805bbcdffb63c092204e3eb63d3d153cc84bd560a06058176959494928 "$archive" |
            sha256sum -c -

        echo '[2/4] Extract on SHARE'
        if [[ ! -e $source_data ]]; then
            unpack_stage=$(mktemp -d "$share/.$dataset.unpack.XXXXXX")
            tar -xf "$archive" -C "$unpack_stage"
            check_layout "$unpack_stage/$dataset"
            mv -Tn -- "$unpack_stage/$dataset" "$source_data"
            rmdir -- "$unpack_stage"
        else
            check_layout "$source_data"
        fi

        echo '[3/4] Copy to NVMe'
        if [[ ! -e $target_data ]]; then
            mkdir -p -- "$copy_stage"
            rsync -ac --no-owner --no-group --partial --info=progress2 -- \
                "$source_data/" "$copy_stage/"
            check_layout "$copy_stage"
            echo '[4/4] Compare file contents'
            diff -qr -- "$source_data" "$copy_stage"
            mv -Tn -- "$copy_stage" "$target_data"
            [[ ! -e $copy_stage ]]
        else
            echo '[4/4] Verify existing NVMe copy'
            check_layout "$target_data"
            diff -qr -- "$source_data" "$target_data"
        fi
        printf 'DATA_PREP_OK\nSHARE %s\nNVMe  %s\n' "$source_data" "$target_data"
        ;;
    cpu-check)
        if [[ -n ${5:-} ]]; then
            printf 'submitted_casl_rev=%s\nsubmitted_zea_rev=%s\n' "$4" "$5"
        elif command -v git >/dev/null; then
            git -C vendor/casl rev-parse HEAD
            git -C vendor/casl/zea rev-parse HEAD
        fi
        py="$env_dir/bin/python"
        "$py" -VV
        "$py" -u scripts/seu_casl_check.py cpu "$source_data" "$target_data"
        ;;
    gpu-smoke)
        module load anaconda3-2024.10-1
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$env_dir"
        module load cuda-12.8
        cudnn_dir=${4:-}
        if [[ -z $cudnn_dir ]]; then
            cudnn_dir=$(python scripts/seu_casl_cudnn.py "$root")
        fi
        cuda_dir=${5:-${CASL_SEU_CUDA_ROOT:-/seu_share/apps/cuda-12.8}}
        export CUDA_HOME="$cuda_dir" CUDA_PATH="$cuda_dir" PATH="$cuda_dir/bin:$PATH"
        export LD_LIBRARY_PATH="$cudnn_dir:$cuda_dir/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$cuda_dir"
        if [[ -n ${6:-} ]]; then
            export PATH="$PATH:${6%/*}"
            git --version
        fi
        printf 'CASL cuDNN directory: %s\nCUDA toolkit and XLA data: %s\n' "$cudnn_dir" "$cuda_dir"

        out="$root/jobs/acceptance/gpu-$SLURM_JOB_ID"
        mkdir -p -- "$out"
        nvidia-smi
        export NVIDIA_TF32_OVERRIDE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
        if ! python scripts/check_gpu.py --output "$out/gpu.json"; then
            module avail cuda cudnn 2>&1 || true
            exit 2
        fi

        JAX_PLATFORMS=cuda python -m cognitive_ultrasound evaluate \
            --config configs/baseline.yaml --data-root "$target_data" \
            --checkpoint "$PWD/checkpoints/official" \
            --split val --limit-cases 1 --frames 5 --methods casl --budgets 7 \
            --metrics psnr ssim --output "$out/evaluation"
        python scripts/seu_casl_check.py gpu "$out/evaluation"
        ;;
    *) printf 'Unknown job: %s\n' "$task" >&2; exit 2 ;;
esac
