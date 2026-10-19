#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# The released model was trained on 4 GPUs (batch 1 per GPU, gradient accumulation 4).
NUM_GPUS="${NUM_GPUS:-4}"
launch_args=(--num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 --dynamo_backend no)
if [ "$NUM_GPUS" -gt 1 ]; then
    launch_args+=(--multi_gpu --gpu_ids all)
fi
exec "${PYTHON_BIN:-python}" -m accelerate.commands.launch \
    "${launch_args[@]}" "$@" train/train_adabridge_sr.py --config "${TRAIN_CONFIG:-configs/train.yml}"
