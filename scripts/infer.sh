#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${PYTHON_BIN:-python}" infer/infer_adabridge_sr.py \
    --train_cfg configs/infer.yml "$@"
