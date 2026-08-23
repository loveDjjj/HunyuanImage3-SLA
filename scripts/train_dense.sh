#!/usr/bin/env bash
# Usage: TRAIN_PARALLEL={single,ddp,zero3} bash scripts/train_dense.sh [CONFIG] [train_sla.py options]
# CONFIG default: configs/train_sla.yaml; stage is always dense
# ZeRO-3 default: TRAIN_PARALLEL=zero3 with configs/accelerate_zero3_16npu.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)-dense"
mkdir -p "${ROOT}/logs/training"
"${ROOT}/scripts/train.sh" "$@" --stage dense 2>&1 | tee "${ROOT}/logs/training/${RUN_ID}.log"
