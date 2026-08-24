#!/usr/bin/env bash
# Usage: TRAIN_PARALLEL={single,ddp,zero3} bash scripts/train_sla.sh [CONFIG] [train_sla.py options]
# CONFIG default: configs/train_sla.yaml; stage default: sla
# ZeRO-3 default: TRAIN_PARALLEL=zero3 with configs/accelerate_zero3_16npu.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/logs/training"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${ROOT}/logs/training/${RUN_ID}.log"
{
  echo "training_log=${LOG_FILE}"
  "${ROOT}/scripts/train.sh" "$@"
} 2>&1 | tee "${LOG_FILE}"
