#!/usr/bin/env bash
# Usage: bash scripts/train_dense.sh [CONFIG] [train_sla.py options]
# CONFIG default: configs/train_sla.yaml; stage is always dense
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)-dense"
mkdir -p "${ROOT}/logs/training"
"${ROOT}/scripts/train.sh" "$@" --stage dense 2>&1 | tee "${ROOT}/logs/training/${RUN_ID}.log"
