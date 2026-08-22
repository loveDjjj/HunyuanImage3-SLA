#!/usr/bin/env bash
# Usage: bash scripts/train_sla.sh [CONFIG] [train_sla.py options]
# CONFIG default: configs/train_sla.yaml; stage default: sla
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/logs/training"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
"${ROOT}/scripts/train.sh" "$@" 2>&1 | tee "${ROOT}/logs/training/${RUN_ID}.log"
