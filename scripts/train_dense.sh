#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)-dense"
mkdir -p "${ROOT}/logs/training"
"${ROOT}/scripts/train.sh" "${1:-${ROOT}/configs/train_sla.yaml}" --stage dense "${@:2}" 2>&1 | tee "${ROOT}/logs/training/${RUN_ID}.log"
