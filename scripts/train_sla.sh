#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/logs/training"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
"${ROOT}/scripts/train.sh" "${1:-${ROOT}/configs/train_sla.yaml}" "${@:2}" 2>&1 | tee "${ROOT}/logs/training/${RUN_ID}.log"
