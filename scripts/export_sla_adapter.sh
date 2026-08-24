#!/usr/bin/env bash
# Usage: bash scripts/export_sla_adapter.sh [CHECKPOINT] [OUTPUT_DIR] [export_sla_adapter.py options]
# CHECKPOINT default: tag named by results/training/default/latest
# OUTPUT_DIR default: results/adapters/<checkpoint-tag>
# Example: bash scripts/export_sla_adapter.sh results/training/default/sla-step-200
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="${ROOT}/results/training/default"

if [[ $# -gt 0 && "$1" != -* ]]; then
  CHECKPOINT="$1"
  shift
else
  if [[ ! -f "${TRAINING_DIR}/latest" ]]; then
    echo "Missing ${TRAINING_DIR}/latest; pass the checkpoint directory explicitly." >&2
    exit 1
  fi
  LATEST_TAG="$(tr -d '[:space:]' < "${TRAINING_DIR}/latest")"
  CHECKPOINT="${TRAINING_DIR}/${LATEST_TAG}"
fi

if [[ "${CHECKPOINT}" != /* ]]; then
  CHECKPOINT="${ROOT}/${CHECKPOINT}"
fi
TAG="$(basename "${CHECKPOINT}")"

if [[ $# -gt 0 && "$1" != -* ]]; then
  OUTPUT_DIR="$1"
  shift
else
  OUTPUT_DIR="${ROOT}/results/adapters/${TAG}"
fi

mkdir -p "${ROOT}/logs/export"
LOG_FILE="${ROOT}/logs/export/$(date +%Y%m%d-%H%M%S)-${TAG}.log"
{
  echo "checkpoint=${CHECKPOINT}"
  echo "output=${OUTPUT_DIR}"
  python "${ROOT}/tools/export_sla_adapter.py" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT_DIR}" \
    "$@"
  python "${ROOT}/tools/inspect_sla_adapter.py" --adapter-dir "${OUTPUT_DIR}"
  echo "export_log=${LOG_FILE}"
} 2>&1 | tee "${LOG_FILE}"
