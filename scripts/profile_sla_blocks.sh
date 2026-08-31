#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT}/configs/block_profile_guidance1.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT}/configs/accelerate_zero3_16npu.yaml}"

export PYTHONPATH="${ROOT}/train:${ROOT}/upstream/DiffSynth-Studio:${ROOT}/upstream/MindIE-SD:${ROOT}/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${ROOT}/logs/block_profile"
LOG_FILE="${ROOT}/logs/block_profile/$(date +%Y%m%d-%H%M%S).log"
{
  echo "block_profile_log=${LOG_FILE}"
  accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    "${ROOT}/tools/profile_sla_blocks.py" --config "${CONFIG}"
} 2>&1 | tee "${LOG_FILE}"
