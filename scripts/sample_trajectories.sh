#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT}/configs/accelerate_zero3_trajectory_8npu.yaml}"
mkdir -p "${ROOT}/logs/trajectory_sampling"
LOG_FILE="${ROOT}/logs/trajectory_sampling/$(date +%Y%m%d-%H%M%S).log"
accelerate launch --config_file "${ACCELERATE_CONFIG}" \
  "${ROOT}/sampling/sample_trajectories.py" "$@" 2>&1 | tee "${LOG_FILE}"
