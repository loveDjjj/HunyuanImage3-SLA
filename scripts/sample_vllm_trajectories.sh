#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-/mnt/share/r50063443/vllm-omni}"
export PYTHONPATH="${ROOT}:${VLLM_OMNI_REPO}:${PYTHONPATH:-}"
mkdir -p "${ROOT}/logs/trajectory_sampling"
LOG_FILE="${ROOT}/logs/trajectory_sampling/vllm-$(date +%Y%m%d-%H%M%S).log"
python "${ROOT}/sampling/sample_vllm_trajectories.py" "$@" 2>&1 | tee "${LOG_FILE}"
