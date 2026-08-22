#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/train:${ROOT}/upstream/DiffSynth-Studio:${ROOT}/upstream/MindIE-SD:${ROOT}/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
mkdir -p "${ROOT}/logs/sampling"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${ROOT}/sampling/sample_latents.py" --config "${1:-${ROOT}/configs/sampling.yaml}" "${@:2}" 2>&1 | tee "${ROOT}/logs/sampling/${RUN_ID}.log"
else
  python "${ROOT}/sampling/sample_latents.py" --config "${1:-${ROOT}/configs/sampling.yaml}" "${@:2}" 2>&1 | tee "${ROOT}/logs/sampling/${RUN_ID}.log"
fi
