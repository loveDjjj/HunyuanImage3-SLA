#!/usr/bin/env bash
# Usage: NPROC_PER_NODE=8 bash scripts/sample.sh [CONFIG] [--resume]
# CONFIG default: configs/sampling.yaml; NPROC_PER_NODE default: 1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
mkdir -p "${ROOT}/logs/sampling"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
CONFIG="${ROOT}/configs/sampling.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${ROOT}/sampling/sample_latents.py" --config "${CONFIG}" "$@" 2>&1 | tee "${ROOT}/logs/sampling/${RUN_ID}.log"
else
  python "${ROOT}/sampling/sample_latents.py" --config "${CONFIG}" "$@" 2>&1 | tee "${ROOT}/logs/sampling/${RUN_ID}.log"
fi
