#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/train:${ROOT}/upstream/DiffSynth-Studio:${ROOT}/upstream/MindIE-SD:${ROOT}/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
mkdir -p "${ROOT}/logs/sampling"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python "${ROOT}/sampling/sample_latents.py" --config "${1:-${ROOT}/configs/sampling.yaml}" "${@:2}" 2>&1 | tee "${ROOT}/logs/sampling/${RUN_ID}.log"
