#!/usr/bin/env bash
# Usage: bash scripts/train.sh [CONFIG] [train_sla.py options]
# CONFIG default: configs/train_sla.yaml
# Options include: --stage {dense,sla}, --max-steps N, --resume-from CHECKPOINT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/train:${ROOT}/upstream/DiffSynth-Studio:${ROOT}/upstream/MindIE-SD:${ROOT}/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"

CONFIG="${ROOT}/configs/train_sla.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC_PER_NODE}" "${ROOT}/train/train_sla.py" --config "${CONFIG}" "$@"
else
  python "${ROOT}/train/train_sla.py" --config "${CONFIG}" "$@"
fi
