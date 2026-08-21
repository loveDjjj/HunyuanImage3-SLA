#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/train:${ROOT}/upstream/DiffSynth-Studio:${ROOT}/upstream/MindIE-SD:${ROOT}/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"

# Single NPU: bash scripts/train.sh configs/train_sla.yaml
# Multi NPU: NPROC_PER_NODE=8 bash scripts/train.sh configs/train_sla.yaml
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC_PER_NODE}" "${ROOT}/train/train_sla.py" --config "$1" "${@:2}"
else
  python "${ROOT}/train/train_sla.py" --config "$1" "${@:2}"
fi
