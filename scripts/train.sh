#!/usr/bin/env bash
# Usage: TRAIN_PARALLEL={single,ddp,zero3} bash scripts/train.sh [CONFIG] [train_sla.py options]
# CONFIG default: configs/train_sla.yaml
# Options include: --stage {dense,sla}, --max-steps N, --resume-from CHECKPOINT
# TRAIN_PARALLEL default: single; zero3 config default: configs/accelerate_zero3_16npu.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/train:${ROOT}/upstream/DiffSynth-Studio:${ROOT}/upstream/MindIE-SD:${ROOT}/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${ROOT}/configs/train_sla.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-single}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT}/configs/accelerate_zero3_16npu.yaml}"

if [[ "${TRAIN_PARALLEL}" == "zero3" ]]; then
  accelerate launch --config_file "${ACCELERATE_CONFIG}" "${ROOT}/train/train_sla.py" --config "${CONFIG}" "$@"
elif [[ "${TRAIN_PARALLEL}" == "ddp" || "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC_PER_NODE}" "${ROOT}/train/train_sla.py" --config "${CONFIG}" "$@"
elif [[ "${TRAIN_PARALLEL}" == "single" ]]; then
  python "${ROOT}/train/train_sla.py" --config "${CONFIG}" "$@"
else
  echo "Unsupported TRAIN_PARALLEL=${TRAIN_PARALLEL}; expected single, ddp, or zero3" >&2
  exit 2
fi
