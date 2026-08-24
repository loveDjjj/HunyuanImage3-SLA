#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${BADCASE_DATASET_ROOT:-/mnt/share/r50063443/HunyuanImage3-SLA/datasets/test}"
VLLM_OMNI_URL="${VLLM_OMNI_URL:-http://127.0.0.1:8000}"

cd "$REPO_ROOT"
python tools/run_badcase_eval.py \
  --dataset-root "$DATASET_ROOT" \
  --base-url "$VLLM_OMNI_URL" \
  "$@"
