#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${BADCASE_DATASET_ROOT:-/mnt/share/r50063443/HunyuanImage3-SLA/datasets/test}"

cd "$REPO_ROOT"
python tools/prepare_badcase_eval.py --dataset-root "$DATASET_ROOT" "$@"
