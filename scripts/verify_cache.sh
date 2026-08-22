#!/usr/bin/env bash
# Usage: bash scripts/verify_cache.sh [CACHE_DIR]
# CACHE_DIR default: data/cache
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${ROOT}/data/cache"
if [[ $# -gt 0 ]]; then
  CACHE_DIR="$1"
fi
python "${ROOT}/tools/verify_latent.py" --cache-dir "${CACHE_DIR}"
