#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${ROOT}/tools/verify_latent.py" --cache-dir "${1:-${ROOT}/data/cache}"
