#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-${ROOT}/results/adapters/sla-zero-init}"
if [[ $# -gt 0 ]]; then shift; fi

python "${ROOT}/tools/create_zero_sla_adapter.py" \
  --output "${OUTPUT}" \
  --config "${ROOT}/configs/train_sla_trajectory.yaml" \
  "$@"
python "${ROOT}/tools/inspect_sla_adapter.py" --adapter-dir "${OUTPUT}"
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
