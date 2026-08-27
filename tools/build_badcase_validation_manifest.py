#!/usr/bin/env python3
"""Build a prompt-level trajectory manifest from badcase_t2i cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_manifest(cases_path: Path, output_path: Path, limit: int = 0) -> list[dict]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"Badcase file must contain a list, got {type(rows).__name__}.")
    selected = rows[: limit or None]
    manifest = []
    seen = set()
    for row in selected:
        sample_id = f"badcase_t2i_{row['index']}"
        if sample_id in seen:
            raise ValueError(f"Duplicate badcase index: {row['index']!r}")
        seen.add(sample_id)
        manifest.append(
            {
                "id": sample_id,
                "prompt": str(row["prompt"]),
                "seed": int(row["seed"]),
                "source_index": str(row["index"]),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("datasets/test/badcase_t2i/cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/validation/badcase_t2i/prompts.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    rows = build_manifest(args.cases, args.output, args.limit)
    print(json.dumps({"output": str(args.output), "sample_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
