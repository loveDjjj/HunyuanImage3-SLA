#!/usr/bin/env python3
"""Filter a COYO metadata JSONL into a reproducible download manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def acceptable(row: dict, args: argparse.Namespace) -> bool:
    caption = str(row.get("text", row.get("caption", ""))).strip()
    width, height = int(row.get("width") or 0), int(row.get("height") or 0)
    if not row.get("id") or not row.get("url") or not caption or min(width, height) < args.min_side:
        return False
    if float(row.get("clip_similarity_vitl14", 1.0) or 0.0) < args.min_clip_similarity:
        return False
    if float(row.get("aesthetic_score_laion_v2", 10.0) or 0.0) < args.min_aesthetic:
        return False
    return row.get("nsfw") not in (True, "true", "TRUE", 1, "1") and row.get("watermark") not in (True, "true", "TRUE", 1, "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="COYO metadata JSONL")
    parser.add_argument("--output", required=True, help="candidate JSONL for download_images.py")
    parser.add_argument("--candidate-count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--min-clip-similarity", type=float, default=0.28)
    parser.add_argument("--min-aesthetic", type=float, default=4.5)
    args = parser.parse_args()
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")

    rng, reservoir, eligible = random.Random(args.seed), [], 0
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in tqdm(handle, desc="filter COYO", unit="row"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not acceptable(row, args):
                continue
            eligible += 1
            item = {"id": str(row["id"]), "url": row["url"], "caption": str(row.get("text", row.get("caption", ""))).strip(), "width": int(row.get("width") or 0), "height": int(row.get("height") or 0)}
            if len(reservoir) < args.candidate_count:
                reservoir.append(item)
            else:
                index = rng.randrange(eligible)
                if index < args.candidate_count:
                    reservoir[index] = item
    reservoir.sort(key=lambda item: item["id"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in reservoir), encoding="utf-8")
    print(f"eligible={eligible} candidates={len(reservoir)} output={output}")


if __name__ == "__main__":
    main()
