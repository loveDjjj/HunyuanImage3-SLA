#!/usr/bin/env python3
"""Filter a COYO metadata JSONL into a reproducible download manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def read_rows(path: Path):
    if path.suffix != ".parquet":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading COYO Parquet requires pyarrow: python -m pip install pyarrow") from exc
    columns = ["id", "url", "text", "width", "height", "clip_similarity_vitl14", "aesthetic_score_laion_v2", "nsfw_score_opennsfw2", "nsfw_score_gantman", "watermark_score"]
    parquet = pq.ParquetFile(path)
    available = [name for name in columns if name in parquet.schema.names]
    for batch in parquet.iter_batches(columns=available, batch_size=65536):
        for row in batch.to_pylist():
            yield row


def acceptable(row: dict, args: argparse.Namespace) -> bool:
    caption = str(row.get("text", row.get("caption", ""))).strip()
    width, height = int(row.get("width") or 0), int(row.get("height") or 0)
    if not row.get("id") or not row.get("url") or not caption or min(width, height) < args.min_side:
        return False
    if float(row.get("clip_similarity_vitl14", 1.0) or 0.0) < args.min_clip_similarity:
        return False
    if float(row.get("aesthetic_score_laion_v2", 10.0) or 0.0) < args.min_aesthetic:
        return False
    nsfw = max(float(row.get("nsfw_score_opennsfw2", 0.0) or 0.0), float(row.get("nsfw_score_gantman", 0.0) or 0.0))
    watermark = float(row.get("watermark_score", 0.0) or 0.0)
    return nsfw < args.max_nsfw and watermark < args.max_watermark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="COYO metadata JSONL or official Parquet shard")
    parser.add_argument("--output", required=True, help="candidate JSONL for download_images.py")
    parser.add_argument("--candidate-count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--min-clip-similarity", type=float, default=0.28)
    parser.add_argument("--min-aesthetic", type=float, default=4.5)
    parser.add_argument("--max-nsfw", type=float, default=0.1)
    parser.add_argument("--max-watermark", type=float, default=0.5)
    args = parser.parse_args()
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")

    rng, reservoir, eligible = random.Random(args.seed), [], 0
    for row in tqdm(read_rows(Path(args.input)), desc="filter COYO", unit="row"):
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
