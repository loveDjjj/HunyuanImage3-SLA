#!/usr/bin/env python3
"""Create a unique-image Flickr30k manifest for offline VAE sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from tqdm import tqdm


def stable_caption(captions: list[str], seed: int, sample_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return captions[int.from_bytes(digest[:8], "little") % len(captions)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Karpathy dataset_flickr30k.json")
    parser.add_argument("--images-dir", required=True, help="Directory containing Flickr30k JPEG files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")

    images_dir = Path(args.images_dir)
    data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    candidates: list[dict] = []
    seen: set[str] = set()
    for item in tqdm(data["images"], desc="prepare Flickr30k", unit="image"):
        if args.split != "all" and item.get("split") != args.split:
            continue
        filename = item["filename"]
        path = images_dir / filename
        sample_id = str(item.get("imgid", Path(filename).stem))
        captions = [str(sentence["raw"]).strip() for sentence in item.get("sentences", []) if str(sentence.get("raw", "")).strip()]
        if sample_id in seen or not path.is_file() or not captions:
            continue
        seen.add(sample_id)
        candidates.append({"id": sample_id, "image_path": filename, "caption": stable_caption(captions, args.seed, sample_id)})
    if len(candidates) < args.sample_count:
        raise RuntimeError(f"Only {len(candidates)} usable unique images; need {args.sample_count}.")
    candidates.sort(key=lambda row: row["id"])
    selected = random.Random(args.seed).sample(candidates, args.sample_count)
    selected.sort(key=lambda row: row["id"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    print(f"split={args.split} unique_images={len(candidates)} selected={len(selected)} output={output}")


if __name__ == "__main__":
    main()
