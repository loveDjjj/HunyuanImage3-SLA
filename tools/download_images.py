#!/usr/bin/env python3
"""Independent URL downloader for COYO-style JSONL metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="JSONL with id, url and caption/text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.metadata).read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        rows = rows[:args.limit]
    accepted = []
    for row in tqdm(rows, desc="download", unit="image"):
        sample_id = str(row["id"])
        target = output / f"{sample_id}.jpg"
        if not target.exists():
            try:
                response = requests.get(row["url"], timeout=args.timeout, headers={"User-Agent": "HunyuanImage3-SLA/1.0"})
                response.raise_for_status()
                temporary = target.with_suffix(".incomplete")
                temporary.write_bytes(response.content)
                with Image.open(temporary) as image:
                    image.verify()
                temporary.replace(target)
            except (requests.RequestException, OSError):
                temporary.unlink(missing_ok=True)
                continue
        accepted.append({"id": sample_id, "image_path": target.name, "caption": row.get("caption", row.get("text", ""))})
    manifest = output / "metadata.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    print(f"downloaded={len(accepted)} manifest={manifest}")


if __name__ == "__main__":
    main()
