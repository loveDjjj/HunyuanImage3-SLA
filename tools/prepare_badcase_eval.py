#!/usr/bin/env python3
"""Download badcase evaluation assets and rewrite cases.json local paths."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from PIL import Image
from tqdm import tqdm


DEFAULT_DATASET_ROOT = Path("/mnt/share/r50063443/HunyuanImage3-SLA/datasets/test")
TASKS = ("badcase_ti2i", "badcase_t2i")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_cases(path: Path) -> tuple[object, list[dict]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("cases") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(
            f"{path} must contain a JSON list, or an object with a 'cases' list"
        )
    return document, rows


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def safe_index(value: object) -> str:
    if value is None:
        raise ValueError("Case is missing index")
    sample_id = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
        raise ValueError(f"Invalid case index: {value!r}")
    return sample_id


def url_filename(url: str, fallback: str, ordinal: int = 0) -> str:
    basename = Path(unquote(urlparse(url).path)).name
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    if not basename:
        basename = fallback
    path = Path(basename)
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        basename = f"{path.stem or fallback}.png"
    if ordinal:
        path = Path(basename)
        basename = f"{path.stem}_{ordinal}{path.suffix}"
    return basename


def download_image(
    session: requests.Session, url: str, target: Path, timeout: float
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        try:
            validate_image(target)
            return target
        except OSError:
            target.unlink()

    temporary = target.with_name(f".{target.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        with session.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            content_type = (
                response.headers.get("Content-Type", "").split(";", 1)[0].strip()
            )
            if content_type and not content_type.startswith("image/"):
                raise ValueError(
                    f"Expected an image from {url}, got Content-Type {content_type!r}"
                )
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        validate_image(temporary)
        temporary.replace(target)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def prepare_task(
    task_dir: Path, task_name: str, timeout: float
) -> tuple[int, list[str]]:
    cases_path = task_dir / "cases.json"
    if not cases_path.is_file():
        raise FileNotFoundError(f"Missing cases file: {cases_path}")

    document, rows = load_cases(cases_path)
    baseline_root = task_dir / "baseline_images"
    input_root = task_dir / "input_images"
    baseline_root.mkdir(parents=True, exist_ok=True)
    (task_dir / "output_images").mkdir(parents=True, exist_ok=True)
    if task_name == "badcase_ti2i":
        input_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "HunyuanImage3-SLA-badcase-eval/1.0"
    failures: list[str] = []
    downloaded = 0

    for row in tqdm(rows, desc=f"prepare {task_name}", unit="case"):
        try:
            sample_id = safe_index(row.get("index"))
        except ValueError as exc:
            failures.append(f"index={row.get('index')}: {exc}")
            continue

        try:
            baseline_url = row.get("baseline_url")
            if not isinstance(baseline_url, str) or not baseline_url:
                raise ValueError("case has no baseline_url")
            filename = url_filename(baseline_url, "baseline.png")
            target = download_image(
                session, baseline_url, baseline_root / sample_id / filename, timeout
            )
            row["baseline_image"] = str(target.resolve())
            downloaded += 1
        except (OSError, ValueError, requests.RequestException) as exc:
            failures.append(f"index={sample_id} baseline: {exc}")

        if task_name == "badcase_ti2i":
            try:
                image_urls = row.get("image_urls")
                if not isinstance(image_urls, list) or not image_urls:
                    raise ValueError("i2i case has no image_urls")
                local_inputs = []
                used_names: set[str] = set()
                for ordinal, url in enumerate(image_urls, start=1):
                    if not isinstance(url, str) or not url:
                        raise ValueError(f"invalid image_urls[{ordinal - 1}]")
                    filename = url_filename(url, f"input_{ordinal}.png")
                    if filename in used_names:
                        filename = url_filename(url, f"input_{ordinal}.png", ordinal)
                    used_names.add(filename)
                    target = download_image(
                        session, url, input_root / sample_id / filename, timeout
                    )
                    local_inputs.append(str(target.resolve()))
                    downloaded += 1
                row["inputs"] = local_inputs
            except (OSError, ValueError, requests.RequestException) as exc:
                failures.append(f"index={sample_id} input: {exc}")

    atomic_write_json(cases_path, document)
    return downloaded, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = TASKS if args.task == "all" else (args.task,)
    all_failures: list[str] = []
    total_downloaded = 0
    for task_name in task_names:
        downloaded, failures = prepare_task(
            args.dataset_root / task_name, task_name, args.timeout
        )
        total_downloaded += downloaded
        all_failures.extend(f"{task_name}: {failure}" for failure in failures)

    summary = {"downloaded_or_reused": total_downloaded, "failures": len(all_failures)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for failure in all_failures:
        print(f"ERROR {failure}")
    if all_failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
