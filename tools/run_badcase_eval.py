#!/usr/bin/env python3
"""Run HunyuanImage3 badcase T2I/I2I evaluation against vLLM-Omni."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_badcase_eval import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    TASKS,
    load_cases,
    safe_index,
)


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def valid_output(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except OSError:
        return False


def decode_response_image(response: requests.Response) -> bytes:
    try:
        payload = response.json()
        encoded = payload["data"][0]["b64_json"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Invalid image response: {response.text[:1000]}") from exc
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Image response contains an empty b64_json field")
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        return image_bytes
    except (ValueError, OSError) as exc:
        raise ValueError("Server returned invalid base64 image data") from exc


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    retries: int,
    timeout: float,
    **kwargs: Any,
) -> requests.Response:
    for attempt in range(retries + 1):
        try:
            for upload in kwargs.get("files", ()):
                file_value = upload[1]
                handle = file_value[1] if isinstance(file_value, tuple) else file_value
                if hasattr(handle, "seek"):
                    handle.seek(0)
            response = session.request(method, url, timeout=timeout, **kwargs)
            if response.status_code < 400:
                return response
            if response.status_code not in RETRYABLE_STATUS or attempt == retries:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:2000]}"
                )
        except requests.RequestException:
            if attempt == retries:
                raise
        time.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def t2i_request(
    session: requests.Session, base_url: str, row: dict, args: argparse.Namespace
) -> bytes:
    body: dict[str, Any] = {
        "prompt": row["prompt"],
        "n": 1,
        "size": args.t2i_size,
        "response_format": "b64_json",
        "output_format": "png",
        "num_inference_steps": args.steps,
        "seed": int(row["seed"]),
    }
    if args.model:
        body["model"] = args.model
    if args.bot_task:
        body["bot_task"] = args.bot_task
    if args.system_prompt_type:
        body["use_system_prompt"] = args.system_prompt_type
    response = request_with_retries(
        session,
        "POST",
        f"{base_url}/v1/images/generations",
        args.retries,
        args.timeout,
        json=body,
    )
    return decode_response_image(response)


def i2i_request(
    session: requests.Session, base_url: str, row: dict, args: argparse.Namespace
) -> bytes:
    input_paths = row.get("inputs")
    if not isinstance(input_paths, list) or not input_paths:
        raise ValueError(
            "i2i case has no local inputs; run prepare_badcase_eval.py first"
        )
    form: dict[str, str] = {
        "prompt": str(row["prompt"]),
        "n": "1",
        "size": args.i2i_size,
        "response_format": "b64_json",
        "output_format": "png",
        "num_inference_steps": str(args.steps),
        "seed": str(int(row["seed"])),
    }
    if args.model:
        form["model"] = args.model
    if args.bot_task:
        form["bot_task"] = args.bot_task
    if args.system_prompt_type:
        form["sys_type"] = args.system_prompt_type

    with ExitStack() as stack:
        files = []
        for input_path in input_paths:
            path = Path(input_path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing input image: {path}")
            handle = stack.enter_context(path.open("rb"))
            files.append(("image", (path.name, handle, "application/octet-stream")))
        response = request_with_retries(
            session,
            "POST",
            f"{base_url}/v1/images/edits",
            args.retries,
            args.timeout,
            data=form,
            files=files,
        )
    return decode_response_image(response)


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def output_paths(task_dir: Path, run_name: str | None) -> tuple[Path, Path]:
    if run_name:
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", run_name).strip("_")
        if not safe_name:
            raise ValueError(f"Invalid run name: {run_name!r}")
        run_root = task_dir / "runs" / safe_name
        return run_root / "output_images", run_root / "run_results.jsonl"
    return task_dir / "output_images", task_dir / "run_results.jsonl"


def run_task(
    task_dir: Path, task_name: str, session: requests.Session, args: argparse.Namespace
) -> tuple[int, int]:
    _, rows = load_cases(task_dir / "cases.json")
    output_root, results_path = output_paths(task_dir, args.run_name)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = rows[args.offset :]
    if args.limit:
        selected = selected[: args.limit]
    completed = 0
    failed = 0

    with results_path.open("a", encoding="utf-8") as results:
        for row in tqdm(selected, desc=f"generate {task_name}", unit="case"):
            sample_id = safe_index(row.get("index"))
            output_path = output_root / sample_id / f"seed_{int(row['seed'])}.png"
            if not args.overwrite and valid_output(output_path):
                continue
            started = time.monotonic()
            record: dict[str, Any] = {
                "task": task_name,
                "index": sample_id,
                "seed": str(row["seed"]),
                "output": str(output_path.resolve()),
            }
            try:
                if task_name == "badcase_ti2i":
                    image_bytes = i2i_request(session, args.base_url, row, args)
                else:
                    image_bytes = t2i_request(session, args.base_url, row, args)
                atomic_write_bytes(output_path, image_bytes)
                record.update(
                    status="ok", elapsed_seconds=round(time.monotonic() - started, 3)
                )
                completed += 1
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                failed += 1
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
    return completed, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--steps", type=int, default=8, help="Distilled model denoising steps"
    )
    parser.add_argument("--t2i-size", default="1024x1024")
    parser.add_argument("--i2i-size", default="auto")
    parser.add_argument("--bot-task", default=None)
    parser.add_argument("--system-prompt-type", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Isolate outputs under runs/<name>/output_images instead of overwriting the default run",
    )
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.offset < 0 or args.limit < 0 or args.retries < 0:
        parser.error("--offset, --limit, and --retries must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    session = requests.Session()
    if args.api_key:
        session.headers["Authorization"] = f"Bearer {args.api_key}"
    task_names = TASKS if args.task == "all" else (args.task,)
    completed = 0
    failed = 0
    for task_name in task_names:
        task_completed, task_failed = run_task(
            args.dataset_root / task_name, task_name, session, args
        )
        completed += task_completed
        failed += task_failed
    print(
        json.dumps(
            {"completed": completed, "failed": failed}, ensure_ascii=False, indent=2
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
