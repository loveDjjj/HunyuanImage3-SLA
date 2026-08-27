#!/usr/bin/env python3
"""Collect Hunyuan Stage-0 conditions or Dense trajectories with vLLM-Omni."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cache_schema import write_json  # noqa: E402
from common.trajectory_schema import write_trajectory_atomic  # noqa: E402
from sampling.vllm_trajectory_adapter import build_vllm_trajectory_artifact  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("stage0", "dit", "full"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "vllm_trajectory_sampling.yaml"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--deploy-config", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _safe_id(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    if not result:
        raise ValueError(f"Invalid sample id: {value!r}")
    return result


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("id", row.get("sample_id", index))
            row.setdefault("seed", 42)
            rows.append(row)
    return rows[: limit or None]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_stage0_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, limit)
    root = path.parent
    loaded = []
    for row in rows:
        sample_path = _resolve(root, row["path"])
        loaded.append(json.loads(sample_path.read_text(encoding="utf-8")))
    return loaded


def _group_rows_by_seed(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("seed", 42))].append(row)
    return dict(grouped)


def _rebuild_stage0_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted((output_dir / "samples").glob("sample_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "sample_id": row["sample_id"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "path": str(path.relative_to(output_dir)),
            }
        )
    temporary = output_dir / "manifest.jsonl.incomplete"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "manifest.jsonl")


def _rebuild_trajectory_manifest(output_dir: Path) -> None:
    rows = []
    for sample_dir in sorted((output_dir / "samples").glob("sample_*")):
        if not (sample_dir / "READY.json").is_file():
            continue
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "sample_id": metadata["sample_id"],
                "prompt": metadata["prompt"],
                "seed": metadata["seed"],
                "path": str(sample_dir.relative_to(output_dir)),
            }
        )
    temporary = output_dir / "manifest.jsonl.incomplete"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "manifest.jsonl")


def _runtime(cfg: dict[str, Any]):
    repo = Path(cfg["vllm_omni_repo"])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from transformers import AutoTokenizer
    from vllm.sampling_params import RequestOutputKind
    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
        resolve_stop_token_ids,
    )
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import _truncate_at_cot_end

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    return {
        "repo": repo,
        "Omni": Omni,
        "OmniDiffusionSamplingParams": OmniDiffusionSamplingParams,
        "RequestOutputKind": RequestOutputKind,
        "build_prompt_tokens": build_prompt_tokens,
        "resolve_stop_token_ids": resolve_stop_token_ids,
        "truncate": _truncate_at_cot_end,
        "tokenizer": tokenizer,
    }


def _prompt(row: dict[str, Any], cfg: dict[str, Any], runtime: dict[str, Any], *, with_ar: bool):
    prompt = {
        "prompt": str(row["prompt"]),
        "sample_id": _safe_id(str(row.get("id", row.get("sample_id")))),
        "height": 1024,
        "width": 1024,
        "use_system_prompt": cfg["use_system_prompt"],
        "modalities": ["image"],
    }
    if with_ar:
        built = runtime["build_prompt_tokens"](
            prompt["prompt"],
            runtime["tokenizer"],
            task="t2i",
            bot_task=cfg["bot_task"],
            sys_type=cfg["use_system_prompt"],
        )
        prompt["prompt_token_ids"] = built.token_ids
    else:
        prompt["extra"] = {
            "ar_generated_text": row["cot_text"],
            "ar_generated_token_ids": row.get("generated_token_ids", []),
            "ar_prompt_token_ids": row.get("prompt_token_ids", []),
            "sample_id": prompt["sample_id"],
        }
    return prompt


def _sampling_params(omni, cfg: dict[str, Any], runtime: dict[str, Any], seed: int):
    params = list(omni.default_sampling_params_list)
    stop_ids = runtime["resolve_stop_token_ids"](
        task="t2i",
        bot_task=cfg["bot_task"],
        tokenizer=runtime["tokenizer"],
        image_size=cfg["image_size"],
    )
    for value in params:
        if isinstance(value, runtime["OmniDiffusionSamplingParams"]):
            value.num_inference_steps = int(cfg["num_inference_steps"])
            value.guidance_scale = float(cfg["guidance_scale"])
            value.guidance_scale_provided = True
            value.seed = int(seed)
            value.height = 1024
            value.width = 1024
            value.output_type = "latent"
            value.return_teacher_trajectory = True
        else:
            value.temperature = 0.0
            value.top_p = 1.0
            value.top_k = -1
            value.max_tokens = int(cfg["max_new_tokens"])
            value.stop_token_ids = stop_ids
            if hasattr(value, "output_kind"):
                value.output_kind = runtime["RequestOutputKind"].FINAL_ONLY
    return params


def _stage0(args, cfg, runtime, rows, deploy_config: Path):
    output_dir = _resolve(ROOT, cfg["stage0_output_dir"])
    pending = []
    for row in rows:
        sample_id = _safe_id(str(row["id"]))
        path = output_dir / "samples" / f"sample_{sample_id}.json"
        if not (args.resume and path.is_file()):
            pending.append(row)
    if not pending:
        _rebuild_stage0_manifest(output_dir)
        return

    omni = runtime["Omni"](
        model=cfg["model_path"], deploy_config=str(deploy_config), mode="text-to-image", enforce_eager=True
    )
    prompts = [_prompt(row, cfg, runtime, with_ar=True) for row in pending]
    by_tokens = defaultdict(deque)
    for prompt, row in zip(prompts, pending):
        by_tokens[tuple(prompt["prompt_token_ids"])].append(row)
    outputs = omni.generate(
        prompts=prompts,
        sampling_params_list=_sampling_params(omni, cfg, runtime, seed=42),
        py_generator=True,
        use_tqdm=True,
    )
    with closing(outputs):
        for result in outputs:
            request_output = getattr(result, "request_output", None)
            completions = getattr(request_output, "outputs", None) or []
            if not completions:
                continue
            candidates = by_tokens.get(tuple(getattr(request_output, "prompt_token_ids", None) or []))
            if not candidates:
                raise RuntimeError("Could not map Stage-0 output back to its manifest prompt.")
            row = candidates.popleft()
            completion = completions[0]
            text = getattr(completion, "cumulative_text", None) or getattr(completion, "text", "") or ""
            sample_id = _safe_id(str(row["id"]))
            write_json(
                output_dir / "samples" / f"sample_{sample_id}.json",
                {
                    "sample_id": sample_id,
                    "prompt": str(row["prompt"]),
                    "seed": int(row.get("seed", 42)),
                    "prompt_token_ids": list(getattr(request_output, "prompt_token_ids", None) or []),
                    "generated_token_ids": list(getattr(completion, "cumulative_token_ids", None) or []),
                    "cot_text": runtime["truncate"](text),
                    "bot_task": cfg["bot_task"],
                    "use_system_prompt": cfg["use_system_prompt"],
                    "height": 1024,
                    "width": 1024,
                },
            )
    _rebuild_stage0_manifest(output_dir)


def _trajectory(args, cfg, runtime, rows, deploy_config: Path, *, full: bool):
    output_dir = _resolve(ROOT, cfg["output_dir"])
    pending = []
    for row in rows:
        sample_id = _safe_id(str(row.get("id", row.get("sample_id"))))
        if not (args.resume and (output_dir / "samples" / f"sample_{sample_id}" / "READY.json").is_file()):
            pending.append(row)
    if not pending:
        _rebuild_trajectory_manifest(output_dir)
        return

    omni = runtime["Omni"](
        model=cfg["model_path"], deploy_config=str(deploy_config), mode="text-to-image", enforce_eager=True
    )
    repo_commit = _git_commit(ROOT)
    vllm_commit = _git_commit(runtime["repo"])
    grouped = _group_rows_by_seed(pending)
    pending_writes: deque[Future] = deque()
    trajectory_count = 0

    def convert_and_write(trajectory: dict[str, Any], sample_id: str, seed: int) -> None:
        metadata, tensors = build_vllm_trajectory_artifact(
            trajectory,
            sample_id=sample_id,
            seed=seed,
            model_path=cfg["model_path"],
            vllm_commit=vllm_commit,
            repository_commit=repo_commit,
            bot_task=cfg["bot_task"],
            use_system_prompt=cfg["use_system_prompt"],
        )
        write_trajectory_atomic(output_dir / "samples" / f"sample_{sample_id}", metadata, tensors)

    def consume(outputs, seed: int, writer: ThreadPoolExecutor) -> None:
        nonlocal trajectory_count
        for result in outputs:
            trajectory = (getattr(result, "multimodal_output", None) or {}).get("trajectory")
            if not isinstance(trajectory, dict):
                continue
            trajectory_count += 1
            source_metadata = trajectory.get("metadata") or {}
            sample_id = _safe_id(str(source_metadata.get("sample_id") or ""))
            pending_writes.append(writer.submit(convert_and_write, trajectory, sample_id, seed))
            if len(pending_writes) >= 2:
                pending_writes.popleft().result()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="trajectory-writer") as writer:
        if len(grouped) == 1:
            seed, group = next(iter(grouped.items()))
            outputs = omni.generate(
                prompts=[_prompt(row, cfg, runtime, with_ar=full) for row in group],
                sampling_params_list=_sampling_params(omni, cfg, runtime, seed),
                py_generator=True,
                use_tqdm=True,
            )
            with closing(outputs):
                consume(outputs, seed, writer)
        else:
            try:
                for seed, group in sorted(grouped.items()):
                    outputs = omni.generate(
                        prompts=[_prompt(row, cfg, runtime, with_ar=full) for row in group],
                        sampling_params_list=_sampling_params(omni, cfg, runtime, seed),
                        py_generator=False,
                        use_tqdm=True,
                    )
                    consume(outputs, seed, writer)
            finally:
                omni.close()
        while pending_writes:
            pending_writes.popleft().result()
    if trajectory_count != len(pending):
        raise RuntimeError(
            f"vLLM completed {len(pending)} request(s) but returned {trajectory_count} teacher trajectory payload(s)."
        )
    _rebuild_trajectory_manifest(output_dir)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runtime = _runtime(cfg)
    deploy_key = {
        "stage0": "stage0_deploy_config",
        "dit": "dit_deploy_config",
        "full": "full_deploy_config",
    }[args.phase]
    deploy_config = args.deploy_config or _resolve(runtime["repo"], cfg[deploy_key])
    rows = (
        _load_stage0_rows(args.manifest, args.limit)
        if args.phase == "dit"
        else _read_jsonl(args.manifest, args.limit)
    )
    if args.phase == "stage0":
        _stage0(args, cfg, runtime, rows, deploy_config)
    else:
        _trajectory(args, cfg, runtime, rows, deploy_config, full=args.phase == "full")


if __name__ == "__main__":
    main()
