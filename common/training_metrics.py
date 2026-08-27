"""Append-only training metrics with resume-safe truncation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, output_dir: str | Path, resume_step: int, ema_decay: float = 0.9):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "metrics.jsonl"
        self.ema_decay = float(ema_decay)
        self.loss_ema: float | None = None
        retained = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record["step"]) <= resume_step:
                    retained.append(record)
        if retained:
            self.loss_ema = float(retained[-1].get("loss_ema", retained[-1]["loss"]))
        temporary = self.path.with_suffix(".jsonl.incomplete")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in retained:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, self.path)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        loss = float(record["loss"])
        self.loss_ema = (
            loss
            if self.loss_ema is None
            else self.ema_decay * self.loss_ema + (1.0 - self.ema_decay) * loss
        )
        record["loss_ema"] = self.loss_ema
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        return record
