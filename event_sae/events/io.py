"""Tiny JSONL helpers (load + write)."""

from __future__ import annotations

import json
from pathlib import Path

from event_sae import resolve_groot_artifact_path


def load_jsonl(path: str | Path) -> list[dict]:
    path = resolve_groot_artifact_path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = resolve_groot_artifact_path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
