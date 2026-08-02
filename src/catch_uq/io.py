"""Small JSONL and configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
    return rows


def append_jsonl(path: str | Path, row: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
