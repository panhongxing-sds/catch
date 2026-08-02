"""Validated local benchmark and perturbation data loading."""

from __future__ import annotations

from pathlib import Path

from .io import read_jsonl

QUESTION_KEYS = ("question", "problem", "prompt")
ANSWER_KEYS = ("answer", "reference", "solution", "target")


def _first(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if row.get(key) is not None:
            return str(row[key]).strip()
    return ""


def load_benchmark(path: str | Path, dataset: str) -> list[dict]:
    output: list[dict] = []
    for index, raw in enumerate(read_jsonl(path)):
        question = _first(raw, QUESTION_KEYS)
        reference = _first(raw, ANSWER_KEYS)
        if not question or not reference:
            raise ValueError(f"{path}: row {index + 1} needs a question/problem and answer/reference")
        output.append(
            {
                "id": str(raw.get("id", f"{dataset}_{index}")),
                "dataset": dataset,
                "question": question,
                "reference": reference,
            }
        )
    return output


def load_variants(path: str | Path) -> dict[str, list[str]]:
    variants: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        record_id = str(row["id"])
        candidates = [str(value).strip() for value in row.get("variants", []) if str(value).strip()]
        if len(candidates) != 4:
            raise ValueError(f"{path}: {record_id} has {len(candidates)} variants; Appendix A.2.1 requires 4")
        variants[record_id] = candidates
    return variants


def validate_scope(rows: list[dict], expected: int, dataset: str) -> None:
    if len(rows) != expected:
        raise ValueError(
            f"{dataset}: found {len(rows)} examples, expected exactly {expected} for Appendix D.5.1"
        )
