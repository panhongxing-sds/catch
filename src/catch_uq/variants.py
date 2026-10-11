"""Offline controlled generation with a separate GPT-5.6 verifier."""

from __future__ import annotations

import re
from collections import Counter

from .api import chat
from .prompts import text_perturbation_messages, verification_messages

FAMILIES = (
    "surface rewording",
    "role-preserving entity or scenario substitution",
    "condition reordering",
    "irrelevant distractor insertion",
)


def protected_anchors(text: str) -> tuple[tuple[str, int], ...]:
    pattern = (
        r"(?:\d+(?:\.\d+)?%?|<=|>=|!=|==|[+*/=<>]|"
        r"\b(?:mm|cm|m|km|mg|g|kg|ml|l|s|min|h|hz|pa|k|\u00b0c|\u00b0f)\b)"
    )
    matches = (
        match.lower() for match in re.findall(pattern, text, flags=re.IGNORECASE)
    )
    return tuple(sorted(Counter(matches).items()))


def generate_verified_variants(
    question: str,
    model: str = "deepseek-v4-pro",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_attempts_per_family: int = 8,
    *,
    generator: dict | None = None,
    verifier: dict | None = None,
) -> list[str]:
    generator = generator or {
        "model": model,
        "api_style": "chat_completions",
        "api_key_env": "DEEPSEEK_API_KEY",
    }
    verifier = verifier or {
        "model": "gpt-5.6",
        "api_style": "responses",
        "api_key_env": "OPENAI_API_KEY",
    }
    accepted: list[str] = []
    anchors = protected_anchors(question)
    for family in FAMILIES:
        for _ in range(max_attempts_per_family):
            messages = text_perturbation_messages(question)
            messages[-1]["content"] += f"\nRequired transformation type: {family}."
            candidate = chat(messages, generator, temperature=temperature, top_p=top_p)
            if candidate == question or candidate in accepted:
                continue
            if protected_anchors(candidate) != anchors:
                continue
            verdict = chat(verification_messages(question, candidate), verifier)
            if verdict.strip().upper() == "EQUIVALENT":
                accepted.append(candidate)
                break
        else:
            raise RuntimeError(
                f"Could not obtain a verified {family} variant after {max_attempts_per_family} attempts"
            )
    return accepted
