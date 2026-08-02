"""Offline DeepSeek-compatible generation and verification of text variants."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter

from .prompts import text_perturbation_messages, verification_messages

FAMILIES = (
    "surface rewording",
    "role-preserving entity or scenario substitution",
    "condition reordering",
    "irrelevant distractor insertion",
)


def _chat(
    messages: list[dict],
    model: str,
    temperature: float,
    top_p: float,
    thinking: bool = False,
    request_attempts: int = 5,
) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required and is never written to disk")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    payload = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p}
    if thinking:
        payload["thinking"] = {"type": "enabled"}
    for attempt in range(request_attempts):
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            return str(result["choices"][0]["message"]["content"]).strip()
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
        except urllib.error.URLError:
            pass
        if attempt + 1 < request_attempts:
            time.sleep(2**attempt)
    raise RuntimeError(f"API request failed after {request_attempts} attempts")


def protected_anchors(text: str) -> tuple[tuple[str, int], ...]:
    pattern = (
        r"(?:\d+(?:\.\d+)?%?|<=|>=|!=|==|[+*/=<>]|"
        r"\b(?:mm|cm|m|km|mg|g|kg|ml|l|s|min|h|hz|pa|k|\u00b0c|\u00b0f)\b)"
    )
    matches = (match.lower() for match in re.findall(pattern, text, flags=re.IGNORECASE))
    return tuple(sorted(Counter(matches).items()))


def generate_verified_variants(
    question: str,
    model: str = "deepseek-v4-pro",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_attempts_per_family: int = 8,
) -> list[str]:
    accepted: list[str] = []
    anchors = protected_anchors(question)
    for family in FAMILIES:
        for _ in range(max_attempts_per_family):
            messages = text_perturbation_messages(question)
            messages[-1]["content"] += f"\nRequired transformation type: {family}."
            candidate = _chat(messages, model, temperature=temperature, top_p=top_p)
            if candidate == question or candidate in accepted:
                continue
            if protected_anchors(candidate) != anchors:
                continue
            verdict = _chat(
                verification_messages(question, candidate),
                model,
                temperature=0.0,
                top_p=1.0,
                thinking=True,
            )
            if verdict.strip().upper() == "EQUIVALENT":
                accepted.append(candidate)
                break
        else:
            raise RuntimeError(
                f"Could not obtain a verified {family} variant after {max_attempts_per_family} attempts"
            )
    return accepted
