"""Final-answer extraction, normalization, and equivalence checks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


@dataclass(frozen=True)
class ExtractedAnswer:
    raw: str
    normalized: str
    char_start: int | None


def _clean_fallback(raw: str, absolute_start: int) -> tuple[str, int]:
    math_spans = list(re.finditer(r"\$([^$]+)\$", raw))
    if math_spans:
        match = math_spans[-1]
        return match.group(1).strip(), absolute_start + match.start(1)
    raw = re.split(
        r"\s+(?:I\s+hope|This\s+is|Thus\s+the)\b",
        raw,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    raw = re.split(r"(?<!\d)\.(?:\s|$)", raw, maxsplit=1)[0]
    cleaned = raw.strip().strip(":").strip().rstrip(".$")
    offset = raw.find(cleaned) if cleaned else 0
    return cleaned, absolute_start + max(0, offset)


def _last_boxed(text: str) -> tuple[str, int] | None:
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        brace = text.find("{", start)
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace + 1 : i].strip(), start
    return None


def extract_final_answer(text: str) -> ExtractedAnswer:
    boxed = _last_boxed(text)
    if boxed:
        raw, start = boxed
        return ExtractedAnswer(raw, normalize_answer(raw), start)

    marker = re.compile(
        r"(?:final\s+answer(?:\s+is)?|answer\s+is|therefore(?:,?\s+the\s+answer\s+is)?)"
        r"\s*[:=]?\s*([^\n]+)",
        re.IGNORECASE,
    )
    matches = list(marker.finditer(text))
    if matches:
        raw, start = _clean_fallback(matches[-1].group(1), matches[-1].start(1))
        return ExtractedAnswer(raw, normalize_answer(raw), start)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    raw_line = lines[-1] if lines else ""
    line_start = text.rfind(raw_line) if raw_line else 0
    raw, start = _clean_fallback(raw_line, line_start)
    return ExtractedAnswer(raw, normalize_answer(raw), start if raw else None)


def normalize_answer(value: str) -> str:
    s = value.strip().strip("$")
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
    s = s.replace("\\,", "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    previous = None
    while previous != s:
        previous = s
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\pi", "pi")
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^(-?[A-Za-z0-9]+)", r"**(\1)", s)
    if "=" in s:
        left, right = s.split("=", 1)
        if re.fullmatch(r"\s*[A-Za-z][A-Za-z0-9_]*\s*", left):
            s = right
    s = re.sub(r"\s+", "", s).lower()
    if s.endswith("%"):
        try:
            return str(float(s[:-1]) / 100.0)
        except ValueError:
            pass
    try:
        return f"{float(s):.12g}"
    except ValueError:
        return s


def answers_equivalent(left: str, right: str, tolerance: float = 1e-6) -> bool:
    a, b = normalize_answer(left), normalize_answer(right)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
        return math.isclose(fa, fb, rel_tol=tolerance, abs_tol=tolerance)
    except ValueError:
        pass
    try:
        local = {"pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt}
        ea = parse_expr(a, local_dict=local, transformations=_TRANSFORMATIONS, evaluate=True)
        eb = parse_expr(b, local_dict=local, transformations=_TRANSFORMATIONS, evaluate=True)
        return sp.simplify(ea - eb) == 0
    except Exception:  # noqa: BLE001 - model text can trigger several SymPy parser exceptions
        return False
