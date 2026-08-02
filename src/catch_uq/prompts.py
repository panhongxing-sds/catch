"""Prompt templates from Appendix A.3."""

from __future__ import annotations

TEXT_PERTURBATION_SYSTEM = """Rewrite the following math problem into a mathematically equivalent variant.

Requirements:
1. Preserve all numerical values, units, mathematical relations, constraints, and the final question target.
2. Do not solve the problem.
3. Do not add hints or explanations.
4. Do not change the expected final answer.
5. Apply one of the following transformation types when possible:
   (a) surface rewording;
   (b) role-preserving entity or scenario substitution;
   (c) condition reordering;
   (d) irrelevant distractor insertion.
6. Any added distractor must be clearly irrelevant to the calculation and must not introduce a new condition.
7. Output only the rewritten problem."""

SEMANTIC_VERIFICATION_SYSTEM = """Determine whether the candidate problem is mathematically and semantically equivalent to the original problem.

Check whether the candidate preserves:
1. all numerical values and units;
2. all mathematical relations and operators;
3. all assumptions and constraints;
4. the quantity being asked for;
5. the expected final answer.

Surface wording, entity names, condition order, and clearly irrelevant distractors may differ. A candidate is not equivalent if it adds, removes, or changes any information that can affect the solution.

Return only EQUIVALENT or NOT EQUIVALENT."""

LLAMA_MATH_SYSTEM = """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:
Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem."""


def reasoning_messages(problem: str, model_name: str) -> list[dict[str, str]]:
    """Return the Qwen or Llama message template described in Table 4."""
    if "qwen" in model_name.lower():
        return [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"{problem} Let's think step by step and output the final answer within \\boxed{{}}.",
            },
        ]
    return [
        {"role": "system", "content": LLAMA_MATH_SYSTEM},
        {"role": "user", "content": problem},
    ]


def text_perturbation_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TEXT_PERTURBATION_SYSTEM},
        {"role": "user", "content": f"Problem:\n{problem}"},
    ]


def verification_messages(original: str, candidate: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SEMANTIC_VERIFICATION_SYSTEM},
        {
            "role": "user",
            "content": f"Original problem:\n{original}\nCandidate problem:\n{candidate}",
        },
    ]
