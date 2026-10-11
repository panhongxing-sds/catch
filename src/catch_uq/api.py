"""Small explicit API adapters; credentials are read only from environment variables."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache


def _request_json(url: str, payload: dict, api_key_env: str, attempts: int = 5) -> dict:
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(
            f"{api_key_env} is required; API credentials are never stored in output records"
        )
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise RuntimeError(
                    f"API request rejected with HTTP {error.code}; check endpoint/model access"
                ) from None
        except (urllib.error.URLError, TimeoutError):
            pass
        if attempt + 1 < attempts:
            time.sleep(2**attempt)
    raise RuntimeError(f"API request failed after {attempts} attempts")


@lru_cache(maxsize=1)
def _local_generator(model_name: str):
    from .generation import load_model_and_tokenizer

    return load_model_and_tokenizer(model_name)


def chat(
    messages: list[dict],
    spec: dict,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Use DeepSeek Chat Completions, OpenAI Responses, or an HF generator."""
    style = spec.get("api_style", "chat_completions")
    model = spec["model"]
    if style == "huggingface":
        import torch

        from .generation import GenerationSettings, generate_greedy

        local_model, tokenizer = _local_generator(model)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        seed = int(spec.get("seed", 42))
        # Seed once at construction/use-sequence level, not repeatedly per resampling attempt.
        if not getattr(local_model, "_catch_generator_seeded", False):
            torch.manual_seed(seed)
            local_model._catch_generator_seeded = True
        result = generate_greedy(
            local_model,
            tokenizer,
            prompt,
            GenerationSettings(
                max_new_tokens=int(spec.get("max_new_tokens", 2048)),
                do_sample=True,
                temperature=temperature if temperature is not None else 0.8,
                top_p=top_p if top_p is not None else 0.95,
            ),
        )
        return result["response"].strip()
    if style == "responses":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip(
            "/"
        )
        payload = {"model": model, "input": messages, "store": False}
        if spec.get("reasoning_effort"):
            payload["reasoning"] = {"effort": spec["reasoning_effort"]}
        # The paper specifies a separate GPT-5.6 verification call, not verifier sampling parameters.
        result = _request_json(
            f"{base}/responses", payload, spec.get("api_key_env", "OPENAI_API_KEY")
        )
        if result.get("status") not in {None, "completed"}:
            raise RuntimeError("Verifier did not return a completed response")
        return "".join(
            block.get("text", "")
            for item in result.get("output", [])
            if item.get("type") == "message"
            for block in item.get("content", [])
            if block.get("type") == "output_text"
        ).strip()
    if style != "chat_completions":
        raise ValueError(f"Unknown API style: {style}")
    base = os.environ.get(
        spec.get("base_url_env", "DEEPSEEK_BASE_URL"),
        spec.get("base_url", "https://api.deepseek.com"),
    ).rstrip("/")
    payload = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    result = _request_json(
        f"{base}/chat/completions", payload, spec.get("api_key_env", "DEEPSEEK_API_KEY")
    )
    return str(result["choices"][0]["message"]["content"]).strip()
