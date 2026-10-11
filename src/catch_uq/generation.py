"""Hugging Face greedy generation with CATCH token traces."""

from __future__ import annotations

from dataclasses import dataclass

from .answers import extract_final_answer
from .prompts import reasoning_messages


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 2048
    top_m: int = 10
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


def load_model_and_tokenizer(
    model_name: str, device_map: str = "auto", precision: str = "bfloat16"
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtypes = {"bfloat16": torch.bfloat16}
    if precision not in dtypes:
        raise ValueError(f"Unsupported CATCH precision: {precision}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtypes[precision],
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, problem: str, model_name: str) -> str:
    return tokenizer.apply_chat_template(
        reasoning_messages(problem, model_name),
        tokenize=False,
        add_generation_prompt=True,
    )


def _token_trace(
    generated_ids, score_steps, tokenizer, top_m: int
) -> tuple[list[dict], list[str]]:
    import torch

    trace: list[dict] = []
    token_texts: list[str] = []
    for token_id, logits in zip(generated_ids.tolist(), score_steps):
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)
        values, indices = torch.topk(logprobs, k=min(top_m, logprobs.numel()))
        chosen = float(logprobs[token_id].item())
        probabilities = torch.exp(logprobs)
        full_entropy = float(-(probabilities * logprobs).sum().item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        token_texts.append(token_text)
        trace.append(
            {
                "token_id": int(token_id),
                "token": token_text,
                "logprob": chosen,
                "entropy": full_entropy,
                "top_token_ids": [int(x) for x in indices.tolist()],
                "top_logprobs": [float(x) for x in values.tolist()],
            }
        )
    return trace, token_texts


def generate_greedy(
    model, tokenizer, prompt: str, settings: GenerationSettings
) -> dict:
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    sampling = {}
    if settings.do_sample:
        if settings.temperature <= 0:
            raise ValueError("Stochastic decoding requires positive temperature")
        sampling = {"temperature": settings.temperature, "top_p": settings.top_p}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            pad_token_id=pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_logits=True,
            **sampling,
        )
    generated_ids = output.sequences[0, prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    extracted = extract_final_answer(response)
    trace, _ = _token_trace(generated_ids, output.logits, tokenizer, settings.top_m)
    # Decode prefixes together: byte-level tokenizers need not decode additively.
    prefixes = [
        tokenizer.decode(generated_ids[:i], skip_special_tokens=True)
        for i in range(1, len(generated_ids) + 1)
    ]
    alpha = None
    if extracted.char_start is not None:
        for i, prefix in enumerate(prefixes):
            # An incomplete UTF-8 prefix can end in replacement characters.
            stable_prefix = prefix.rstrip("\ufffd")
            if len(stable_prefix) > extracted.char_start:
                alpha = i
                break
    return {
        "response": response,
        "answer_raw": extracted.raw,
        "answer": extracted.normalized,
        "answer_start_token": alpha,
        "token_trace": trace,
        "decoding": {
            "do_sample": settings.do_sample,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
        },
    }
