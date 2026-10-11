import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from catch_uq.generation import GenerationSettings, generate_greedy
from catch_uq.perturbations import LowRankGaussianPerturber


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompt, return_tensors):
        return transformers.BatchEncoding(
            {
                "input_ids": torch.tensor([[2, 3]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }
        )

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids if int(i) > 1)


def test_actual_hf_decode_captures_unprocessed_logits_and_restores_qk():
    torch.manual_seed(42)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.eval()
    # Processed scores suppress these tokens; the measured probabilities must
    # still come from raw model logits, including all ten retained candidates.
    model.generation_config.suppress_tokens = list(range(3, 32))
    model.generation_config.eos_token_id = None
    tokenizer = TinyTokenizer()
    expected = torch.log_softmax(
        model(input_ids=torch.tensor([[2, 3]])).logits[0, -1].float(), dim=-1
    )
    result = generate_greedy(
        model, tokenizer, "prompt", GenerationSettings(max_new_tokens=3)
    )
    assert len(result["token_trace"]) == 3
    assert result["token_trace"][0]["top_logprobs"] == pytest.approx(
        torch.topk(expected, 10).values.tolist()
    )
    assert all(
        math.isfinite(p) for t in result["token_trace"] for p in t["top_logprobs"]
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    perturber = LowRankGaussianPerturber(model)
    with perturber.sampled(42):
        generate_greedy(
            model, tokenizer, "prompt", GenerationSettings(max_new_tokens=2)
        )
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
