import catch_uq.variants as variants_module
from catch_uq.variants import generate_verified_variants, protected_anchors


def test_protected_anchors_preserve_numbers_units_and_relations_as_multiset():
    original = "A 6.5 m cable and a 2 m cable satisfy 6.5 > 2."
    reordered = "Because 2 < 6.5 is equivalent, compare a 2 M cable with a 6.5 M cable."
    assert protected_anchors(original) != protected_anchors(reordered)


def test_protected_anchors_ignore_surface_order_but_not_content():
    original = "Use 3 kg and 4 kg, then compute 3+4."
    reordered = "Compute 3+4 after taking 4 kg and 3 kg."
    changed = "Compute 3+5 after taking 5 kg and 3 kg."
    assert protected_anchors(original) == protected_anchors(reordered)
    assert protected_anchors(original) != protected_anchors(changed)


def test_rephrase_pipeline_uses_four_families_and_independent_verification(monkeypatch):
    question = "A package has mass 3 kg. What is its mass?"
    candidates = iter(
        [
            "Reworded: a package has mass 3 kg. What is its mass?",
            "A parcel has mass 3 kg. What is its mass?",
            "What is the mass, given that a package has mass 3 kg?",
            "A package has mass 3 kg. The sky is blue. What is its mass?",
        ]
    )
    calls = []

    def fake_chat(messages, model, temperature, top_p, thinking=False, request_attempts=5):
        calls.append((messages, model, temperature, top_p, thinking, request_attempts))
        return "EQUIVALENT" if thinking else next(candidates)

    monkeypatch.setattr(variants_module, "_chat", fake_chat)
    output = generate_verified_variants(question)
    assert len(output) == 4
    generation_calls = calls[0::2]
    verifier_calls = calls[1::2]
    assert all(call[1:4] == ("deepseek-v4-pro", 0.8, 0.95) for call in generation_calls)
    assert all(call[4] is False for call in generation_calls)
    assert all(call[1:5] == ("deepseek-v4-pro", 0.0, 1.0, True) for call in verifier_calls)
