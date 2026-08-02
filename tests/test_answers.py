from catch_uq.answers import answers_equivalent, extract_final_answer, normalize_answer


def test_boxed_answer_has_priority_and_span():
    result = extract_final_answer("First 12. Therefore $\\boxed{\\frac{1}{2}}$.")
    assert result.raw == r"\frac{1}{2}"
    assert answers_equivalent(result.normalized, "0.5")
    assert result.char_start is not None


def test_marker_then_final_line_fallback():
    assert extract_final_answer("work\nThe final answer is: 42.").normalized == "42"
    assert extract_final_answer("work\n17").normalized == "17"
    symbolic = extract_final_answer("Therefore, the answer is $x^2+1$. I hope it is correct.")
    assert answers_equivalent(symbolic.normalized, "x^2+1")


def test_numeric_and_symbolic_equivalence():
    assert answers_equivalent("2/4", "0.5")
    assert answers_equivalent("x+x", "2*x")
    assert answers_equivalent(r"\sqrt{4}", "2")
    assert answers_equivalent(r"\frac{\pi}{2}", "pi/2")
    assert answers_equivalent("x = 7", "7")
    assert normalize_answer("25%") == "0.25"
    assert normalize_answer("(1, 2)") == "(1,2)"
    assert normalize_answer("1,000") == "1000"
