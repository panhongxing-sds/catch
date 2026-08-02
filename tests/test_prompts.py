from catch_uq.prompts import reasoning_messages


def test_qwen_reasoning_prompt_matches_table_4():
    messages = reasoning_messages("PROBLEM", "Qwen/Qwen2.5-3B-Instruct")
    assert messages[0]["content"] == (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )
    assert messages[1]["content"] == (
        "PROBLEM Let's think step by step and output the final answer within \\boxed{}."
    )


def test_llama_reasoning_prompt_has_full_step_template_and_fixed_conclusion():
    messages = reasoning_messages("PROBLEM", "meta-llama/Llama-3.1-8B-Instruct")
    system = messages[0]["content"]
    assert "## Step 1: [Concise description]" in system
    assert "## Step 2: [Concise description]" in system
    assert "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct." in system
    assert messages[1] == {"role": "user", "content": "PROBLEM"}
