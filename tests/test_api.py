import pytest

from catch_uq import api


def test_responses_verifier_parses_output_text_and_keeps_credentials_external(
    monkeypatch,
):
    calls = []

    def fake_request(url, payload, key_env):
        calls.append((url, payload, key_env))
        return {
            "status": "completed",
            "output": [
                {"type": "reasoning", "content": []},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "EQUIVALENT"}],
                },
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    messages = [{"role": "user", "content": "Check these two questions"}]
    assert (
        api.chat(messages, {"model": "gpt-5.6", "api_style": "responses"})
        == "EQUIVALENT"
    )
    assert calls[0][1] == {"model": "gpt-5.6", "input": messages, "store": False}
    assert calls[0][2] == "OPENAI_API_KEY"


def test_generator_sampling_arguments_are_forwarded(monkeypatch):
    def fake_request(url, payload, key_env):
        assert payload["temperature"] == 0.8 and payload["top_p"] == 0.95
        return {"choices": [{"message": {"content": "Variant"}}]}

    monkeypatch.setattr(api, "_request_json", fake_request)
    assert (
        api.chat([], {"model": "deepseek-v4-pro"}, temperature=0.8, top_p=0.95)
        == "Variant"
    )


def test_missing_credentials_fail_before_network_access(monkeypatch):
    monkeypatch.delenv("CATCH_TEST_NONEXISTENT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="required"):
        api._request_json("https://example.invalid", {}, "CATCH_TEST_NONEXISTENT_KEY")
