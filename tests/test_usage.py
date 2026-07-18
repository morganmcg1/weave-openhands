from types import SimpleNamespace

from weave_openhands.instrumentation import _provider_name, _usage_attributes


def test_provider_name_uses_otel_genai_values() -> None:
    assert _provider_name("openai/gpt-4.1-mini") == "openai"
    assert _provider_name("anthropic/claude-haiku-4-5-20251001") == "anthropic"
    assert _provider_name("azure/gpt-4.1-mini") == "azure.ai.openai"
    assert _provider_name("bedrock/anthropic.claude-haiku") == "aws.bedrock"
    assert _provider_name("test-model") == ""


def test_usage_attributes_cover_provider_and_openhands_token_shapes() -> None:
    raw_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=18,
            cache_read_input_tokens=40,
            cache_creation_input_tokens=25,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
        )
    )

    assert _usage_attributes(raw_response, response=None) == {
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 18,
        "gen_ai.usage.cache_read.input_tokens": 40,
        "gen_ai.usage.cache_creation.input_tokens": 25,
        "gen_ai.usage.reasoning.output_tokens": 7,
    }
