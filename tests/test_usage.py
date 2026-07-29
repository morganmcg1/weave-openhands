from types import SimpleNamespace

from openhands.sdk.llm.utils.metrics import Metrics

from weave_openhands.instrumentation import (
    _finish_reasons,
    _metrics_baseline,
    _metrics_usage_since,
    _provider_name,
    _usage_attributes,
)


def test_provider_name_uses_otel_genai_values() -> None:
    assert _provider_name("openai/gpt-4.1-mini") == "openai"
    assert _provider_name("anthropic/claude-haiku-4-5-20251001") == "anthropic"
    assert _provider_name("azure/gpt-4.1-mini") == "azure.ai.openai"
    assert _provider_name("bedrock/anthropic.claude-haiku") == "aws.bedrock"
    assert _provider_name("test-model") == ""


def test_responses_api_finish_reasons_are_always_present() -> None:
    completed = SimpleNamespace(status="completed")
    assert _finish_reasons(completed, SimpleNamespace(tool_calls=[object()])) == [
        "tool_call"
    ]
    assert _finish_reasons(completed, SimpleNamespace(tool_calls=[])) == ["stop"]
    assert _finish_reasons(
        SimpleNamespace(status="incomplete"), SimpleNamespace(tool_calls=[])
    ) == ["incomplete"]


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

    assert _usage_attributes(raw_response) == {
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 18,
        "gen_ai.usage.cache_read.input_tokens": 40,
        "gen_ai.usage.cache_creation.input_tokens": 25,
        "gen_ai.usage.reasoning.output_tokens": 7,
    }


def test_missing_provider_usage_falls_back_to_each_calls_metrics_delta() -> None:
    metrics = Metrics(model_name="test-model")
    llm = SimpleNamespace(metrics=metrics)

    first_baseline = _metrics_baseline(llm)
    metrics.add_token_usage(
        prompt_tokens=100,
        completion_tokens=20,
        cache_read_tokens=30,
        cache_write_tokens=10,
        reasoning_tokens=5,
        context_window=1_000,
        response_id="first",
    )
    first_usage = _metrics_usage_since(llm, first_baseline)

    second_baseline = _metrics_baseline(llm)
    metrics.add_token_usage(
        prompt_tokens=40,
        completion_tokens=8,
        cache_read_tokens=12,
        cache_write_tokens=3,
        reasoning_tokens=2,
        context_window=1_000,
        response_id="second",
    )
    second_usage = _metrics_usage_since(llm, second_baseline)

    raw_response = SimpleNamespace(usage=None)
    assert _usage_attributes(raw_response, first_usage) == {
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 20,
        "gen_ai.usage.cache_read.input_tokens": 30,
        "gen_ai.usage.cache_creation.input_tokens": 10,
        "gen_ai.usage.reasoning.output_tokens": 5,
    }
    assert _usage_attributes(raw_response, second_usage) == {
        "gen_ai.usage.input_tokens": 40,
        "gen_ai.usage.output_tokens": 8,
        "gen_ai.usage.cache_read.input_tokens": 12,
        "gen_ai.usage.cache_creation.input_tokens": 3,
        "gen_ai.usage.reasoning.output_tokens": 2,
    }
