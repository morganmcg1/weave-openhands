from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from examples.logging_showcase import redact, run_scenario
from weave_openhands import TracingConfig, instrument


def integration_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope.name == "weave_openhands"
    ]


@pytest.mark.parametrize(
    ("scenario", "expected_roots", "expected_chats", "expected_tools"),
    [
        ("basic", 1, 1, 1),
        ("multi-turn", 2, 2, 2),
        ("tools-and-skills", 1, 3, 3),
        ("error-recovery", 1, 3, 3),
        ("privacy-async", 1, 1, 1),
    ],
)
def test_logging_showcase_scenarios(
    scenario: str,
    expected_roots: int,
    expected_chats: int,
    expected_tools: int,
    tmp_path,
    trace_exporter: InMemorySpanExporter,
) -> None:
    config = (
        TracingConfig(content_transform=redact)
        if scenario == "privacy-async"
        else TracingConfig()
    )
    instrument(config)

    result = run_scenario(scenario, tmp_path)

    spans = integration_spans(trace_exporter)
    roots = [span for span in spans if span.name.startswith("invoke_agent")]
    chats = [span for span in spans if span.name.startswith("chat")]
    tools = [span for span in spans if span.name.startswith("execute_tool")]
    assert len(roots) == expected_roots
    assert len(chats) == expected_chats
    assert len(tools) == expected_tools
    assert result.turns == expected_roots
    assert result.llm_calls == expected_chats

    if scenario == "multi-turn":
        assert len({span.context.trace_id for span in roots}) == 2
        assert {span.attributes["gen_ai.conversation.id"] for span in spans} == {
            result.conversation_id
        }
    elif scenario == "tools-and-skills":
        root = roots[0]
        assert (
            "observability-policy"
            in root.attributes["weave.openhands.skills.activated"]
        )
        assert "demo-guidance" in root.attributes["weave.openhands.skills.invoked"]
        assert any(span.name == "execute_tool showcase" for span in tools)
    elif scenario == "error-recovery":
        failed = next(span for span in tools if span.name == "execute_tool showcase")
        assert failed.status.status_code is StatusCode.ERROR
        assert any(span.status.status_code is StatusCode.UNSET for span in tools)
    elif scenario == "privacy-async":
        exported = json.dumps(
            [dict(span.attributes or {}) for span in spans],
            default=str,
        )
        assert "SECRET" not in exported
        assert "[REDACTED]" in exported
