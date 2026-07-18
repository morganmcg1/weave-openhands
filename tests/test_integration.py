from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

import pytest
from openhands.sdk import Agent, AgentContext, Conversation, Tool, register_tool
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.skills import Skill
from openhands.sdk.testing import TestLLM
from openhands.sdk.tool import Action, Observation, ToolDefinition, ToolExecutor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from weave_openhands import TracingConfig, instrument, is_instrumented

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class RaisingAction(Action):
    value: str = ""


class RaisingObservation(Observation):
    result: str = ""


class RaisingExecutor(ToolExecutor[RaisingAction, RaisingObservation]):
    def __call__(
        self, action: RaisingAction, conversation: Any = None
    ) -> RaisingObservation:
        raise ValueError(f"cannot process {action.value}")


class RaisingTool(ToolDefinition[RaisingAction, RaisingObservation]):
    name = "raising_tool"

    @classmethod
    def create(cls, conv_state: ConversationState | None = None) -> Sequence[Self]:
        return [
            cls(
                description="Always raises for tracing tests",
                action_type=RaisingAction,
                observation_type=RaisingObservation,
                executor=RaisingExecutor(),
            )
        ]


register_tool("WeaveOpenHandsRaisingTool", RaisingTool)


def finish_message(message: str, call_id: str = "finish-call") -> Message:
    return Message(
        role="assistant",
        content=[TextContent(text="Finishing the task")],
        tool_calls=[
            MessageToolCall(
                id=call_id,
                name="finish",
                arguments=json.dumps({"message": message}),
                origin="completion",
            )
        ],
    )


def raising_message() -> Message:
    return Message(
        role="assistant",
        content=[TextContent(text="Trying the raising tool")],
        tool_calls=[
            MessageToolCall(
                id="raise-call",
                name="raising_tool",
                arguments='{"value":"bad-input"}',
                origin="completion",
            )
        ],
    )


def integration_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope.name == "weave_openhands"
    ]


def span_named(spans: list[ReadableSpan], prefix: str) -> ReadableSpan:
    matches = [span for span in spans if span.name.startswith(prefix)]
    assert len(matches) == 1, [span.name for span in spans]
    return matches[0]


def parse_attribute(span: ReadableSpan, name: str) -> Any:
    assert span.attributes is not None
    return json.loads(str(span.attributes[name]))


def make_conversation(tmp_path, responses: list[Message | Exception]) -> Conversation:
    skill = Skill(
        name="repository-rules",
        description="Repository-specific coding rules",
        content="Always preserve public interfaces.",
        source="/workspace/.agents/skills/repository-rules/SKILL.md",
        is_agentskills_format=True,
    )
    context = AgentContext(
        skills=[skill],
        system_message_suffix="CUSTOM SYSTEM CONTEXT",
    )
    agent = Agent(
        llm=TestLLM.from_messages(responses, model="test-model"),
        tools=[],
        agent_context=context,
    )
    return Conversation(agent=agent, workspace=tmp_path, visualizer=None)


def assert_standard_tree(spans: list[ReadableSpan], conversation_id: str) -> None:
    root = span_named(spans, "invoke_agent")
    chat = span_named(spans, "chat")
    tool = span_named(spans, "execute_tool")

    assert root.parent is None
    assert chat.parent is not None
    assert tool.parent is not None
    assert chat.parent.span_id == root.context.span_id
    assert tool.parent.span_id == root.context.span_id
    assert {span.context.trace_id for span in spans} == {root.context.trace_id}
    for span in (root, chat, tool):
        assert span.attributes is not None
        assert span.attributes["gen_ai.conversation.id"] == conversation_id
        assert span.attributes["integration.name"] == "weave-openhands"
        assert span.attributes["integration.meta.package_name"] == "openhands-sdk"


def test_real_sync_conversation_records_full_context(
    tmp_path, trace_exporter: InMemorySpanExporter
) -> None:
    instrument(TracingConfig(agent_name="software-engineer"))
    instrument(TracingConfig(agent_name="software-engineer"))
    assert is_instrumented()
    conversation = make_conversation(tmp_path, [finish_message("Task complete")])
    conversation.send_message("Inspect SECRET-free context")

    conversation.run()

    spans = integration_spans(trace_exporter)
    assert [span.name for span in spans] == [
        "chat test-model",
        "execute_tool finish",
        "invoke_agent software-engineer",
    ]
    assert_standard_tree(spans, str(conversation.id))

    root = span_named(spans, "invoke_agent")
    chat = span_named(spans, "chat")
    tool = span_named(spans, "execute_tool")
    assert root.attributes is not None
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.agent.name"] == "software-engineer"
    assert root.attributes["weave.openhands.skills.available.count"] == 1
    assert "repository-rules" in str(
        root.attributes["weave.openhands.skills.available"]
    )
    assert "CUSTOM SYSTEM CONTEXT" in str(root.attributes["gen_ai.system_instructions"])
    assert "finish" in str(root.attributes["gen_ai.tool.definitions"])
    assert "SystemPromptEvent" in str(root.attributes["weave.openhands.context.events"])
    assert "ActionEvent" in str(root.attributes["weave.openhands.run.events"])
    assert "Task complete" in str(root.attributes["gen_ai.output.messages"])
    assert "api_key" not in str(root.attributes["weave.openhands.agent.config"])

    assert chat.attributes is not None
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.request.model"] == "test-model"
    assert "Inspect SECRET-free context" in str(
        chat.attributes["gen_ai.input.messages"]
    )
    assert "CUSTOM SYSTEM CONTEXT" in str(chat.attributes["gen_ai.system_instructions"])
    assert "finish-call" in str(chat.attributes["gen_ai.output.messages"])
    assert "finish" in str(chat.attributes["gen_ai.tool.definitions"])

    assert tool.attributes is not None
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "finish"
    assert tool.attributes["gen_ai.tool.call.id"] == "finish-call"
    assert "Task complete" in str(tool.attributes["gen_ai.tool.call.arguments"])
    assert "FinishObservation" in str(tool.attributes["gen_ai.tool.call.result"])


@pytest.mark.asyncio
async def test_real_async_conversation_has_the_same_span_contract(
    tmp_path, trace_exporter: InMemorySpanExporter
) -> None:
    instrument()
    conversation = make_conversation(tmp_path, [finish_message("Async complete")])
    conversation.send_message("Run asynchronously")

    await conversation.arun()

    spans = integration_spans(trace_exporter)
    assert_standard_tree(spans, str(conversation.id))
    assert "Async complete" in str(
        span_named(spans, "invoke_agent").attributes["gen_ai.output.messages"]
    )


def test_tool_errors_are_visible_and_do_not_break_the_trace_tree(
    tmp_path, trace_exporter: InMemorySpanExporter
) -> None:
    instrument()
    agent = Agent(
        llm=TestLLM.from_messages(
            [raising_message(), finish_message("Recovered after tool error")]
        ),
        tools=[Tool(name="WeaveOpenHandsRaisingTool")],
    )
    conversation = Conversation(agent=agent, workspace=tmp_path, visualizer=None)
    conversation.send_message("Exercise error handling")

    conversation.run()

    spans = integration_spans(trace_exporter)
    root = span_named(spans, "invoke_agent")
    tool_spans = [span for span in spans if span.name.startswith("execute_tool")]
    assert len(tool_spans) == 2
    raising_span = next(
        span for span in tool_spans if span.name.endswith("raising_tool")
    )
    finish_span = next(span for span in tool_spans if span.name.endswith("finish"))
    assert raising_span.status.status_code is StatusCode.ERROR
    assert finish_span.status.status_code is StatusCode.UNSET
    assert raising_span.parent is not None
    assert finish_span.parent is not None
    assert raising_span.parent.span_id == root.context.span_id
    assert finish_span.parent.span_id == root.context.span_id
    assert "cannot process bad-input" in str(
        raising_span.attributes["gen_ai.tool.call.result"]
    )
    assert root.status.status_code is StatusCode.UNSET


def test_llm_failure_closes_and_marks_chat_and_root_spans(
    tmp_path, trace_exporter: InMemorySpanExporter
) -> None:
    instrument()
    conversation = make_conversation(tmp_path, [RuntimeError("provider unavailable")])
    conversation.send_message("This call will fail")

    with pytest.raises(Exception, match="provider unavailable"):
        conversation.run()

    spans = integration_spans(trace_exporter)
    root = span_named(spans, "invoke_agent")
    chat = span_named(spans, "chat")
    assert root.status.status_code is StatusCode.ERROR
    assert chat.status.status_code is StatusCode.ERROR
    assert root.end_time is not None
    assert chat.end_time is not None


def test_exception_payloads_follow_the_content_policy(
    tmp_path, trace_exporter: InMemorySpanExporter
) -> None:
    instrument(TracingConfig(capture_content=False))
    conversation = make_conversation(tmp_path, [RuntimeError("PRIVATE ERROR")])
    conversation.send_message("PRIVATE PROMPT")

    with pytest.raises(Exception, match="PRIVATE ERROR"):
        conversation.run()

    spans = integration_spans(trace_exporter)
    serialized = json.dumps(
        [
            {
                "attributes": dict(span.attributes or {}),
                "status": span.status.description,
                "events": [dict(event.attributes or {}) for event in span.events],
            }
            for span in spans
        ],
        default=str,
    )
    assert "PRIVATE ERROR" not in serialized
    assert "PRIVATE PROMPT" not in serialized
    assert "builtins.RuntimeError" in serialized


def test_capture_content_false_keeps_structure_only(
    tmp_path, trace_exporter: InMemorySpanExporter
) -> None:
    instrument(TracingConfig(capture_content=False))
    conversation = make_conversation(tmp_path, [finish_message("PRIVATE RESULT")])
    conversation.send_message("PRIVATE PROMPT")

    conversation.run()

    spans = integration_spans(trace_exporter)
    serialized_attributes = json.dumps(
        [dict(span.attributes or {}) for span in spans], default=str
    )
    assert "PRIVATE PROMPT" not in serialized_attributes
    assert "PRIVATE RESULT" not in serialized_attributes
    assert "CUSTOM SYSTEM CONTEXT" not in serialized_attributes
    assert "gen_ai.input.messages" not in serialized_attributes
    assert "gen_ai.tool.call.arguments" not in serialized_attributes
    assert {span.attributes["gen_ai.operation.name"] for span in spans} == {
        "invoke_agent",
        "chat",
        "execute_tool",
    }


def test_content_transform_redacts_all_exported_payloads(
    tmp_path, trace_exporter: InMemorySpanExporter
) -> None:
    instrument(
        TracingConfig(
            content_transform=lambda value: value.replace("SECRET", "[REDACTED]")
        )
    )
    conversation = make_conversation(tmp_path, [finish_message("SECRET result")])
    conversation.send_message("SECRET prompt")

    conversation.run()

    spans = integration_spans(trace_exporter)
    serialized_attributes = json.dumps(
        [dict(span.attributes or {}) for span in spans], default=str
    )
    assert "SECRET" not in serialized_attributes
    assert "[REDACTED] prompt" in serialized_attributes
    assert "[REDACTED] result" in serialized_attributes


def test_multiple_turns_are_separate_traces_in_one_conversation(
    tmp_path, trace_exporter: InMemorySpanExporter
) -> None:
    instrument()
    conversation = make_conversation(
        tmp_path,
        [
            finish_message("First answer", "finish-1"),
            finish_message("Second answer", "finish-2"),
        ],
    )
    conversation.send_message("First turn")
    conversation.run()
    conversation.send_message("Second turn")
    conversation.run()

    spans = integration_spans(trace_exporter)
    roots = [span for span in spans if span.name.startswith("invoke_agent")]
    assert len(roots) == 2
    assert len({span.context.trace_id for span in roots}) == 2
    assert {span.attributes["gen_ai.conversation.id"] for span in roots} == {
        str(conversation.id)
    }
    assert "First answer" in str(roots[0].attributes["gen_ai.output.messages"])
    assert "Second answer" in str(roots[1].attributes["gen_ai.output.messages"])
