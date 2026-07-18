from __future__ import annotations

import json
from types import SimpleNamespace

from weave_openhands.config import TracingConfig
from weave_openhands.serialization import (
    json_dumps,
    message_to_semconv,
    messages_to_semconv,
    system_instructions,
    to_jsonable,
)


def content(text: str):
    return SimpleNamespace(type="text", text=text)


def test_messages_follow_genai_semantic_conventions() -> None:
    config = TracingConfig()
    messages = [
        SimpleNamespace(role="system", content=[content("Be precise")]),
        SimpleNamespace(role="user", content=[content("Inspect the repository")]),
        SimpleNamespace(
            role="assistant",
            content=[content("I will inspect it")],
            reasoning_content="Need the file list",
            thinking_blocks=[],
            responses_reasoning_item=None,
            tool_calls=[
                SimpleNamespace(
                    id="call-1", name="terminal", arguments='{"command":"ls"}'
                )
            ],
        ),
        SimpleNamespace(
            role="tool",
            content=[content("README.md")],
            tool_call_id="call-1",
        ),
    ]

    assert system_instructions(messages, config) == [
        {"type": "text", "content": "Be precise"}
    ]
    converted = messages_to_semconv(messages, config)
    assert [message["role"] for message in converted] == [
        "user",
        "assistant",
        "tool",
    ]
    assert converted[1]["parts"] == [
        {"type": "reasoning", "content": "Need the file list"},
        {"type": "text", "content": "I will inspect it"},
        {
            "type": "tool_call",
            "id": "call-1",
            "name": "terminal",
            "arguments": '{"command":"ls"}',
        },
    ]
    assert converted[2]["parts"] == [
        {"type": "tool_call_response", "response": "README.md", "id": "call-1"}
    ]


def test_opaque_reasoning_payloads_are_never_serialized() -> None:
    config = TracingConfig()
    value = {
        "visible": "keep me",
        "encrypted_content": "ciphertext",
        "nested": {"signature": "signature-value", "text": "visible thought"},
    }

    serialized = json_dumps(value, config)

    assert json.loads(serialized) == {
        "nested": {"text": "visible thought"},
        "visible": "keep me",
    }
    assert "ciphertext" not in serialized
    assert "signature-value" not in serialized


def test_content_transform_reaches_every_string() -> None:
    config = TracingConfig(content_transform=lambda value: value.replace("SECRET", "X"))
    value = {"SECRET-key": ["SECRET-value", {"value": "SECRET-nested"}]}

    transformed = to_jsonable(value, config)

    # Mapping keys describe schema fields and are intentionally stable.
    assert transformed == {"SECRET-key": ["X-value", {"value": "X-nested"}]}


def test_tool_response_with_no_content_is_valid() -> None:
    message = SimpleNamespace(role="tool", content=[], tool_call_id="call-2")

    assert message_to_semconv(message, TracingConfig()) == {
        "role": "tool",
        "parts": [{"type": "tool_call_response", "response": "", "id": "call-2"}],
    }
