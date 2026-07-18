from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from weave_openhands.config import TracingConfig

_OPAQUE_REASONING_KEYS = {"encrypted_content", "signature"}


def json_dumps(value: Any, config: TracingConfig) -> str:
    return json.dumps(
        to_jsonable(value, config),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def to_jsonable(value: Any, config: TracingConfig) -> Any:
    """Convert SDK values to JSON while omitting opaque reasoning payloads.

    OpenHands explicitly marks encrypted reasoning as data that must not be logged.
    Human-readable reasoning, prompts, tool arguments, and tool results remain
    available when content capture is enabled.
    """

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return config.transform(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value, config)
    if isinstance(value, BaseModel):
        try:
            dumped = value.model_dump(mode="json", exclude_none=True)
        except Exception:
            dumped = value.model_dump(exclude_none=True)
        return to_jsonable(dumped, config)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value), config)
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item, config)
            for key, item in value.items()
            if str(key) not in _OPAQUE_REASONING_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item, config) for item in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item, config) for item in value]
    return config.transform(str(value))


def message_to_semconv(message: Any, config: TracingConfig) -> dict[str, Any]:
    role = str(getattr(message, "role", "user"))
    parts: list[dict[str, Any]] = []

    if role == "tool":
        response = _content_text(getattr(message, "content", ()), config)
        part: dict[str, Any] = {
            "type": "tool_call_response",
            "response": response,
        }
        if tool_call_id := getattr(message, "tool_call_id", None):
            part["id"] = config.transform(str(tool_call_id))
        parts.append(part)
        return {"role": role, "parts": parts}

    reasoning = _reasoning_text(message, config)
    if reasoning:
        parts.append({"type": "reasoning", "content": reasoning})

    for content in getattr(message, "content", ()) or ():
        content_type = getattr(content, "type", "")
        if content_type == "text" or hasattr(content, "text"):
            parts.append(
                {
                    "type": "text",
                    "content": config.transform(str(getattr(content, "text", ""))),
                }
            )
        elif content_type == "image" or hasattr(content, "image_urls"):
            for url in getattr(content, "image_urls", ()) or ():
                parts.append(
                    {
                        "type": "uri",
                        "modality": "image",
                        "uri": config.transform(str(url)),
                    }
                )

    for tool_call in getattr(message, "tool_calls", ()) or ():
        parts.append(
            {
                "type": "tool_call",
                "id": config.transform(str(getattr(tool_call, "id", ""))),
                "name": config.transform(str(getattr(tool_call, "name", ""))),
                "arguments": config.transform(str(getattr(tool_call, "arguments", ""))),
            }
        )

    return {"role": role, "parts": parts}


def messages_to_semconv(
    messages: Iterable[Any], config: TracingConfig
) -> list[dict[str, Any]]:
    return [
        message_to_semconv(message, config)
        for message in messages
        if getattr(message, "role", None) != "system"
    ]


def system_instructions(
    messages: Iterable[Any], config: TracingConfig
) -> list[dict[str, str]]:
    instructions: list[dict[str, str]] = []
    for message in messages:
        if getattr(message, "role", None) != "system":
            continue
        for content in getattr(message, "content", ()) or ():
            text = getattr(content, "text", None)
            if text is not None:
                instructions.append(
                    {"type": "text", "content": config.transform(str(text))}
                )
    return instructions


def tool_definitions(tools: Iterable[Any], config: TracingConfig) -> list[Any]:
    definitions: list[Any] = []
    for tool in tools:
        try:
            definition = tool.to_openai_tool(add_security_risk_prediction=True)
        except Exception:
            try:
                definition = tool.to_mcp_tool()
            except Exception:
                definition = {
                    "name": getattr(tool, "name", type(tool).__name__),
                    "description": getattr(tool, "description", ""),
                }
        definitions.append(to_jsonable(definition, config))
    return definitions


def event_message(event: Any) -> Any | None:
    converter = getattr(event, "to_llm_message", None)
    if converter is None:
        return None
    try:
        return converter()
    except Exception:
        return None


def _content_text(content_items: Iterable[Any], config: TracingConfig) -> str:
    values: list[str] = []
    for content in content_items or ():
        if hasattr(content, "text"):
            values.append(config.transform(str(content.text)))
        elif hasattr(content, "image_urls"):
            values.extend(config.transform(str(url)) for url in content.image_urls)
    return "\n".join(values)


def _reasoning_text(message: Any, config: TracingConfig) -> str:
    values: list[str] = []
    if direct := getattr(message, "reasoning_content", None):
        values.append(str(direct))
    for block in getattr(message, "thinking_blocks", ()) or ():
        if getattr(block, "type", None) == "thinking" and hasattr(block, "thinking"):
            values.append(str(block.thinking))
    item = getattr(message, "responses_reasoning_item", None)
    if item is not None:
        values.extend(str(value) for value in (getattr(item, "summary", ()) or ()))
        values.extend(str(value) for value in (getattr(item, "content", ()) or ()))
    return config.transform("\n".join(values)) if values else ""
