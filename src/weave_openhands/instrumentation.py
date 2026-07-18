from __future__ import annotations

import functools
import importlib.metadata
import json
import logging
import threading
import traceback
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, StatusCode

from weave_openhands.config import TracingConfig
from weave_openhands.serialization import (
    event_message,
    json_dumps,
    message_to_semconv,
    messages_to_semconv,
    system_instructions,
    to_jsonable,
    tool_definitions,
)

logger = logging.getLogger(__name__)

_TRACER_NAME = "weave_openhands"
_INTEGRATION_NAME = "weave-openhands"


@dataclass(slots=True)
class _Patch:
    target: Any
    name: str
    original: Any
    replacement: Any


@dataclass(slots=True)
class _RunTrace:
    conversation: Any
    span: Span
    conversation_id: str
    event_count_before: int


_config = TracingConfig()
_patches: list[_Patch] = []
_patch_lock = threading.RLock()
_active_runs: dict[int, _RunTrace] = {}
_current_run: ContextVar[_RunTrace | None] = ContextVar(
    "weave_openhands_current_run", default=None
)


def instrument(config: TracingConfig | None = None) -> None:
    """Instrument the installed OpenHands SDK.

    This function is idempotent. Calling it again updates the active configuration
    without stacking wrappers.
    """

    global _config
    with _patch_lock:
        if config is not None:
            _config = config
        if _patches:
            return

        from openhands.sdk.agent import agent as agent_module
        from openhands.sdk.agent import utils as agent_utils
        from openhands.sdk.agent.agent import Agent
        from openhands.sdk.conversation.impl.local_conversation import (
            LocalConversation,
        )

        _apply_patch(LocalConversation, "run", _wrap_run)
        _apply_patch(LocalConversation, "arun", _wrap_arun)
        _apply_patch(Agent, "_execute_action_event", _wrap_tool_execution)
        _apply_patch(agent_module, "make_llm_completion", _wrap_llm_call)
        _apply_patch(agent_module, "amake_llm_completion", _wrap_async_llm_call)
        _apply_patch(agent_utils, "make_llm_completion", _wrap_llm_call)
        _apply_patch(agent_utils, "amake_llm_completion", _wrap_async_llm_call)


def uninstrument() -> None:
    """Restore all SDK functions replaced by :func:`instrument`."""

    global _config
    with _patch_lock:
        for patch in reversed(_patches):
            if getattr(patch.target, patch.name) is patch.replacement:
                setattr(patch.target, patch.name, patch.original)
        _patches.clear()
        _active_runs.clear()
        _config = TracingConfig()


def is_instrumented() -> bool:
    return bool(_patches)


def flush(timeout_millis: int = 10_000) -> bool:
    """Flush the active OpenTelemetry provider."""

    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if force_flush is None:
        return True
    return bool(force_flush(timeout_millis=timeout_millis))


def _apply_patch(target: Any, name: str, factory: Any) -> None:
    original = getattr(target, name)
    replacement = factory(original)
    setattr(target, name, replacement)
    _patches.append(_Patch(target, name, original, replacement))


def _wrap_run(original: Any) -> Any:
    @functools.wraps(original)
    def wrapped(conversation: Any, *args: Any, **kwargs: Any) -> Any:
        return _trace_sync_run(
            conversation, lambda: original(conversation, *args, **kwargs)
        )

    return wrapped


def _wrap_arun(original: Any) -> Any:
    @functools.wraps(original)
    async def wrapped(conversation: Any, *args: Any, **kwargs: Any) -> Any:
        return await _trace_async_run(
            conversation, lambda: original(conversation, *args, **kwargs)
        )

    return wrapped


def _trace_sync_run(conversation: Any, call: Any) -> Any:
    run = _start_run(conversation)
    token = _current_run.set(run)
    otel_token = otel_context.attach(trace.set_span_in_context(run.span))
    try:
        return call()
    except BaseException as error:
        _record_error(run.span, error)
        raise
    finally:
        otel_context.detach(otel_token)
        _current_run.reset(token)
        _finish_run(run)


async def _trace_async_run(conversation: Any, call: Any) -> Any:
    run = _start_run(conversation)
    token = _current_run.set(run)
    otel_token = otel_context.attach(trace.set_span_in_context(run.span))
    try:
        return await call()
    except BaseException as error:
        _record_error(run.span, error)
        raise
    finally:
        otel_context.detach(otel_token)
        _current_run.reset(token)
        _finish_run(run)


def _start_run(conversation: Any) -> _RunTrace:
    conversation_id = str(conversation.id)
    model = str(getattr(conversation.agent.llm, "model", ""))
    span = trace.get_tracer(_TRACER_NAME).start_span(
        f"invoke_agent {_config.agent_name}", context=Context()
    )
    _set_attributes(
        span,
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": _config.agent_name,
            "gen_ai.conversation.id": conversation_id,
            "gen_ai.request.model": model,
            "weave.openhands.agent.class": type(conversation.agent).__name__,
            "weave.openhands.run.mode": "local",
            **_integration_attributes(),
        },
    )
    run = _RunTrace(
        conversation=conversation,
        span=span,
        conversation_id=conversation_id,
        event_count_before=len(conversation.state.events),
    )
    with _patch_lock:
        _active_runs[id(conversation)] = run
    return run


def _finish_run(run: _RunTrace) -> None:
    try:
        _enrich_run(run)
    except Exception as error:
        logger.exception("Failed to enrich OpenHands run span")
        _record_error(run.span, error, "weave-openhands enrichment failed")
    finally:
        with _patch_lock:
            _active_runs.pop(id(run.conversation), None)
        run.span.end()


def _enrich_run(run: _RunTrace) -> None:
    conversation = run.conversation
    events = list(conversation.state.events)
    new_events = events[run.event_count_before :]
    attrs: dict[str, Any] = {
        "weave.openhands.run.start_event_index": run.event_count_before,
        "weave.openhands.run.end_event_index": len(events),
        "weave.openhands.run.status": str(conversation.state.execution_status),
    }

    input_message = _latest_user_message(events[: run.event_count_before])
    if _config.capture_content and input_message is not None:
        attrs["gen_ai.input.messages"] = json_dumps(
            [message_to_semconv(input_message, _config)], _config
        )

    output_messages = _output_messages(new_events)
    if _config.capture_content and output_messages:
        attrs["gen_ai.output.messages"] = json_dumps(
            [message_to_semconv(message, _config) for message in output_messages],
            _config,
        )

    system_event = next(
        (event for event in events if type(event).__name__ == "SystemPromptEvent"),
        None,
    )
    if system_event is not None:
        instructions = _system_event_instructions(system_event)
        if _config.capture_content and instructions:
            attrs["gen_ai.system_instructions"] = json_dumps(instructions, _config)
        tools = getattr(system_event, "tools", ()) or ()
        attrs["weave.openhands.tools.count"] = len(tools)
        if _config.capture_content:
            attrs["gen_ai.tool.definitions"] = json_dumps(
                tool_definitions(tools, _config), _config
            )

    attrs.update(_skill_attributes(conversation))
    attrs["weave.openhands.context.event_count"] = len(events)
    if _config.capture_content:
        attrs["weave.openhands.context.events"] = json_dumps(
            [_event_snapshot(event) for event in events], _config
        )
        attrs["weave.openhands.run.events"] = json_dumps(
            [_event_snapshot(event) for event in new_events], _config
        )
        attrs["weave.openhands.agent.config"] = json_dumps(
            _safe_agent_config(conversation.agent), _config
        )
    _set_attributes(run.span, attrs)


def _wrap_llm_call(original: Any) -> Any:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        llm, messages, tools, call_context = _llm_arguments(args, kwargs)
        span = _start_chat_span(llm, messages, tools, call_context)
        try:
            response = original(*args, **kwargs)
            _finish_chat_span(span, response)
            return response
        except BaseException as error:
            _record_error(span, error)
            span.end()
            raise

    return wrapped


def _wrap_async_llm_call(original: Any) -> Any:
    @functools.wraps(original)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        llm, messages, tools, call_context = _llm_arguments(args, kwargs)
        span = _start_chat_span(llm, messages, tools, call_context)
        try:
            response = await original(*args, **kwargs)
            _finish_chat_span(span, response)
            return response
        except BaseException as error:
            _record_error(span, error)
            span.end()
            raise

    return wrapped


def _llm_arguments(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, list[Any], list[Any], Any]:
    llm = kwargs.get("llm", args[0] if args else None)
    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
    tools = kwargs.get("tools", args[2] if len(args) > 2 else None) or []
    call_context = kwargs.get(
        "call_context",
        args[4] if len(args) > 4 else getattr(llm, "_call_context", None),
    )
    return llm, list(messages or []), list(tools), call_context


def _start_chat_span(
    llm: Any, messages: list[Any], tools: list[Any], call_context: Any
) -> Span:
    model = str(getattr(llm, "model", ""))
    run = _current_run.get()
    parent_context = trace.set_span_in_context(run.span) if run is not None else None
    span = trace.get_tracer(_TRACER_NAME).start_span(
        f"chat {model}".rstrip(), context=parent_context
    )
    conversation_id = (
        run.conversation_id
        if run is not None
        else str(getattr(call_context, "session_id", "") or "")
    )
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        "weave.openhands.llm.usage_id": str(getattr(llm, "usage_id", "")),
        "weave.openhands.tools.count": len(tools),
        **_integration_attributes(),
    }
    if conversation_id:
        attrs["gen_ai.conversation.id"] = conversation_id
    _copy_request_settings(attrs, llm)
    if _config.capture_content:
        input_messages = messages_to_semconv(messages, _config)
        instructions = system_instructions(messages, _config)
        if input_messages:
            attrs["gen_ai.input.messages"] = json_dumps(input_messages, _config)
        if instructions:
            attrs["gen_ai.system_instructions"] = json_dumps(instructions, _config)
        attrs["gen_ai.tool.definitions"] = json_dumps(
            tool_definitions(tools, _config), _config
        )
    _set_attributes(span, attrs)
    return span


def _finish_chat_span(span: Span, response: Any) -> None:
    attrs: dict[str, Any] = {}
    message = getattr(response, "message", None)
    if _config.capture_content and message is not None:
        attrs["gen_ai.output.messages"] = json_dumps(
            [message_to_semconv(message, _config)], _config
        )
    raw_response = getattr(response, "raw_response", None)
    if raw_response is not None:
        response_id = getattr(raw_response, "id", None)
        response_model = getattr(raw_response, "model", None)
        if response_id:
            attrs["gen_ai.response.id"] = str(response_id)
        if response_model:
            attrs["gen_ai.response.model"] = str(response_model)
        attrs.update(_usage_attributes(raw_response, response))
        finish_reasons = _finish_reasons(raw_response)
        if finish_reasons:
            attrs["gen_ai.response.finish_reasons"] = finish_reasons
        if _config.capture_content:
            attrs["weave.openhands.llm.raw_response"] = json_dumps(
                raw_response, _config
            )
    _set_attributes(span, attrs)
    span.end()


def _wrap_tool_execution(original: Any) -> Any:
    @functools.wraps(original)
    def wrapped(
        agent: Any, conversation: Any, action_event: Any, *args: Any, **kwargs: Any
    ) -> Any:
        run = _run_for_conversation(conversation)
        parent_context = (
            trace.set_span_in_context(run.span) if run is not None else None
        )
        tool_name = str(getattr(action_event, "tool_name", "unknown"))
        span = trace.get_tracer(_TRACER_NAME).start_span(
            f"execute_tool {tool_name}", context=parent_context
        )
        conversation_id = str(conversation.id)
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.conversation.id": conversation_id,
            "gen_ai.tool.call.id": str(getattr(action_event, "tool_call_id", "")),
            "weave.openhands.tool.security_risk": str(
                getattr(action_event, "security_risk", "")
            ),
            **_integration_attributes(),
        }
        if _config.capture_content:
            tool_call = getattr(action_event, "tool_call", None)
            attrs["weave.openhands.tool.summary"] = _config.transform(
                str(getattr(action_event, "summary", "") or "")
            )
            attrs["gen_ai.tool.call.arguments"] = _config.transform(
                str(getattr(tool_call, "arguments", ""))
            )
            attrs["weave.openhands.tool.action"] = json_dumps(action_event, _config)
        _set_attributes(span, attrs)
        try:
            result = original(agent, conversation, action_event, *args, **kwargs)
            if _config.capture_content:
                _set_attributes(
                    span,
                    {
                        "gen_ai.tool.call.result": json_dumps(result, _config),
                        "weave.openhands.tool.result_events": json_dumps(
                            result, _config
                        ),
                    },
                )
            if any(type(event).__name__ == "AgentErrorEvent" for event in result):
                span.set_status(StatusCode.ERROR, "OpenHands tool execution failed")
            return result
        except BaseException as error:
            _record_error(span, error)
            raise
        finally:
            span.end()

    return wrapped


def _run_for_conversation(conversation: Any) -> _RunTrace | None:
    current = _current_run.get()
    if current is not None and current.conversation is conversation:
        return current
    with _patch_lock:
        return _active_runs.get(id(conversation))


def _latest_user_message(events: list[Any]) -> Any | None:
    for event in reversed(events):
        if type(event).__name__ != "MessageEvent":
            continue
        if getattr(event, "source", None) == "user":
            return event_message(event)
    return None


def _output_messages(events: list[Any]) -> list[Any]:
    output: list[Any] = []
    for event in events:
        if (
            type(event).__name__ == "MessageEvent"
            and getattr(event, "source", None) == "agent"
        ):
            if message := event_message(event):
                output.append(message)
        elif type(event).__name__ == "ActionEvent":
            action = getattr(event, "action", None)
            if getattr(event, "tool_name", None) == "finish" and hasattr(
                action, "message"
            ):
                from openhands.sdk.llm import Message, TextContent

                output.append(
                    Message(
                        role="assistant",
                        content=[TextContent(text=str(getattr(action, "message", "")))],
                    )
                )
            elif message := event_message(event):
                output.append(message)
    return output


def _system_event_instructions(event: Any) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for content in (
        getattr(event, "system_prompt", None),
        getattr(event, "dynamic_context", None),
    ):
        if content is not None and hasattr(content, "text"):
            values.append(
                {"type": "text", "content": _config.transform(str(content.text))}
            )
    return values


def _skill_attributes(conversation: Any) -> dict[str, Any]:
    state = conversation.state
    context = getattr(conversation.agent, "agent_context", None)
    skills = list(getattr(context, "skills", ()) or ())
    attrs: dict[str, Any] = {
        "weave.openhands.skills.available.count": len(skills),
        "weave.openhands.skills.activated": list(
            getattr(state, "activated_knowledge_skills", ()) or ()
        ),
        "weave.openhands.skills.invoked": list(
            getattr(state, "invoked_skills", ()) or ()
        ),
    }
    if _config.capture_content:
        attrs["weave.openhands.skills.available"] = json_dumps(
            [_skill_snapshot(skill) for skill in skills], _config
        )
    else:
        attrs["weave.openhands.skills.available.names"] = [
            str(getattr(skill, "name", "")) for skill in skills
        ]
    return attrs


def _safe_agent_config(agent: Any) -> dict[str, Any]:
    llm = agent.llm
    return {
        "agent_class": type(agent).__name__,
        "llm": _safe_llm_config(llm),
        "system_prompt_filename": getattr(agent, "system_prompt_filename", None),
        "system_prompt_kwargs": getattr(agent, "system_prompt_kwargs", None),
        "include_default_tools": getattr(agent, "include_default_tools", None),
        "filter_tools_regex": getattr(agent, "filter_tools_regex", None),
        "mcp_servers": sorted((getattr(agent, "mcp_config", {}) or {}).keys()),
    }


def _safe_llm_config(llm: Any) -> dict[str, Any]:
    fields = (
        "model",
        "model_canonical_name",
        "usage_id",
        "api_version",
        "subscription_vendor",
        "aws_region_name",
        "num_retries",
        "retry_multiplier",
        "retry_min_wait",
        "retry_max_wait",
        "timeout",
        "max_message_chars",
        "temperature",
        "top_p",
        "top_k",
        "max_input_tokens",
        "max_output_tokens",
        "stream",
        "drop_params",
        "modify_params",
        "disable_vision",
        "disable_stop_word",
        "caching_prompt",
        "native_tool_calling",
        "force_string_serializer",
        "inline_image_urls",
        "reasoning_effort",
        "reasoning_summary",
        "enable_encrypted_reasoning",
        "prompt_cache_retention",
        "extended_thinking_budget",
        "seed",
        "fallback_strategy",
    )
    return {
        field: value
        for field in fields
        if (value := getattr(llm, field, None)) is not None
    }


def _skill_snapshot(skill: Any) -> dict[str, Any]:
    fields = (
        "name",
        "content",
        "trigger",
        "source",
        "inputs",
        "is_agentskills_format",
        "version",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed_tools",
        "disable_model_invocation",
        "resources",
    )
    return {
        field: value
        for field in fields
        if (value := getattr(skill, field, None)) is not None
    }


def _event_snapshot(event: Any) -> Any:
    if type(event).__name__ != "SystemPromptEvent":
        return event
    snapshot = event.model_dump(exclude={"tools"}, exclude_none=True)
    snapshot["tools"] = tool_definitions(getattr(event, "tools", ()) or (), _config)
    return snapshot


def _copy_request_settings(attrs: dict[str, Any], llm: Any) -> None:
    fields = {
        "temperature": "gen_ai.request.temperature",
        "top_p": "gen_ai.request.top_p",
        "top_k": "gen_ai.request.top_k",
        "max_output_tokens": "gen_ai.request.max_tokens",
        "seed": "gen_ai.request.seed",
    }
    for field, attribute in fields.items():
        value = getattr(llm, field, None)
        if value is not None:
            attrs[attribute] = value


def _usage_attributes(raw_response: Any, response: Any) -> dict[str, int]:
    usage = getattr(raw_response, "usage", None)
    if usage is None:
        usage = getattr(
            getattr(response, "metrics", None), "accumulated_token_usage", None
        )
    if usage is None:
        return {}

    def first(*names: str) -> int:
        for name in names:
            value = _value(usage, name)
            if value is not None:
                return int(value)
        return 0

    attrs: dict[str, int] = {}
    input_tokens = first("input_tokens", "prompt_tokens")
    output_tokens = first("output_tokens", "completion_tokens")
    if input_tokens:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens:
        attrs["gen_ai.usage.output_tokens"] = output_tokens

    input_details = _value(usage, "input_tokens_details") or _value(
        usage, "prompt_tokens_details"
    )
    output_details = _value(usage, "output_tokens_details") or _value(
        usage, "completion_tokens_details"
    )
    cache_read = _value(input_details, "cached_tokens", 0) or _value(
        usage, "cache_read_tokens", 0
    )
    reasoning = _value(output_details, "reasoning_tokens", 0) or _value(
        usage, "reasoning_tokens", 0
    )
    cache_write = _value(usage, "cache_write_tokens", 0)
    if cache_read:
        attrs["gen_ai.usage.cache_read.input_tokens"] = int(cache_read)
    if cache_write:
        attrs["gen_ai.usage.cache_creation.input_tokens"] = int(cache_write)
    if reasoning:
        attrs["gen_ai.usage.reasoning_tokens"] = int(reasoning)
    return attrs


def _finish_reasons(raw_response: Any) -> list[str]:
    reasons: list[str] = []
    for choice in getattr(raw_response, "choices", ()) or ():
        value = _value(choice, "finish_reason")
        if value:
            reasons.append(str(value))
    return reasons


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _integration_attributes() -> dict[str, Any]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "integration.name": _INTEGRATION_NAME,
        "integration.version": version(_INTEGRATION_NAME),
        "integration.meta.package_name": "openhands-sdk",
        "integration.meta.package_version": version("openhands-sdk"),
    }


def _set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None or value == "":
            continue
        scalar = isinstance(value, (str, bool, int, float))
        scalar_sequence = isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, bool, int, float)) for item in value
        )
        if scalar or scalar_sequence:
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, json.dumps(to_jsonable(value, _config)))


def _record_error(span: Span, error: BaseException, prefix: str = "") -> None:
    error_type = f"{type(error).__module__}.{type(error).__qualname__}"
    attributes: dict[str, Any] = {"exception.type": error_type}
    if _config.capture_content:
        message = _config.transform(str(error))
        description = f"{prefix}: {message}" if prefix else message
        attributes["exception.message"] = message
        attributes["exception.stacktrace"] = _config.transform(
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
        )
    else:
        description = prefix or error_type
    span.add_event("exception", attributes=attributes)
    span.set_status(StatusCode.ERROR, description)
