# weave-openhands

W&B Weave tracing for the [OpenHands Software Agent SDK](https://docs.openhands.dev/sdk/).
It records OpenHands conversations using the OpenTelemetry GenAI span conventions
understood by Weave's Agents view:

```text
invoke_agent openhands
├── chat <model>
├── execute_tool <tool>
├── chat <model>
└── execute_tool finish
```

Each in-process `LocalConversation.run()` or `LocalConversation.arun()` becomes a
root agent invocation. LLM requests and tool executions are child spans, and every
span carries the same `gen_ai.conversation.id` so multiple turns remain grouped as
one conversation.

## What is captured

Content capture is on by default. The integration records:

- the exact system prompt and dynamic context exposed to the model;
- every input and output message, including visible reasoning and tool calls;
- all resolved tool schemas, tool arguments, results, summaries, and security risk;
- available, activated, and invoked OpenHands skills;
- the complete OpenHands event stream for the conversation and the events added by
  each run;
- model request settings, response metadata, finish reasons, and token usage;
- errors on the root, LLM, and tool spans.

The root, chat, and tool spans use `gen_ai.operation.name` values
`invoke_agent`, `chat`, and `execute_tool`, matching Weave's other agent
integrations.

## Install

This repository is not yet published to PyPI. Install it directly from GitHub:

```bash
pip install "weave-openhands @ git+https://github.com/morganmcg1/weave-openhands.git"
```

Python 3.12 and 3.13 are supported.

## Use

Initialize the integration before importing or constructing OpenHands agents:

```python
from weave_openhands import init

init("my-team/my-weave-project", agent_name="openhands")

from openhands.sdk import Agent, Conversation
from openhands.sdk.llm import LLM

agent = Agent(llm=LLM(model="your-provider/your-model", usage_id="coding-agent"))
conversation = Conversation(agent=agent)
conversation.send_message("Fix the failing tests")
conversation.run()
```

`init()` initializes Weave first, then patches the OpenHands SDK boundaries. This
ordering matters because OpenTelemetry has one process-wide tracer provider.
Call `weave_openhands.finish()` during explicit shutdown when you need to flush
queued spans immediately.

### End-to-end example

[`examples/rich_agent.py`](examples/rich_agent.py) runs a deterministic, ten-turn
OpenHands agent with visible reasoning, repeated user requests in one conversation,
built-in thinking and finishing, progressive-disclosure and keyword-triggered
skills, a resource-declaring custom workspace tool, persisted events, callbacks,
and full Weave tracing. It also records an expected tool failure and recovery, then
dispatches two final tool calls against the same declared file resource:

```bash
uv run python examples/rich_agent.py
```

It defaults to `wandb-applied-ai-team/test-openhands-weave`; set `WEAVE_PROJECT` to
send it elsewhere. The example uses OpenHands' `TestLLM`, so it exercises the real
agent/tool/event loop deterministically without a model-provider API key or model
spend.

[`examples/provider_cost_agent.py`](examples/provider_cost_agent.py) runs one real
OpenAI agent and one real Anthropic agent, asserts that both report non-zero token
usage and cost, and sends both traces to the same Weave project:

```bash
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
  uv run python examples/provider_cost_agent.py
```

The integration emits standard `gen_ai.provider.name`, `gen_ai.request.model`,
`gen_ai.response.model`, and `gen_ai.usage.*` attributes. Weave uses the model and
token attributes to calculate per-span and conversation-level USD costs at query
time from its pricing table.

### Redact content

Full context is useful for debugging but can contain source code, personal data,
credentials returned by tools, or other sensitive material. Transform each captured
payload string before export with a redactor:

```python
from weave_openhands import init


def redact(value: str) -> str:
    return value.replace("super-secret", "[REDACTED]")


init("my-team/my-project", content_transform=redact)
```

To retain span structure and metadata without prompts, messages, event payloads,
tool arguments, or results:

```python
init("my-team/my-project", capture_content=False)
```

The integration never intentionally serializes the LLM API key or OpenHands secret
store. It also omits OpenHands' opaque `encrypted_content` and `signature` reasoning
fields. This is not a general secret scanner: values echoed into prompts or tool
results are captured unless your `content_transform` removes them.

## Attribute contract

| Span | Standard attributes | OpenHands detail |
| --- | --- | --- |
| Agent run | `gen_ai.operation.name`, `gen_ai.agent.name`, `gen_ai.conversation.id`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions` | all events, per-run events, skill state, safe agent configuration |
| LLM call | `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions`, `gen_ai.usage.*` | exact raw provider response |
| Tool call | `gen_ai.operation.name`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` | action event, result events, summary, security risk |

All integration spans also include `integration.name`, `integration.version`,
`integration.meta.package_name`, and `integration.meta.package_version`.

## Scope and limitations

- Deep instrumentation currently targets the in-process `LocalConversation` and
  standard OpenHands `Agent`. Remote/ACP conversations only expose their internal
  LLM and tool activity if this package is initialized in the server process where
  that work actually runs.
- Initialize either this package or OpenHands' generic OTLP observability exporter
  as the owner of the process-wide trace provider. Enabling both can duplicate spans
  or route spans through the wrong exporter.
- Spans are created at runtime boundaries rather than reconstructed from logs. This
  provides exact model/tool inputs but intentionally couples the integration to the
  supported OpenHands major version. The initial compatibility suite targets
  OpenHands 1.36.1, and the dependency is capped below 2.0.

## Development

```bash
uv sync --dev --python 3.12
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv run python -m build
```

The test suite runs real synchronous and asynchronous OpenHands conversations with
`TestLLM` and an in-memory OpenTelemetry exporter. No model or W&B credentials are
needed.

## License

Apache-2.0.
