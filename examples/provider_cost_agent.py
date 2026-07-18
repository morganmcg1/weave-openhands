"""Run real OpenAI and Anthropic agents and trace tokens and costs to Weave."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from weave_openhands import finish, init

PROJECT = os.getenv("WEAVE_PROJECT", "wandb-applied-ai-team/test-openhands-weave")
AGENT_NAME = "openhands-provider-cost-demo"


@dataclass(frozen=True)
class ProviderCase:
    name: str
    model: str
    key_environment_variable: str


def main() -> None:
    client = init(PROJECT, agent_name=AGENT_NAME)

    # Import OpenHands after Weave owns the process-wide OpenTelemetry provider.
    from openhands.sdk import LLM, Agent, AgentContext, Conversation
    from pydantic import SecretStr
    from weave.trace.urls import agent_conversation_path

    cases = [
        ProviderCase(
            name="openai",
            model=os.getenv("OPENAI_MODEL", "openai/gpt-4.1-mini"),
            key_environment_variable="OPENAI_API_KEY",
        ),
        ProviderCase(
            name="anthropic",
            model=os.getenv("ANTHROPIC_MODEL", "anthropic/claude-haiku-4-5-20251001"),
            key_environment_variable="ANTHROPIC_API_KEY",
        ),
    ]

    failures: list[str] = []
    try:
        for case in cases:
            try:
                api_key = SecretStr(os.environ[case.key_environment_variable])
                llm = LLM(
                    model=case.model,
                    api_key=api_key,
                    usage_id=f"provider-cost-{case.name}",
                    temperature=0,
                    max_output_tokens=1_280,
                    extended_thinking_budget=1_024,
                    num_retries=1,
                )
                agent = Agent(
                    llm=llm,
                    tools=[],
                    agent_context=AgentContext(
                        system_message_suffix=(
                            "PROVIDER_COST_TEST: Call the finish tool immediately "
                            "with a short confirmation. Do not call any other tool."
                        )
                    ),
                )

                with TemporaryDirectory(prefix=f"openhands-{case.name}-") as directory:
                    conversation = Conversation(
                        agent=agent,
                        workspace=Path(directory),
                        max_iteration_per_run=3,
                        visualizer=None,
                    )
                    conversation.send_message(
                        f"PROVIDER_COST_MARKER_{case.name.upper()}: confirm this real "
                        "provider trace by calling finish now."
                    )
                    conversation.run()

                metrics = agent.llm.metrics
                usage = metrics.accumulated_token_usage
                if (
                    usage is None
                    or usage.prompt_tokens <= 0
                    or usage.completion_tokens <= 0
                ):
                    raise RuntimeError(
                        f"{case.name} did not report non-zero token usage"
                    )
                if metrics.accumulated_cost <= 0:
                    raise RuntimeError(f"{case.name} did not report a non-zero cost")

                conversation_id = str(conversation.id)
                print(f"provider={case.name}")
                print("status=success")
                print(f"model={case.model}")
                print(f"conversation_id={conversation_id}")
                print(f"input_tokens={usage.prompt_tokens}")
                print(f"output_tokens={usage.completion_tokens}")
                print(f"cost_usd={metrics.accumulated_cost:.9f}")
                print(
                    "weave_url="
                    + agent_conversation_path(
                        client.entity,
                        client.project,
                        conversation_id,
                    )
                )
            except Exception as error:
                failures.append(case.name)
                print(f"provider={case.name}")
                print("status=error")
                print(f"error_type={type(error).__name__}")
    finally:
        finish()

    if failures:
        raise RuntimeError("provider checks failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
