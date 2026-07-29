"""Run five deterministic OpenHands logging patterns in W&B Weave."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Self

from weave_openhands import finish, init

PROJECT = os.getenv("WEAVE_PROJECT", "wandb-applied-ai-team/test-openhands-weave")
SCENARIOS = {
    "basic": "One agent turn with system, user, model, and finish-tool data.",
    "multi-turn": "Two traces grouped into one conversation.",
    "tools-and-skills": "Tool schema, arguments, result, reasoning, and skill state.",
    "error-recovery": "A failed tool span followed by a successful recovery.",
    "privacy-async": "Async execution with content redaction or metadata-only capture.",
}


@dataclass(frozen=True)
class ScenarioResult:
    conversation_id: str
    turns: int
    llm_calls: int
    status: str
    activated_skills: tuple[str, ...] = ()
    invoked_skills: tuple[str, ...] = ()


def redact(value: str) -> str:
    return value.replace("SECRET", "[REDACTED]")


def _tool_message(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str,
    text: str,
    reasoning: str = "",
):
    from openhands.sdk.llm import Message, MessageToolCall, TextContent

    message = Message(
        role="assistant",
        content=[TextContent(text=text)],
        tool_calls=[
            MessageToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(arguments, sort_keys=True),
                origin="completion",
            )
        ],
    )
    message.reasoning_content = reasoning
    return message


def _finish_message(message: str, call_id: str):
    return _tool_message(
        "finish",
        {"message": message},
        call_id=call_id,
        text="Finishing the task.",
    )


@cache
def _register_showcase_tool() -> None:
    from openhands.sdk import register_tool
    from openhands.sdk.tool import Action, Observation, ToolDefinition, ToolExecutor

    class ShowcaseAction(Action):
        operation: Literal["write", "read", "fail"]
        value: str = ""

    class ShowcaseObservation(Observation):
        operation: str
        path: str
        value: str

    class ShowcaseExecutor(ToolExecutor[ShowcaseAction, ShowcaseObservation]):
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace

        def __call__(
            self,
            action: ShowcaseAction,
            conversation: Any = None,
        ) -> ShowcaseObservation:
            path = self.workspace / "showcase.txt"
            if action.operation == "fail":
                raise ValueError("DEMO_EXPECTED_FAILURE")
            if action.operation == "write":
                path.write_text(action.value)
            value = path.read_text()
            return ShowcaseObservation.from_text(
                text=f"{action.operation} completed for {path.name}",
                operation=action.operation,
                path=path.name,
                value=value,
            )

    class ShowcaseTool(ToolDefinition[ShowcaseAction, ShowcaseObservation]):
        @classmethod
        def create(
            cls,
            conv_state: Any = None,
            **params: Any,
        ) -> Sequence[Self]:
            if params:
                raise ValueError("ShowcaseTool does not accept parameters")
            if conv_state is None:
                raise ValueError("ShowcaseTool requires conversation state")
            return [
                cls(
                    description="Write, read, or intentionally fail showcase.txt.",
                    action_type=ShowcaseAction,
                    observation_type=ShowcaseObservation,
                    executor=ShowcaseExecutor(Path(conv_state.workspace.working_dir)),
                )
            ]

    register_tool("WeaveLoggingShowcaseTool", ShowcaseTool)


def _make_conversation(
    workspace: Path,
    scenario: str,
    responses: list[Any],
    *,
    tools: list[Any] | None = None,
    agent_context: Any = None,
):
    from openhands.sdk import Agent, AgentContext, Conversation
    from openhands.sdk.testing import TestLLM

    agent = Agent(
        llm=TestLLM.from_messages(
            responses,
            model=f"test/weave-{scenario}",
            usage_id=f"weave-{scenario}",
        ),
        tools=tools or [],
        agent_context=agent_context
        or AgentContext(system_message_suffix=f"WEAVE_SHOWCASE_SCENARIO: {scenario}"),
    )
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        visualizer=None,
    )
    return agent, conversation


def run_scenario(scenario: str, workspace: Path) -> ScenarioResult:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    from openhands.sdk import AgentContext, Tool
    from openhands.sdk.skills import KeywordTrigger, Skill

    turns = 1
    if scenario == "basic":
        responses = [_finish_message("Basic trace complete.", "basic-finish")]
        agent, conversation = _make_conversation(workspace, scenario, responses)
        conversation.send_message("Record a basic OpenHands agent trace.")
        conversation.run()
    elif scenario == "multi-turn":
        responses = [
            _finish_message("First turn complete.", "multi-finish-1"),
            _finish_message("Follow-up complete.", "multi-finish-2"),
        ]
        agent, conversation = _make_conversation(workspace, scenario, responses)
        conversation.send_message("Create the first turn.")
        conversation.run()
        conversation.send_message("Continue the same conversation.")
        conversation.run()
        turns = 2
    elif scenario == "tools-and-skills":
        _register_showcase_tool()
        responses = [
            _tool_message(
                "invoke_skill",
                {"name": "demo-guidance"},
                call_id="skill-invoke",
                text="Loading the task-specific guidance.",
                reasoning="The skill contains the required output convention.",
            ),
            _tool_message(
                "showcase",
                {"operation": "write", "value": "tool result captured"},
                call_id="showcase-write",
                text="Writing the requested showcase artifact.",
                reasoning=(
                    "A custom tool demonstrates schema, argument, and result logging."
                ),
            ),
            _finish_message("Tool and skill trace complete.", "tools-finish"),
        ]
        skill_path = workspace / ".agents/skills/demo-guidance/SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("Use the showcase tool and report its result.\n")
        context = AgentContext(
            system_message_suffix="Demonstrate tool and skill observability.",
            skills=[
                Skill(
                    name="demo-guidance",
                    description="Task-specific logging guidance.",
                    content=skill_path.read_text(),
                    source=str(skill_path),
                    is_agentskills_format=True,
                ),
                Skill(
                    name="observability-policy",
                    description="Rules for observability requests.",
                    content="Preserve tool evidence in the trace.",
                    trigger=KeywordTrigger(keywords=["observability"]),
                ),
            ],
        )
        agent, conversation = _make_conversation(
            workspace,
            scenario,
            responses,
            tools=[Tool(name="WeaveLoggingShowcaseTool")],
            agent_context=context,
        )
        conversation.send_message(
            "For this observability example, invoke demo-guidance, write the "
            "showcase artifact, and finish."
        )
        conversation.run()
    elif scenario == "error-recovery":
        _register_showcase_tool()
        responses = [
            _tool_message(
                "showcase",
                {"operation": "fail"},
                call_id="expected-failure",
                text="Running the intentionally failing validation.",
                reasoning="The first attempt should create an error tool span.",
            ),
            _tool_message(
                "showcase",
                {"operation": "write", "value": "recovered"},
                call_id="recovery-write",
                text="Recovering with a valid write.",
                reasoning="The failure is recoverable without ending the agent turn.",
            ),
            _finish_message("Recovered from the expected failure.", "recovery-finish"),
        ]
        agent, conversation = _make_conversation(
            workspace,
            scenario,
            responses,
            tools=[Tool(name="WeaveLoggingShowcaseTool")],
        )
        conversation.send_message("Demonstrate a failed tool call and recovery.")
        conversation.run()
    else:
        responses = [
            _finish_message(
                "SECRET result is redacted in Weave.",
                "privacy-finish",
            )
        ]
        agent, conversation = _make_conversation(workspace, scenario, responses)
        conversation.send_message("Handle this SECRET prompt asynchronously.")
        asyncio.run(conversation.arun())

    status = conversation.state.execution_status
    return ScenarioResult(
        conversation_id=str(conversation.id),
        turns=turns,
        llm_calls=agent.llm.call_count,
        status=str(getattr(status, "value", status)),
        activated_skills=tuple(conversation.state.activated_knowledge_skills),
        invoked_skills=tuple(conversation.state.invoked_skills),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Disable prompt, message, event, argument, and result capture.",
    )
    args = parser.parse_args()
    if args.metadata_only and args.scenario != "privacy-async":
        parser.error("--metadata-only is only available for privacy-async")
    return args


def main() -> None:
    args = parse_args()
    client = init(
        PROJECT,
        agent_name=f"openhands-{args.scenario}",
        capture_content=not args.metadata_only,
        content_transform=redact if args.scenario == "privacy-async" else None,
    )

    from weave.trace.urls import agent_conversation_path

    try:
        with TemporaryDirectory(prefix=f"openhands-{args.scenario}-") as directory:
            result = run_scenario(args.scenario, Path(directory))
            print(f"scenario={args.scenario}")
            print(f"conversation_id={result.conversation_id}")
            print(f"turns={result.turns}")
            print(f"llm_calls={result.llm_calls}")
            print(f"status={result.status}")
            if result.activated_skills:
                print("activated_skills=" + ",".join(result.activated_skills))
            if result.invoked_skills:
                print("invoked_skills=" + ",".join(result.invoked_skills))
            print(
                "weave_url="
                + agent_conversation_path(
                    client.entity,
                    client.project,
                    result.conversation_id,
                )
            )
    finally:
        finish()


if __name__ == "__main__":
    main()
