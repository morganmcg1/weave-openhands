"""Run a feature-rich OpenHands agent and export its trace to W&B Weave."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Self

from weave_openhands import finish, init

PROJECT = os.getenv("WEAVE_PROJECT", "wandb-applied-ai-team/test-openhands-weave")
AGENT_NAME = "openhands-rich-demo"


def main() -> None:
    client = init(PROJECT, agent_name=AGENT_NAME)

    # Import OpenHands after Weave owns the process-wide OpenTelemetry provider.
    from openhands.sdk import (
        Agent,
        AgentContext,
        Conversation,
        Tool,
        register_tool,
    )
    from openhands.sdk.llm import Message, MessageToolCall, TextContent
    from openhands.sdk.skills import KeywordTrigger, Skill
    from openhands.sdk.testing import TestLLM
    from openhands.sdk.tool import (
        Action,
        DeclaredResources,
        Observation,
        ToolAnnotations,
        ToolDefinition,
        ToolExecutor,
    )
    from weave.trace.urls import agent_conversation_path

    class WorkspaceReportAction(Action):
        title: str
        findings: list[str]

    class WorkspaceReportObservation(Observation):
        path: str
        bytes_written: int
        source_excerpt: str

    class WorkspaceReportExecutor(
        ToolExecutor[WorkspaceReportAction, WorkspaceReportObservation]
    ):
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace

        def __call__(
            self,
            action: WorkspaceReportAction,
            conversation: Any = None,
        ) -> WorkspaceReportObservation:
            brief = (self.workspace / "brief.md").read_text()
            report = "\n".join(
                [
                    f"# {action.title}",
                    "",
                    "## Source brief",
                    brief.strip(),
                    "",
                    "## Findings",
                    *(f"- {finding}" for finding in action.findings),
                    "",
                ]
            )
            output = self.workspace / "agent_report.md"
            output.write_text(report)
            return WorkspaceReportObservation.from_text(
                text=f"Created {output.name} with {len(action.findings)} findings.",
                path=output.name,
                bytes_written=len(report.encode()),
                source_excerpt=brief[:120],
            )

    class WorkspaceReportTool(
        ToolDefinition[WorkspaceReportAction, WorkspaceReportObservation]
    ):
        def declared_resources(self, action: Action) -> DeclaredResources:
            return DeclaredResources(
                keys=("file:agent_report.md",),
                declared=True,
            )

        @classmethod
        def create(
            cls,
            conv_state: Any = None,
            **params: Any,
        ) -> Sequence[Self]:
            if params:
                raise ValueError("WorkspaceReportTool does not accept parameters")
            if conv_state is None:
                raise ValueError("WorkspaceReportTool requires conversation state")
            workspace = Path(conv_state.workspace.working_dir)
            return [
                cls(
                    description=(
                        "Read brief.md from the workspace and create agent_report.md "
                        "with structured findings."
                    ),
                    action_type=WorkspaceReportAction,
                    observation_type=WorkspaceReportObservation,
                    executor=WorkspaceReportExecutor(workspace),
                    annotations=ToolAnnotations(
                        title="workspace_report",
                        readOnlyHint=False,
                        destructiveHint=False,
                        idempotentHint=True,
                        openWorldHint=False,
                    ),
                )
            ]

    def tool_call(
        name: str,
        arguments: dict[str, Any],
        *,
        call_id: str,
        reasoning: str,
    ) -> Message:
        message = Message(
            role="assistant",
            content=[TextContent(text=f"Calling {name}.")],
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

    register_tool("WorkspaceReportTool", WorkspaceReportTool)

    responses = [
        tool_call(
            "think",
            {
                "thought": (
                    "Plan: inspect the exposed skills, invoke the tracing skill, "
                    "write a workspace report, then finish."
                )
            },
            call_id="call-think",
            reasoning="I should make the plan explicit before changing the workspace.",
        ),
        tool_call(
            "invoke_skill",
            {"name": "trace-review"},
            call_id="call-skill",
            reasoning="The progressive skill contains the report quality checklist.",
        ),
        tool_call(
            "workspace_report",
            {
                "title": "OpenHands Weave Observability Report",
                "findings": [
                    "The full system and dynamic context is available on agent spans.",
                    "LLM and tool calls share one conversation identifier.",
                    "Skill activation and invocation are recorded independently.",
                ],
            },
            call_id="call-report",
            reasoning="I now have the plan, skill guidance, and workspace brief.",
        ),
        tool_call(
            "finish",
            {
                "message": (
                    "Created agent_report.md after planning, activating context, "
                    "and invoking the trace-review skill."
                )
            },
            call_id="call-finish",
            reasoning="The requested workspace artifact is complete.",
        ),
    ]

    try:
        with TemporaryDirectory(prefix="openhands-weave-demo-") as directory:
            workspace = Path(directory)
            (workspace / "brief.md").write_text(
                "WORKSPACE_BRIEF_MARKER: Demonstrate complete agent observability.\n"
            )
            skill_path = workspace / ".agents/skills/trace-review/SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(
                "PROGRESSIVE_SKILL_INSTRUCTION: Include evidence from the workspace.\n"
            )

            context = AgentContext(
                system_message_suffix=(
                    "DEMO_SYSTEM_INSTRUCTION: expose planning, tools, and skill state."
                ),
                skills=[
                    Skill(
                        name="trace-review",
                        description="Checklist for a trace-aware report.",
                        content=skill_path.read_text(),
                        source=str(skill_path),
                        is_agentskills_format=True,
                    ),
                    Skill(
                        name="observability-policy",
                        description="Rules activated for observability tasks.",
                        content=(
                            "TRIGGERED_SKILL_INSTRUCTION: report tool evidence and "
                            "conversation identity."
                        ),
                        trigger=KeywordTrigger(keywords=["observability"]),
                    ),
                ],
            )
            agent = Agent(
                llm=TestLLM.from_messages(
                    responses,
                    model="test/openhands-rich-demo",
                    usage_id="rich-demo",
                ),
                tools=[Tool(name="WorkspaceReportTool")],
                agent_context=context,
            )

            event_types: list[str] = []

            def on_event(event: Any) -> None:
                event_types.append(type(event).__name__)

            conversation = Conversation(
                agent=agent,
                workspace=workspace,
                persistence_dir=workspace / ".conversation-state",
                callbacks=[on_event],
                visualizer=None,
            )
            conversation.send_message(
                "Create an observability report from brief.md. Think first, invoke "
                "the trace-review skill, use the workspace report tool, and finish."
            )
            conversation.run()

            report = (workspace / "agent_report.md").read_text()
            conversation_id = str(conversation.id)
            print(f"conversation_id={conversation_id}")
            print(
                "activated_skills="
                + ",".join(conversation.state.activated_knowledge_skills)
            )
            print("invoked_skills=" + ",".join(conversation.state.invoked_skills))
            print(f"event_count={len(event_types)}")
            print(f"report_bytes={len(report.encode())}")
            print(
                "weave_url="
                + agent_conversation_path(
                    client.entity,
                    client.project,
                    conversation_id,
                )
            )
    finally:
        finish()


if __name__ == "__main__":
    main()
