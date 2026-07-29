from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import Any

import weave

import weave_openhands
from weave_openhands import TracingConfig


def test_init_imports_openhands_before_weave_autopatching(monkeypatch) -> None:
    calls: list[tuple[str, Any]] = []
    client = object()

    def fake_weave_init(project_name: str) -> object:
        calls.append(("weave", project_name))
        return client

    def fake_instrument(config: TracingConfig) -> None:
        calls.append(("openhands", config))

    monkeypatch.setattr(weave, "init", fake_weave_init)
    monkeypatch.setattr(weave_openhands, "instrument", fake_instrument)

    result = weave_openhands.init(
        "team/project",
        agent_name="coding-agent",
        capture_content=False,
    )

    assert result is client
    assert calls[0] == (
        "openhands",
        TracingConfig(agent_name="coding-agent", capture_content=False),
    )
    assert calls[1] == ("weave", "team/project")


def test_init_avoids_generic_litellm_double_instrumentation() -> None:
    script = textwrap.dedent(
        """
        import litellm
        import weave
        import weave_openhands

        original = litellm.completion

        def fake_init(_project_name):
            weave.integrations.patch_litellm()
            return object()

        weave.init = fake_init
        weave_openhands.init("team/project")

        from openhands.sdk.llm import llm as openhands_llm

        assert litellm.completion is not original
        assert openhands_llm.litellm_completion is original
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1"},
    )
    assert result.returncode == 0, result.stderr


def test_finish_flushes_before_closing_weave(monkeypatch) -> None:
    calls: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        weave_openhands,
        "flush",
        lambda timeout: calls.append(("flush", timeout)),
    )
    monkeypatch.setattr(weave, "finish", lambda: calls.append(("finish", None)))

    weave_openhands.finish(2_500)

    assert calls == [("flush", 2_500), ("finish", None)]
