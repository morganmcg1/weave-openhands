from __future__ import annotations

from typing import Any

import weave

import weave_openhands
from weave_openhands import TracingConfig


def test_init_configures_weave_before_openhands(monkeypatch) -> None:
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
    assert calls[0] == ("weave", "team/project")
    assert calls[1] == (
        "openhands",
        TracingConfig(agent_name="coding-agent", capture_content=False),
    )


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
