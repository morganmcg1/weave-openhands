from __future__ import annotations

from types import SimpleNamespace

from weave_openhands.instrumentation import _safe_llm_config, _skill_snapshot


def test_safe_llm_config_excludes_credentials_and_request_bodies() -> None:
    llm = SimpleNamespace(
        model="provider/model",
        usage_id="primary",
        temperature=0.2,
        api_key="secret-api-key",
        aws_access_key_id="secret-access-key",
        aws_secret_access_key="secret-access-key-value",
        extra_headers={"Authorization": "Bearer secret"},
        litellm_extra_body={"password": "secret"},
        base_url="https://user:secret@example.com",
    )

    snapshot = _safe_llm_config(llm)

    assert snapshot == {
        "model": "provider/model",
        "usage_id": "primary",
        "temperature": 0.2,
    }
    assert "secret" not in str(snapshot)


def test_skill_snapshot_excludes_mcp_connection_configuration() -> None:
    skill = SimpleNamespace(
        name="issue-tracker",
        content="Use the issue tracker for project tasks.",
        description="Issue tracker workflow",
        source="/skills/issue-tracker/SKILL.md",
        mcp_tools={"server": {"env": {"TOKEN": "secret-mcp-token"}}},
    )

    snapshot = _skill_snapshot(skill)

    assert snapshot == {
        "name": "issue-tracker",
        "content": "Use the issue tracker for project tasks.",
        "source": "/skills/issue-tracker/SKILL.md",
        "description": "Issue tracker workflow",
    }
    assert "secret-mcp-token" not in str(snapshot)
