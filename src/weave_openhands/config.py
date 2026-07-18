from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

ContentTransform = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class TracingConfig:
    """Controls what the integration records.

    Full content capture is enabled by default because the integration is intended
    for agent debugging. ``content_transform`` can redact application-specific
    secrets before any string is attached to a span.
    """

    agent_name: str = "openhands"
    capture_content: bool = True
    content_transform: ContentTransform | None = None

    def __post_init__(self) -> None:
        if not self.agent_name.strip():
            raise ValueError("agent_name must be non-empty")

    def transform(self, value: str) -> str:
        if self.content_transform is None:
            return value
        return self.content_transform(value)
