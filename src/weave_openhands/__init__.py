from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from weave_openhands.config import ContentTransform, TracingConfig
from weave_openhands.instrumentation import (
    flush,
    instrument,
    is_instrumented,
    uninstrument,
)

try:
    __version__ = version("weave-openhands")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


def init(
    project_name: str,
    *,
    agent_name: str = "openhands",
    capture_content: bool = True,
    content_transform: ContentTransform | None = None,
):
    """Instrument OpenHands, then initialize Weave.

    Importing OpenHands first preserves its direct LiteLLM bindings before Weave's
    generic integration autopatching runs. This integration records those model
    calls itself, so tracing LiteLLM again would duplicate them.
    """

    instrument(
        TracingConfig(
            agent_name=agent_name,
            capture_content=capture_content,
            content_transform=content_transform,
        )
    )

    import weave

    return weave.init(project_name)


def finish(timeout_millis: int = 10_000) -> None:
    """Flush agent spans and finish the active Weave client."""

    import weave

    flush(timeout_millis)
    weave.finish()


__all__ = [
    "ContentTransform",
    "TracingConfig",
    "__version__",
    "finish",
    "flush",
    "init",
    "instrument",
    "is_instrumented",
    "uninstrument",
]
