"""
guiclaw.interfaces
==================
Core protocol definitions for the guiclaw GUI automation framework.

All types here are free of third-party dependencies so that any host agent
can conform to the protocols without pulling in extra packages.
For host-agent adapter examples, see repo-root ADAPTERS.md and nanobot/agent/gui_adapter.py.
"""

from __future__ import annotations

import dataclasses
import pathlib
import typing

if typing.TYPE_CHECKING:
    from guiclaw.action import Action
    from guiclaw.observation import Observation


@dataclasses.dataclass(frozen=True)
class ToolCall:
    """A single tool-call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, typing.Any]


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    """Response returned by an :class:`LLMProvider` after one round."""

    content: str
    tool_calls: list[ToolCall] | None = None
    raw: typing.Any = dataclasses.field(default=None, compare=False)
    usage: dict[str, int] = dataclasses.field(default_factory=dict, compare=False)
    ttft_s: float | None = dataclasses.field(default=None, compare=False)
    latency_s: float | None = dataclasses.field(default=None, compare=False)


@dataclasses.dataclass(frozen=True)
class InterventionRequest:
    """Structured request sent to a host when automation must pause."""

    task: str
    reason: str
    step_index: int
    platform: str
    foreground_app: str | None
    target: dict[str, typing.Any]


@dataclasses.dataclass(frozen=True)
class InterventionResolution:
    """Host response indicating whether automation may resume."""

    resume_confirmed: bool
    note: str | None = None


@typing.runtime_checkable
class LLMProvider(typing.Protocol):
    """Structural interface for LLM providers.

    Implementations must be async-safe.  The provider serializes messages to
    its own wire format, calls the remote API, and deserialises into an
    :class:`LLMResponse`.
    """

    async def chat(
        self,
        messages: list[dict[str, typing.Any]],
        tools: list[dict[str, typing.Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse: ...


@typing.runtime_checkable
class DeviceBackend(typing.Protocol):
    """Structural interface for device / OS automation backends.

    Responsibilities: observation (screenshot + metadata), execution (action
    dispatch), and preflight (connectivity checks).
    """

    async def observe(
        self,
        screenshot_path: pathlib.Path,
        timeout: float = 5.0,
    ) -> Observation: ...

    async def execute(
        self,
        action: Action,
        timeout: float = 5.0,
    ) -> str: ...

    async def preflight(self) -> None: ...

    async def list_apps(self) -> list[str]: ...

    @property
    def platform(self) -> str: ...


@typing.runtime_checkable
class InterventionHandler(typing.Protocol):
    """Host callback that coordinates a human intervention handoff."""

    async def request_intervention(
        self,
        request: InterventionRequest,
    ) -> InterventionResolution: ...


ProgressCallback = typing.Callable[[str], typing.Awaitable[None]]
