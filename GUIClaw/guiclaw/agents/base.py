"""Small compatibility base classes for vendored MobileWorld agents.

The source agents are copied from MobileWorld.  GUIClaw does not use their
network client methods directly, but keeping compatible base classes lets the
vendored modules import cleanly for parser and prompt parity tests.
"""

from __future__ import annotations

from typing import Any


class BaseAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.tools = kwargs.get("tools", [])
        self.instruction: str | None = None

    def initialize(self, instruction: str) -> None:
        self.instruction = instruction
        self.initialize_hook(instruction)

    def initialize_hook(self, instruction: str) -> None:
        del instruction

    def reset(self) -> None:
        pass

    def build_openai_client(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def openai_chat_completions_create(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise RuntimeError("Vendored MobileWorld agents are driven through GUIClaw async adapters.")


class MCPAgent(BaseAgent):
    pass
