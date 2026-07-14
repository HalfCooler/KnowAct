"""
guiclaw — Independent GUI automation framework.

Public API exports. ``GuiAgent`` is imported lazily to avoid loading
heavy agent machinery when only lightweight utilities are needed.
"""

from __future__ import annotations

from guiclaw.action import Action, ActionError, describe_action, parse_action
from guiclaw.interfaces import (
    DeviceBackend,
    LLMProvider,
    LLMResponse,
    ProgressCallback,
    ToolCall,
)
from guiclaw.observation import Observation


def __getattr__(name: str) -> object:
    if name == "GuiAgent":
        from guiclaw.agent import GuiAgent
        return GuiAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Action", "ActionError", "parse_action", "describe_action",
    "Observation",
    "LLMProvider", "DeviceBackend", "LLMResponse", "ToolCall", "ProgressCallback",
    "GuiAgent",
]
