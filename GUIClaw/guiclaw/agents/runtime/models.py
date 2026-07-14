"""MobileWorld JSONAction constants mirrored for GUIClaw adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CLICK = "click"
LONG_PRESS = "long_press"
DOUBLE_TAP = "double_tap"
DRAG = "drag"
SCROLL = "scroll"
INPUT_TEXT = "input_text"
OPEN_APP = "open_app"
NAVIGATE_HOME = "navigate_home"
NAVIGATE_BACK = "navigate_back"
KEYBOARD_ENTER = "keyboard_enter"
WAIT = "wait"
ANSWER = "answer"
FINISHED = "finished"
ASK_USER = "ask_user"
MCP = "mcp"
UNKNOWN = "unknown"
ENV_FAIL = "env_fail"


@dataclass
class JSONAction:
    action_type: str
    x: float | None = None
    y: float | None = None
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    text: str | None = None
    app_name: str | None = None
    direction: str | None = None
    duration: float | None = None
    action_json: dict[str, Any] | None = None
    action_name: str | None = None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        data = dict(self.__dict__)
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data

    def dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)
