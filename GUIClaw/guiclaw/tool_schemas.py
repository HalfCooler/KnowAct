"""
guiclaw.tool_schemas
====================
LLM tool/function schema builders for the GUIClaw ``computer_use`` action.

These are pure data-building functions with no runtime dependencies on the
agent loop, backends, or skill system.  They live at the ``guiclaw`` root so
that both the main agent loop and skill-executor infrastructure can import
them without creating circular dependencies.
"""

from __future__ import annotations

import copy
import hashlib
import re
from io import BytesIO
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from guiclaw.skills.deeplink import AppShortcutProfile


def minimal_tool_schema(action_type: str) -> dict[str, Any]:
    """Build a minimal ``computer_use`` tool schema restricted to *action_type*.

    Sending the full action-type enum on every grounder call adds ~200-300
    unnecessary input tokens.  Since the target action type is already known
    at call time, we strip the schema down to only the properties that are
    relevant for that action, cutting prompt size by ~60-70 %.
    """
    _coord = {
        "x": {"type": "number", "description": "Primary X coordinate."},
        "y": {"type": "number", "description": "Primary Y coordinate."},
        "relative": {"type": "boolean", "description": "True if [0,999] relative coords."},
    }
    _coord2 = {
        "x2": {"type": "number", "description": "End X for swipe/drag."},
        "y2": {"type": "number", "description": "End Y for swipe/drag."},
    }
    _text = {"text": {"type": "string", "description": "Text input or app identifier."}}
    _dur = {"duration_ms": {"type": "integer", "description": "Duration in ms."}}
    _summary = {
        "intent": {"type": "string", "description": "The purpose of the selected next action."},
        "summary": {"type": "string", "description": "The current task progress and visible UI state."},
    }

    prop_sets: dict[str, dict] = {
        "tap":               {**_coord},
        "double_tap":        {**_coord},
        "long_press":        {**_coord, **_dur},
        "swipe":             {**_coord, **_coord2},
        "drag":              {**_coord, **_coord2, **_dur},
        "input_text":        {**_text, **_coord},
        "hotkey":            {"key": {"type": "array", "items": {"type": "string"}, "description": "Keys for hotkey."}},
        "scroll":            {**_coord, "text": {"type": "string", "description": "Direction."}, "pixels": {"type": "integer", "description": "Scroll distance."}},
        "wait":              {},
        "open_app":          {**_text},
        "close_app":         {**_text},
        "back":              {},
        "home":              {},
        "done":              {"status": {"type": "string", "enum": ["success", "failure"]}},
        "request_intervention": {**_text},
    }
    props = prop_sets.get(action_type, {})
    props = {"action_type": {"type": "string", "enum": [action_type]}, **props, **_summary}
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": f"Perform a {action_type} GUI action on the device screen.",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": ["action_type"],
            },
        },
    }


def build_shortcut_tool_defs(
    shortcuts: dict[str, AppShortcutProfile],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str, str | None]]]:
    """Convert ``AppShortcutProfile`` entries into independent function tool definitions."""
    tools: list[dict[str, Any]] = []
    action_map: dict[str, tuple[str, str, str, str | None]] = {}
    for profile in shortcuts.values():
        pkg = re.sub(r"[^a-zA-Z0-9_]+", "_", profile.package).strip("_").lower()
        for dl in profile.deep_links:
            safe = _shortcut_slug("_".join(x for x in (dl.scheme, dl.host or "", dl.path or "") if x))
            digest = _shortcut_hash(f"{profile.package}|{dl.component}|{dl.uri_template}|{dl.path_kind or ''}")
            name = f"{pkg}__{safe}__{digest}" if safe else f"{pkg}__{digest}"
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": dl.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            })
            action_map[name] = ("open_deeplink", dl.uri_template, dl.component, None)
        for di in profile.deep_intents:
            safe = _shortcut_slug("_".join(x for x in (di.action.split(".")[-1], di.mime_type or "") if x))
            digest = _shortcut_hash(f"{profile.package}|{di.component}|{di.action}|{di.mime_type or ''}")
            name = f"{pkg}__{safe}__{digest}" if safe else f"{pkg}__{digest}"
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": di.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            })
            action_map[name] = ("open_intent", di.action, di.component, di.mime_type)
    return tools, action_map


COMPUTER_USE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": "Perform a GUI action on the device screen.",
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": [
                        "tap", "double_tap", "long_press", "swipe", "drag",
                        "input_text", "hotkey", "scroll",
                        "wait", "open_app", "close_app", "adb_command",
                        "back", "home", "done", "request_intervention",
                    ],
                },
                "x": {"type": "number", "description": "Primary X coordinate."},
                "y": {"type": "number", "description": "Primary Y coordinate."},
                "x2": {"type": "number", "description": "End X for swipe/drag."},
                "y2": {"type": "number", "description": "End Y for swipe/drag."},
                "text": {
                    "type": "string",
                    "description": (
                        "Text for input_text, direction for scroll, or app identifier "
                        "for open_app/close_app. Use a short reason for "
                        "request_intervention. On Android, use package names."
                    ),
                },
                "key": {"type": "array", "items": {"type": "string"}, "description": "Keys for hotkey."},
                "pixels": {"type": "integer", "description": "Scroll distance."},
                "duration_ms": {"type": "integer", "description": "Duration in ms."},
                "command_id": {"type": "string", "description": "Whitelisted adb_command skill id."},
                "params": {"type": "object", "description": "Typed params for adb_command."},
                "relative": {"type": "boolean", "description": "True if [0,999] relative coords."},
                "status": {"type": "string", "enum": ["success", "failure"], "description": "For done action."},
                "intent": {"type": "string", "description": "The purpose of the selected next action."},
                "summary": {"type": "string", "description": "The current task progress and visible UI state."},
            },
            "required": ["action_type", "intent", "summary"],
        },
    },
}


def build_computer_use_tool(
    *,
    allow_use_skill: bool = False,
    platform: str | None = None,
    available_apps: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Return the native GUI tool schema for the active platform."""
    platform_key = str(platform or "").strip().lower()
    desktop = platform_key in {"macos", "linux", "windows"}
    if not allow_use_skill and not desktop:
        return COMPUTER_USE_TOOL

    tool = copy.deepcopy(COMPUTER_USE_TOOL)
    properties = tool["function"]["parameters"]["properties"]
    if desktop:
        properties["action_type"]["enum"].remove("adb_command")
        properties.pop("command_id")
        properties.pop("params")
        app_names = tuple(
            sorted(
                {" ".join(str(name).split()) for name in available_apps if str(name).strip()},
                key=str.casefold,
            )
        )
        if platform_key == "linux" or not app_names:
            properties["action_type"]["enum"].remove("open_app")
            properties["text"]["description"] = (
                "Text for input_text, direction for scroll, or application name for close_app. "
                "Use a short reason for request_intervention."
            )
        else:
            properties["text"]["description"] = (
                "Text for input_text, direction for scroll, or application name for "
                "open_app/close_app. For open_app, text must exactly match one of these "
                f"installed application names: {'; '.join(app_names)}. "
                "Use a short reason for request_intervention."
            )
    if allow_use_skill:
        properties["action_type"]["enum"].append("use_skill")
        properties["skill_id"] = {
            "type": "string",
            "description": "Exact skill_id copied from the prompt-visible skill catalog.",
        }
        properties["arguments"] = {
            "type": "object",
            "description": "Arguments for the selected prompt-visible skill.",
        }
    return tool


def image_dimensions(raw: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(raw)) as image:
        return image.size


# -- internal helpers ---------------------------------------------------------

def _shortcut_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _shortcut_slug(value: str, *, max_len: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()[:max_len]


__all__ = [
    "COMPUTER_USE_TOOL",
    "build_computer_use_tool",
    "build_shortcut_tool_defs",
    "image_dimensions",
    "minimal_tool_schema",
]
