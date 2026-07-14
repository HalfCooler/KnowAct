"""
guiclaw.observation
===================
Screen-state snapshot consumed by the LLM at each agent step.

``Observation`` is *mutable* because ``screenshot_path`` may be updated after
capture and ``extra`` is enriched incrementally by backends.
"""

from __future__ import annotations

import dataclasses
import re
import textwrap
import typing

_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
_RELATIVE_GRID_MAX = 999


@dataclasses.dataclass
class Observation:
    """A snapshot of the device's current screen state."""

    screenshot_path: str | None
    screen_width: int
    screen_height: int
    foreground_app: str | None = None
    platform: str = "unknown"
    extra: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    @property
    def has_screenshot(self) -> bool:
        return self.screenshot_path is not None

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.screen_width, self.screen_height)

    def to_user_text(
        self,
        task: str,
        step_index: int,
        app_hint: str | None = None,
        coordinate_instruction: str = "Prefer absolute pixel coordinates.",
        include_extra: bool = True,
    ) -> str:
        """Render a structured text block for inclusion in an LLM message."""
        app_name = app_hint or self.foreground_app or "unknown"
        use_relative_coordinates = _uses_relative_coordinates(coordinate_instruction)
        lines: list[str] = [
            f"Step {step_index + 1}",
            f"Task: {task}",
            "",
        ]
        if use_relative_coordinates:
            lines.append(f"Platform: {self.platform}")
        else:
            lines.append(
                f"Screen: {self.screen_width} x {self.screen_height} px "
                f"(platform: {self.platform})"
            )
        lines.extend([
            f"Foreground app: {app_name}",
            f"Coordinates: {coordinate_instruction}",
        ])
        if include_extra and self.extra:
            lines.append("")
            lines.append("Additional context:")
            extra = (
                _extra_with_relative_ui_bounds(self.extra)
                if use_relative_coordinates else self.extra
            )
            for key, value in self.extra.items():
                value = extra.get(key, value)
                value_str = _format_extra_value(key, value)
                if "\n" in value_str:
                    indented = textwrap.indent(value_str, prefix="    ")
                    lines.append(f"  {key}:\n{indented}")
                else:
                    lines.append(f"  {key}: {value_str}")
        return "\n".join(lines)


def _uses_relative_coordinates(coordinate_instruction: str) -> bool:
    normalized = coordinate_instruction.lower()
    return "relative" in normalized and ("[0, 999]" in normalized or "0-999" in normalized)


def _extra_with_relative_ui_bounds(extra: dict[str, typing.Any]) -> dict[str, typing.Any]:
    ui_tree = extra.get("ui_tree")
    if not isinstance(ui_tree, list):
        return extra

    parsed_bounds = [
        parsed
        for node in ui_tree
        if isinstance(node, dict)
        for parsed in [_parse_bounds(node.get("bounds"))]
        if parsed is not None
    ]

    out = dict(extra)
    if parsed_bounds:
        root_x1 = min(bounds[0] for bounds in parsed_bounds)
        root_y1 = min(bounds[1] for bounds in parsed_bounds)
        root_x2 = max(bounds[2] for bounds in parsed_bounds)
        root_y2 = max(bounds[3] for bounds in parsed_bounds)
        width = max(1, root_x2 - root_x1)
        height = max(1, root_y2 - root_y1)
    else:
        root_x1 = 0
        root_y1 = 0
        width = 1
        height = 1

    out["ui_tree"] = [
        _node_with_relative_bounds(
            node,
            root_x1=root_x1,
            root_y1=root_y1,
            width=width,
            height=height,
        )
        for node in ui_tree
    ]
    return out


def _node_with_relative_bounds(
    node: typing.Any,
    *,
    root_x1: int,
    root_y1: int,
    width: int,
    height: int,
) -> typing.Any:
    if not isinstance(node, dict):
        return node
    out = dict(node)
    parsed = _parse_bounds(out.pop("bounds", None))
    if parsed is not None:
        x1, y1, x2, y2 = parsed
        out["relative_bounds"] = (
            f"[{_to_relative(x1 - root_x1, width)},{_to_relative(y1 - root_y1, height)}]"
            f"[{_to_relative(x2 - root_x1, width)},{_to_relative(y2 - root_y1, height)}]"
        )
    return out


def _parse_bounds(value: typing.Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = _BOUNDS_RE.fullmatch(value.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _to_relative(value: int, extent: int) -> int:
    if extent <= 1:
        return 0
    relative = round(value / (extent - 1) * _RELATIVE_GRID_MAX)
    return max(0, min(relative, _RELATIVE_GRID_MAX))


def _format_extra_value(key: str, value: typing.Any) -> str:
    if isinstance(value, list):
        limit = 20 if key == "ui_tree" else 40
        shown = value[:limit]
        suffix = f" ... (+{len(value) - limit} more)" if len(value) > limit else ""
        return f"{shown}{suffix}"
    return str(value)
