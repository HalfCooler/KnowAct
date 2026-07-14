"""Planner-only capability catalog contracts built from live runtime tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from nanobot.agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from nanobot.agent.planning_memory import PlanningMemoryHint

CapabilityType = Literal["gui", "tool", "mcp", "api"]


@dataclass(frozen=True)
class RouteSummary:
    """Compact planner-facing summary of one concrete execution route."""

    route_id: str
    capability: CapabilityType
    kind: str
    summary: str
    use_for: tuple[str, ...] = ()
    avoid_for: tuple[str, ...] = ()
    availability: str = "ready"


@dataclass(frozen=True)
class CapabilityCatalog:
    """Bounded set of currently available planner routes."""

    routes: tuple[RouteSummary, ...] = ()

    def to_prompt_lines(self) -> tuple[str, ...]:
        """Render compact planner-facing summaries without leaking raw schemas."""
        lines: list[str] = []
        for route in self.routes:
            lines.append(
                f"- {route.route_id} [{route.capability}/{route.kind}] {route.availability}: {route.summary}"
            )
            if route.use_for:
                lines.append(f"  use_for: {', '.join(route.use_for)}")
            if route.avoid_for:
                lines.append(f"  avoid_for: {', '.join(route.avoid_for)}")
        return tuple(lines)


@dataclass(frozen=True)
class PlanningContext:
    """Planner-only wrapper for context that should not change the entrypoint shape again."""

    catalog: CapabilityCatalog
    memory_hints: tuple["PlanningMemoryHint", ...] = ()
    gui_memory_context: str = ""  # os_guide / app_guide / icon_guide content for the planner
    active_gui_route: str = ""  # concrete gui route_id for this session (e.g. "gui.adb", "gui.desktop")


class CapabilityCatalogBuilder:
    """Build a compact allowlisted route catalog from the live tool registry."""

    _ROUTE_SPECS = (
        (
            "gui_task",
            "gui.desktop",
            "gui",
            "desktop",
            "Use the GUI subagent to operate apps and device surfaces",
            ("visual workflows", "app navigation", "screen-only actions"),
            ("direct host commands", "structured local tool calls"),
        ),
        (
            "exec",
            "tool.exec_shell",
            "tool",
            "shell",
            "Run short local shell commands on this host (including ADB commands for Android device control and voice assistant shortcuts)",
            ("system toggles", "local automation", "file inspection", "voice assistant tasks via ADB (set alarm, make call, send message, set timer, check weather, play music, create reminder)"),
            ("visual workflows", "unsafe destructive commands"),
        ),
        (
            "read_file",
            "tool.filesystem.read",
            "tool",
            "filesystem",
            "Read UTF-8 text files from the workspace",
            ("file inspection", "source review", "config lookup"),
            ("editing files", "binary inspection"),
        ),
        (
            "write_file",
            "tool.filesystem.write",
            "tool",
            "filesystem",
            "Write full file contents in the workspace",
            ("new files", "full rewrites"),
            ("small targeted edits", "visual workflows"),
        ),
        (
            "edit_file",
            "tool.filesystem.edit",
            "tool",
            "filesystem",
            "Apply targeted text edits to existing files",
            ("surgical code changes", "small content updates"),
            ("binary files", "visual workflows"),
        ),
        (
            "list_dir",
            "tool.filesystem.list",
            "tool",
            "filesystem",
            "List directories and inspect workspace layout",
            ("discovering files", "workspace navigation"),
            ("editing content", "host automation"),
        ),
        (
            "web_search",
            "tool.web.search",
            "tool",
            "web",
            "Search the public web for current information",
            ("finding sources", "latest information"),
            ("reading full pages", "local host operations"),
        ),
        (
            "web_fetch",
            "tool.web.fetch",
            "tool",
            "web",
            "Fetch and extract content from a specific URL",
            ("reading cited pages", "page extraction"),
            ("broad discovery", "local host operations"),
        ),
    )

    def build(
        self,
        tool_registry: ToolRegistry,
        *,
        gui_available: bool,
        exec_enabled: bool,
        gui_backend: str = "local",
    ) -> CapabilityCatalog:
        """Return planner-facing route summaries for currently usable routes.

        When *gui_backend* is ``"adb"`` or ``"scrcpy-adb"``, the gui_task route is emitted as
        ``gui.adb`` (targeting the connected Android device via ADB) instead of
        the default ``gui.desktop``.  All other backend values keep the existing
        ``gui.desktop`` route unchanged.
        """
        tool_names = set(tool_registry.tool_names)
        routes: list[RouteSummary] = []

        for tool_name, route_id, capability, kind, summary, use_for, avoid_for in self._ROUTE_SPECS:
            if tool_name == "gui_task" and not gui_available:
                continue
            if tool_name == "exec" and not exec_enabled:
                continue
            if tool_name not in tool_names:
                continue

            # Override the GUI route metadata when the active backend is ADB.
            if tool_name == "gui_task" and gui_backend in {"adb", "scrcpy-adb"}:
                route_id = "gui.adb"
                kind = "adb"
                summary = "Use the GUI subagent to operate apps on the connected Android device"

            # Override the GUI route metadata when the active backend is iOS/WDA.
            if tool_name == "gui_task" and gui_backend == "ios":
                route_id = "gui.ios"
                kind = "ios"
                summary = "Use the GUI subagent to operate apps on the connected iOS device"

            # Override the GUI route metadata when the active backend is HDC/HarmonyOS.
            if tool_name == "gui_task" and gui_backend == "hdc":
                route_id = "gui.hdc"
                kind = "hdc"
                summary = "Use the GUI subagent to operate apps on the connected HarmonyOS device"

            routes.append(
                RouteSummary(
                    route_id=route_id,
                    capability=capability,
                    kind=kind,
                    summary=summary,
                    use_for=use_for,
                    avoid_for=avoid_for,
                    availability="ready",
                )
            )

        for tool_name in sorted(name for name in tool_names if name.startswith("mcp_")):
            route = self._build_mcp_route(tool_name)
            if route is not None:
                routes.append(route)

        return CapabilityCatalog(routes=tuple(routes))

    def _build_mcp_route(self, tool_name: str) -> RouteSummary | None:
        """Normalize wrapped MCP tool names into stable planner route IDs."""
        suffix = tool_name.removeprefix("mcp_")
        if "_" not in suffix:
            return None
        server_name, original_name = suffix.split("_", 1)
        route_tool_name = original_name.replace("_", ".")
        return RouteSummary(
            route_id=f"mcp.{server_name}.{route_tool_name}",
            capability="mcp",
            kind="mcp",
            summary=f"Call MCP tool '{original_name}' on server '{server_name}'",
            use_for=("server-backed actions", "specialized integrations"),
            avoid_for=("local shell commands", "visual-only workflows"),
            availability="ready",
        )
