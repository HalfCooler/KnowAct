"""TaskPlanner: decomposes tasks into AND/OR/ATOM execution trees via a single LLM call."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from nanobot.agent.capabilities import PlanningContext
from nanobot.agent.planning_memory import serialize_memory_hints

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NodeType = Literal["and", "or", "atom"]
CapabilityType = Literal["gui", "tool", "mcp", "api"]

_VALID_NODE_TYPES = {"and", "or", "atom"}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanNode:
    """A node in the AND/OR/ATOM execution tree.

    AND nodes execute all children sequentially; replanning is triggered on
    any child failure.  OR nodes try children in order until one succeeds;
    replanning is triggered only when all alternatives fail.  ATOM nodes are
    leaf tasks — they carry an ``instruction`` string and a ``capability``
    tag that tells the router which executor to use.
    """

    node_type: NodeType
    instruction: str = ""               # populated for ATOM nodes only
    capability: CapabilityType = "tool"  # populated for ATOM nodes only
    route_id: str | None = None
    route_reason: str = ""
    fallback_route_ids: tuple[str, ...] = ()
    params: dict[str, Any] | None = None  # structured executable params for the routed tool (ATOM only)
    children: tuple[PlanNode, ...] = field(default_factory=tuple)  # populated for AND/OR only

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: dict[str, Any] = {"type": self.node_type}
        if self.node_type == "atom":
            d["instruction"] = self.instruction
            d["capability"] = self.capability
            if self.route_id:
                d["route_id"] = self.route_id
            if self.route_reason:
                d["route_reason"] = self.route_reason
            if self.fallback_route_ids:
                d["fallback_route_ids"] = list(self.fallback_route_ids)
            if self.params is not None:
                d["params"] = self.params
        else:
            d["children"] = [child.to_dict() for child in self.children]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanNode:
        """Deserialize from a JSON dict produced by :meth:`to_dict`."""
        raw_node_type = data["type"]
        if not isinstance(raw_node_type, str):
            raise TypeError(f"Plan node type must be a string, got {type(raw_node_type)}")

        node_type = raw_node_type.strip().lower()
        if node_type not in _VALID_NODE_TYPES:
            raise ValueError(f"Unsupported plan node type: {raw_node_type!r}")

        if node_type == "atom":
            return cls(
                node_type="atom",
                instruction=data.get("instruction", ""),
                capability=data.get("capability", "tool"),
                route_id=data.get("route_id"),
                route_reason=data.get("route_reason", ""),
                fallback_route_ids=tuple(data.get("fallback_route_ids", [])),
                params=data.get("params"),
            )
        children = tuple(cls.from_dict(child) for child in data.get("children", []))
        return cls(node_type=node_type, children=children)


# ---------------------------------------------------------------------------
# LLM tool definition
# ---------------------------------------------------------------------------

def _build_create_plan_tool(active_gui_route: str = "gui.desktop") -> dict[str, Any]:
    """Return the create_plan tool schema with the correct GUI route example."""
    return {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": (
                "Decompose a user task into an AND/OR/ATOM execution tree. "
                "AND nodes execute all children sequentially. "
                "OR nodes try children until one succeeds. "
                f"ATOM nodes are leaf tasks with a capability type (gui/tool/mcp/api), "
                f"and may optionally include route_id (e.g. {active_gui_route} or tool.exec_shell), "
                "route_reason, fallback_route_ids, and params (a dict of concrete executable "
                "parameter values for the routed tool)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tree": {
                        "type": "object",
                        "description": (
                            "Root node.  Each node has 'type' (and/or/atom). "
                            "AND/OR nodes have a 'children' array.  "
                            "ATOM nodes have 'instruction' (human-readable description of what to do) and "
                            "'capability' (gui/tool/mcp/api), and may optionally include 'route_id', "
                            "'route_reason', 'fallback_route_ids', and 'params' (dict of concrete "
                            "executable parameter values for the routed tool)."
                        ),
                    }
                },
                "required": ["tree"],
            },
        },
    }

# ---------------------------------------------------------------------------
# TaskPlanner
# ---------------------------------------------------------------------------

class TaskPlanner:
    """Decomposes tasks into AND/OR/ATOM execution trees via a single LLM call.

    The planner issues one LLM call with the ``create_plan`` tool forced on.
    It optionally reads SKILL.md files through a ``SkillsLoader`` to inform
    the LLM about available capabilities before requesting the decomposition.

    Usage::

        planner = TaskPlanner(llm=my_llm, skills_loader=skills)
        tree: PlanNode = await planner.plan("Turn on Wi-Fi and check weather")
    """

    def __init__(self, llm: Any, skills_loader: Any = None) -> None:
        """
        Args:
            llm: An LLM provider that exposes an async ``chat()`` method
                 accepting ``messages``, ``tools``, and ``tool_choice`` kwargs.
            skills_loader: Optional :class:`~nanobot.agent.skills.SkillsLoader`
                           instance for reading the capability registry.
        """
        self._llm = llm
        self._skills_loader = skills_loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(
        self,
        task: str,
        *,
        context: str = "",
        planning_context: PlanningContext | None = None,
    ) -> PlanNode:
        """Decompose *task* into an execution tree.

        Args:
            task: The user's task description.
            context: Optional additional context (e.g. current state summary,
                     list of already-completed subtasks).

        Returns:
            Root :class:`PlanNode` of the execution tree.

        Note:
            If the LLM fails to call ``create_plan`` (e.g. model misbehaves),
            the method falls back to a single ATOM node with capability ``gui``
            so that execution can always proceed.
        """
        system_prompt = self._build_system_prompt(planning_context=planning_context)
        user_prompt = self._build_user_prompt(task, context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug("Requesting plan for task: %r", task)

        active_gui_route = (
            planning_context.active_gui_route if planning_context and planning_context.active_gui_route else "gui.desktop"
        )
        response = await self._request_plan(messages, active_gui_route=active_gui_route)

        if not response.tool_calls:
            logger.warning(
                "LLM did not call create_plan; falling back to single ATOM node. %s",
                self._format_planner_response_diagnostics(response),
            )
            return PlanNode(node_type="atom", instruction=task, capability="gui")

        args = response.tool_calls[0].arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                logger.warning(
                    "create_plan arguments are not valid JSON; using fallback ATOM. %s",
                    self._format_planner_response_diagnostics(response),
                )
                return PlanNode(node_type="atom", instruction=task, capability="gui")

        tree_data = args.get("tree", {"type": "atom", "instruction": task, "capability": "gui"})
        try:
            return PlanNode.from_dict(tree_data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to parse plan tree: %s; using fallback ATOM. %s",
                exc,
                self._format_planner_response_diagnostics(response),
            )
            return PlanNode(node_type="atom", instruction=task, capability="gui")

    async def replan(
        self,
        task: str,
        *,
        completed: list[str],
        failed: str,
        remaining: list[str],
    ) -> PlanNode:
        """Replan after a failure, providing context about completed and failed work.

        Args:
            task: The original top-level task.
            completed: Instructions of successfully completed subtasks.
            failed: Instruction (or description) of the subtask that failed.
            remaining: Instructions of subtasks that were not yet attempted.

        Returns:
            A new root :class:`PlanNode` for continued execution.
        """
        context_lines: list[str] = []
        if completed:
            context_lines.append("Completed subtasks:\n" + "\n".join(f"- {c}" for c in completed))
        context_lines.append(f"Failed subtask: {failed}")
        if remaining:
            context_lines.append("Remaining subtasks:\n" + "\n".join(f"- {r}" for r in remaining))
        context = "\n\n".join(context_lines)
        logger.debug("Replanning task %r after failure of %r", task, failed)
        return await self.plan(task, context=context)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, *, planning_context: PlanningContext | None = None) -> str:
        """Build the planner system prompt, injecting the capability registry."""
        lines = [
            "You are a task planner. Decompose the user's task into an AND/OR/ATOM execution tree.",
            "",
            "Node types:",
            "- AND: all children must succeed, executed sequentially.  If one fails, replanning occurs.",
            "- OR: try children in order until one succeeds.  If all fail, replanning occurs.",
            "- ATOM: a single actionable subtask.  Must have 'instruction' (what to do) and "
            "'capability' (how to do it).",
            "",
            "Capability types:",
            "- gui: interact with a device screen (tap, type, swipe, navigate apps)",
            "- tool: use a local tool or function",
            "- mcp: call an MCP server tool",
            "- api: make a direct API call",
            "",
            "Rules:",
            "- Simple tasks should be a single ATOM (no unnecessary AND/OR wrapping).",
            "- Use AND for tasks that require sequential steps.",
            "- Use OR for tasks with alternative approaches.",
            "- Each ATOM instruction should be a clear, focused, single-step subgoal.",
            "- Choose the capability type based on the available capabilities listed below.",
            "- When a concrete route is available, include route_id and a short route_reason.",
            "- Add fallback_route_ids for host operations that can safely fall back to GUI.",
            "- When route_id points to a concrete tool or MCP route, include 'params' with the exact "
            "executable parameter values the tool expects.  For example, route 'tool.exec_shell' "
            "uses params={\"command\": \"ls -la /tmp\"}, route 'tool.filesystem.read' uses "
            "params={\"path\": \"/etc/hosts\"}, route 'tool.filesystem.write' uses "
            "params={\"path\": \"out.txt\", \"content\": \"hello\"}, route 'tool.web.search' "
            "uses params={\"query\": \"python asyncio tutorial\"}, and MCP routes use "
            "params={\"input\": \"value\"} or tool-specific keys.",
            "- 'instruction' stays as the human-readable description; 'params' holds machine-executable values.",
            "- For gui capability, params is not needed (the GUI subagent interprets instruction directly).",
        ]

        if planning_context is not None and planning_context.catalog.routes:
            lines.extend(
                [
                    "",
                    "Available routes right now:",
                    *planning_context.catalog.to_prompt_lines(),
                ]
            )
        if planning_context is not None and planning_context.memory_hints:
            rendered_hints = serialize_memory_hints(planning_context.memory_hints)
            lines.extend(
                [
                    "",
                    "Routing memory hints:",
                    *[f"- {line}" for line in rendered_hints],
                ]
            )
            if len(rendered_hints) < len(planning_context.memory_hints):
                lines.append("- ... additional routing hints omitted for brevity")

        # Inject summary lines from SKILL.md files when a loader is available.
        if self._skills_loader is not None:
            try:
                skills = self._skills_loader.list_skills()
                if skills:
                    lines.extend(["", "Available capabilities:"])
                    for skill_info in skills:
                        # Prefer the parsed frontmatter description over raw content lines.
                        meta = self._skills_loader.get_skill_metadata(skill_info["name"])
                        if meta and meta.get("description"):
                            summary = meta["description"]
                        else:
                            content = self._skills_loader.load_skill(skill_info["name"])
                            if content:
                                summary_lines = [ln for ln in content.splitlines() if ln.strip()][:3]
                                summary = " ".join(summary_lines)
                            else:
                                summary = skill_info["name"]
                        lines.append(f"- {skill_info['name']}: {summary}")
            except Exception as exc:  # pragma: no cover — graceful degradation
                logger.warning("Failed to load skills for planner prompt: %s", exc)

        if planning_context is not None and planning_context.gui_memory_context:
            lines.extend(
                [
                    "",
                    "Device and app knowledge (use this to refine GUI task instructions):",
                    planning_context.gui_memory_context,
                ]
            )

        if planning_context is not None and planning_context.active_gui_route:
            lines.extend(
                [
                    "",
                    "Active GUI route:",
                    f"- The current GUI backend is '{planning_context.active_gui_route}'. "
                    f"Use route_id='{planning_context.active_gui_route}' for ALL GUI subtasks "
                    f"that target the device/screen. Do NOT use any other gui.* route_id.",
                ]
            )

        lines.extend(
            [
                "",
                "Call the create_plan tool with your decomposition.",
                "For ATOM nodes, include route_id, route_reason, and fallback_route_ids when the route choice matters.",
            ]
        )
        return "\n".join(lines)

    def _build_user_prompt(self, task: str, context: str) -> str:
        """Build the user prompt, optionally including prior-execution context."""
        parts = [f"Task: {task}"]
        if context:
            parts.extend(["", "Context:", context])
        return "\n".join(parts)

    async def _request_plan(
        self, messages: list[dict[str, Any]], *, active_gui_route: str = "gui.desktop"
    ) -> Any:
        """Request a plan, retrying with auto tool choice when forced choice is unsupported."""
        create_plan_tool = _build_create_plan_tool(active_gui_route)
        forced_tool_choice = {"type": "function", "function": {"name": "create_plan"}}
        response = await self._llm.chat(
            messages=messages,
            tools=[create_plan_tool],
            tool_choice=forced_tool_choice,
        )
        if self._should_retry_with_auto_tool_choice(response):
            logger.warning(
                "Planner forced create_plan tool_choice unsupported; retrying with auto. %s",
                self._format_planner_response_diagnostics(response),
            )
            response = await self._llm.chat(
                messages=messages,
                tools=[create_plan_tool],
                tool_choice="auto",
            )
        return response

    @staticmethod
    def _should_retry_with_auto_tool_choice(response: Any) -> bool:
        """Return True when the provider rejects forced tool choice in thinking mode."""
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason != "error":
            return False
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            return False
        normalized = content.lower()
        return (
            "tool_choice" in normalized
            and "thinking mode" in normalized
            and ("does not support" in normalized or "unsupported" in normalized)
        )

    @staticmethod
    def _format_planner_response_diagnostics(response: Any) -> str:
        """Summarize the planner model response for failure diagnostics."""
        finish_reason = getattr(response, "finish_reason", None)
        content = getattr(response, "content", None)
        tool_calls = getattr(response, "tool_calls", None)

        content_preview = ""
        if isinstance(content, str) and content.strip():
            content_preview = " ".join(content.split())[:240]

        tool_call_names: list[str] = []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls[:5]:
                name = getattr(tool_call, "name", None)
                if isinstance(name, str) and name:
                    tool_call_names.append(name)

        parts = [
            f"finish_reason={finish_reason!r}",
            f"tool_call_count={len(tool_calls) if isinstance(tool_calls, list) else 0}",
        ]
        if tool_call_names:
            parts.append(f"tool_call_names={tool_call_names}")
        if content_preview:
            parts.append(f"content_preview={content_preview!r}")
        return " ".join(parts)
