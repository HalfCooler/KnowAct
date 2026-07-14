"""TreeRouter: walks AND/OR/ATOM plan trees and dispatches ATOM nodes by capability type."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability priority for OR-node ordering (lower value = tried first)
# ---------------------------------------------------------------------------

_CAPABILITY_PRIORITY: dict[str, int] = {"mcp": 0, "tool": 1, "gui": 2, "api": 3}


# ---------------------------------------------------------------------------
# Route resolution tables
# ---------------------------------------------------------------------------

# Maps planner route_id (tool.* form) to the ToolRegistry key used at dispatch.
_ROUTE_ID_TO_TOOL_NAME: dict[str, str] = {
    "tool.exec_shell":       "exec",
    "tool.filesystem.read":  "read_file",
    "tool.filesystem.write": "write_file",
    "tool.filesystem.edit":  "edit_file",
    "tool.filesystem.list":  "list_dir",
    "tool.web.search":       "web_search",
    "tool.web.fetch":        "web_fetch",
}

# Maps registry tool name to the single instruction-derived parameter key.
# Tools absent from this map require structured multi-parameter input and
# cannot be driven from an instruction string alone (e.g. write_file, edit_file).
_INSTRUCTION_PARAM: dict[str, str] = {
    "exec":       "command",
    "read_file":  "path",
    "list_dir":   "path",
    "web_search": "query",
    "web_fetch":  "url",
}


def _is_error_output(output_str: str) -> bool:
    """Return True when *output_str* signals a failed tool execution.

    Two cases are detected:

    * The string starts with ``"Error"`` (legacy tool error convention).
    * The string contains ``"Exit code: N"`` where N is a non-zero integer
      (ExecTool format: ``"STDERR:\\n...\\n\\nExit code: {returncode}"``).
    """
    if output_str.startswith("Error"):
        return True
    match = re.search(r"Exit code:\s*(\d+)", output_str)
    if match and match.group(1) != "0":
        return True
    return False


def _resolve_route(
    route_id: str,
    registry: Any,
) -> tuple[str, str | None] | None:
    """Map a planner route_id to (registry_tool_name, primary_param_key).

    Returns ``None`` when the route_id is unrecognized or the required tool is
    absent from the registry.  Returns ``(tool_name, None)`` for multi-parameter
    tools (write_file, edit_file) that cannot be driven by instruction text alone.

    Local tool routes follow the ``tool.*`` prefix convention.
    MCP routes follow the ``mcp.{server}.{tool}`` convention and resolve to the
    registry key ``mcp_{server}_{tool}`` with ``"input"`` as the primary parameter.
    """
    # Local tool routes
    if route_id in _ROUTE_ID_TO_TOOL_NAME:
        tool_name = _ROUTE_ID_TO_TOOL_NAME[route_id]
        if not registry.has(tool_name):
            return None
        param_key = _INSTRUCTION_PARAM.get(tool_name)  # None for multi-param tools
        return (tool_name, param_key)

    # MCP routes: mcp.{server}.{tool} -> mcp_{server}_{tool}
    if route_id.startswith("mcp."):
        suffix = route_id[4:]
        parts = suffix.split(".", 1)
        if len(parts) == 2:
            tool_name = f"mcp_{parts[0]}_{parts[1]}"
            if not registry.has(tool_name):
                return None
            return (tool_name, "input")

    return None


# ---------------------------------------------------------------------------
# Result / context data structures
# ---------------------------------------------------------------------------


@dataclass
class NodeResult:
    """Result of executing a single plan tree node."""

    success: bool
    output: str = ""
    error: str | None = None
    trace_paths: list[str] = field(default_factory=list)


@dataclass
class RouterContext:
    """Mutable context threaded through the tree during execution.

    Holds references to the executors that ATOM nodes dispatch to, plus
    a running list of completed ATOM instructions (used for replanning).
    """

    task: str
    completed: list[str] = field(default_factory=list)
    gui_agent: Any = None
    tool_registry: Any = None
    mcp_client: Any = None


# ---------------------------------------------------------------------------
# TreeRouter
# ---------------------------------------------------------------------------


class TreeRouter:
    """Walk an AND/OR/ATOM plan tree and dispatch each ATOM to its executor.

    Dispatch table (by :pyattr:`PlanNode.capability`):

    * **gui** — ``GuiAgent.run(instruction)`` with ``max_retries=1``
    * **tool** — ``ToolRegistry.execute(tool_name, params)`` via route resolution
    * **mcp** — ``ToolRegistry.execute(mcp_{server}_{tool}, params)`` via route resolution
    * **api** — not yet implemented

    AND nodes execute children sequentially in declaration order.  Each child
    sees the accumulated ``context.completed`` from all previously finished
    siblings, which preserves planner semantics and avoids shared-state races.
    On the first child failure, replanning is attempted (subject to budget);
    remaining children are not started until the failed child is resolved.

    On any AND-child failure, replanning is attempted if budget remains.

    OR nodes try children in **mcp > tool > gui > api** priority order until
    one succeeds.  If all alternatives fail, replanning is attempted before
    reporting failure.
    """

    def __init__(
        self,
        planner: Any = None,
        max_replans: int = 2,
    ) -> None:
        """
        Args:
            planner: Optional :class:`TaskPlanner` used for replanning after
                     failures.
            max_replans: Maximum number of replan attempts allowed across the
                         whole tree execution.
        """
        self._planner = planner
        self._max_replans = max_replans
        self._replan_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, node: Any, context: RouterContext) -> NodeResult:
        """Recursively execute a :class:`PlanNode` tree."""
        from nanobot.agent.planner import PlanNode

        if not isinstance(node, PlanNode):
            return NodeResult(success=False, error=f"Expected PlanNode, got {type(node)}")

        node_type = str(node.node_type).lower()

        if node_type == "atom":
            return await self._dispatch_atom(node, context)
        if node_type == "and":
            return await self._execute_and(node, context)
        if node_type == "or":
            return await self._execute_or(node, context)
        return NodeResult(success=False, error=f"Unknown node type: {node.node_type}")

    # ------------------------------------------------------------------
    # Composite-node handlers
    # ------------------------------------------------------------------

    async def _execute_and(self, node: Any, context: RouterContext) -> NodeResult:
        """AND: sequential execution in declaration order.

        Children run one at a time.  Each child receives its own snapshot of
        ``context.completed`` at the moment it starts, so it can see the work
        finished by all previous siblings without risking shared-list mutation.
        After each child succeeds its ATOM instruction is merged back into
        ``context.completed`` immediately, making it visible to subsequent
        siblings.

        On any child failure, replanning is attempted (subject to budget).
        If replanning also fails, execution stops and the error is returned
        without running the remaining children.

        All collected outputs and trace paths are preserved across the run.
        """
        all_outputs: list[str] = []
        all_traces: list[str] = []

        for i, child in enumerate(node.children):
            child_ctx = RouterContext(
                task=context.task,
                completed=list(context.completed),
                gui_agent=context.gui_agent,
                tool_registry=context.tool_registry,
                mcp_client=context.mcp_client,
            )
            result = await self.execute(child, child_ctx)
            all_traces.extend(result.trace_paths)

            if result.success:
                if result.output:
                    all_outputs.append(result.output)
                if str(child.node_type).lower() == "atom":
                    context.completed.append(child.instruction)
                continue

            # Child failed — attempt replan before giving up.
            if self._planner is not None and self._replan_count < self._max_replans:
                replan_result = await self._try_replan_and(
                    node=node,
                    failed_index=i,
                    failed_child=child,
                    context=context,
                    all_outputs=list(all_outputs),
                    all_traces=all_traces,
                )
                if replan_result is not None:
                    all_traces.extend(replan_result.trace_paths)
                    if replan_result.success:
                        continue

            merged_outputs = "\n".join(all_outputs)
            return NodeResult(
                success=False,
                output=merged_outputs,
                error=result.error or f"AND child {i} failed",
                trace_paths=all_traces,
            )

        return NodeResult(
            success=True,
            output="\n".join(all_outputs),
            trace_paths=all_traces,
        )

    async def _execute_or(self, node: Any, context: RouterContext) -> NodeResult:
        """OR: try children in priority order (mcp > tool > gui) until one succeeds.

        Children are sorted by capability priority before iteration so that
        the most capable (lowest-latency / highest-fidelity) executor is
        always tried first regardless of the planner's output order.  Within
        the same capability tier the original child order is preserved via a
        stable sort.
        """
        all_traces: list[str] = []
        last_error: str | None = None

        # Stable sort by capability priority; ties preserve original order.
        sorted_children = sorted(
            node.children,
            key=lambda c: _CAPABILITY_PRIORITY.get(getattr(c, "capability", "gui"), 99),
        )

        for child in sorted_children:
            result = await self.execute(child, context)
            all_traces.extend(result.trace_paths)
            if result.success:
                if str(child.node_type).lower() == "atom":
                    context.completed.append(child.instruction)
                return NodeResult(success=True, output=result.output, trace_paths=all_traces)
            last_error = result.error

        # All children failed — attempt replan if budget remains.
        if self._planner is not None and self._replan_count < self._max_replans:
            self._replan_count += 1
            try:
                new_plan = await self._planner.replan(
                    context.task,
                    completed=context.completed,
                    failed="All OR alternatives failed",
                    remaining=[],
                )
                replan_result = await self.execute(new_plan, context)
                all_traces.extend(replan_result.trace_paths)
                return replan_result
            except Exception as exc:
                logger.warning("Replanning after OR failure failed: %s", exc)

        return NodeResult(
            success=False,
            error=last_error or "All OR children failed",
            trace_paths=all_traces,
        )

    # ------------------------------------------------------------------
    # ATOM dispatch
    # ------------------------------------------------------------------

    async def _dispatch_atom(self, node: Any, context: RouterContext) -> NodeResult:
        """Route an ATOM node to the appropriate executor by capability type."""
        capability = node.capability
        instruction = node.instruction
        logger.info("Dispatching ATOM: capability=%s, instruction=%r", capability, instruction)

        try:
            if capability == "gui":
                return await self._run_gui(instruction, context)
            if capability == "tool":
                return await self._run_tool(node, context)
            if capability == "mcp":
                return await self._run_mcp(node, context)
            if capability == "api":
                return NodeResult(success=False, error="API capability not yet implemented")
            return NodeResult(success=False, error=f"Unknown capability: {capability}")
        except Exception as exc:
            logger.error("ATOM dispatch failed: %s", exc)
            return NodeResult(success=False, error=str(exc))

    async def _run_gui(self, instruction: str, context: RouterContext) -> NodeResult:
        """Dispatch to ``GuiAgent.run()`` for GUI tasks."""
        if context.gui_agent is None:
            return NodeResult(success=False, error="No GuiAgent configured for GUI dispatch")
        result = await context.gui_agent.run(instruction, max_retries=1)
        return NodeResult(
            success=result.success,
            output=result.summary,
            error=result.error,
            trace_paths=[result.trace_path] if getattr(result, "trace_path", None) else [],
        )

    async def _dispatch_with_fallback(
        self, node: Any, context: RouterContext,
    ) -> NodeResult:
        """Try primary route_id, then each fallback_route_id in order.

        Handles three special cases:

        * ``gui.desktop`` fallback — delegates to :meth:`_run_gui` when
          ``gui_agent`` is available; skips with a diagnostic when it is absent.
        * Multi-parameter routes (``param_key`` is ``None``) — skipped with a
          warning so the chain can continue to the next fallback.
        * All routes exhausted — returns a structured failure listing every
          route that was tried.

        The fallback chain is intentionally shared between ``_run_tool`` and
        ``_run_mcp``: once fallbacks are present, the capability boundary is
        treated as advisory and the best available route wins.
        """
        route_ids = list(filter(None, [node.route_id])) + list(node.fallback_route_ids or ())
        logger.info(
            "Dispatch: planned_route=%s fallbacks=%s instruction=%r",
            node.route_id, list(node.fallback_route_ids), node.instruction,
        )
        tried: list[str] = []

        for route_id in route_ids:
            # gui.desktop, gui.adb, gui.ios, and gui.hdc are GUI sentinels that delegate to the GUI subagent
            if route_id in ("gui.desktop", "gui.adb", "gui.ios", "gui.hdc"):
                if context.gui_agent is None:
                    logger.warning("Dispatch: %s fallback unavailable (no gui_agent)", route_id)
                    tried.append(f"{route_id}(unavailable:no_gui_agent)")
                    continue
                logger.info(
                    "Dispatch: fallback_taken=%s (primary was %s)", route_id, node.route_id,
                )
                return await self._run_gui(node.instruction, context)

            resolved = _resolve_route(route_id, context.tool_registry)
            if resolved is None:
                logger.warning("Dispatch: route unavailable route_id=%s", route_id)
                tried.append(f"{route_id}(unavailable)")
                continue

            tool_name, param_key = resolved
            if param_key is None and node.params is None:
                logger.warning(
                    "Dispatch: route %s requires structured parameters and no params provided, skipping",
                    route_id,
                )
                tried.append(f"{route_id}(multi-param)")
                continue

            logger.info(
                "Dispatch: resolved_route=%s tool=%s", route_id, tool_name,
            )
            if node.params is not None:
                params = dict(node.params)
            else:
                params = {param_key: node.instruction}
            output = await context.tool_registry.execute(tool_name, params)
            output_str = str(output) if output is not None else ""
            if _is_error_output(output_str):
                logger.warning("Dispatch: route %s failed: %s", route_id, output_str[:120])
                tried.append(f"{route_id}(error)")
                continue

            if route_id != node.route_id:
                logger.info("Dispatch: fallback_taken=%s (primary was %s)", route_id, node.route_id)
            return NodeResult(success=True, output=output_str)

        return NodeResult(
            success=False,
            error=(
                f"All routes failed for {node.capability} atom: "
                f"{node.instruction!r} (tried: {tried})"
            ),
        )

    async def _run_tool(self, node: Any, context: RouterContext) -> NodeResult:
        """Dispatch a tool atom through ToolRegistry using route resolution.

        Resolves the node's ``route_id`` to a registry tool name and single
        primary parameter key, then calls ``ToolRegistry.execute()``.  Returns
        a structured failure when:

        * ``context.tool_registry`` is absent
        * ``node.route_id`` is ``None``
        * the route cannot be found in the registry
        * the route requires structured multi-parameter input (param_key is ``None``)
        * the registry returns an error string starting with ``"Error"``

        When ``fallback_route_ids`` is non-empty, delegates to
        :meth:`_dispatch_with_fallback` which chains through all alternatives
        before reporting failure.
        """
        if context.tool_registry is None:
            return NodeResult(success=False, error="No ToolRegistry configured for tool dispatch")

        if node.route_id is None:
            logger.warning("Dispatch: no route_id on tool atom, instruction=%r", node.instruction)
            return NodeResult(
                success=False,
                error=f"No route_id specified for tool atom: {node.instruction!r}",
            )

        # Delegate to fallback chain when alternatives are declared
        if node.fallback_route_ids:
            return await self._dispatch_with_fallback(node, context)

        resolved = _resolve_route(node.route_id, context.tool_registry)
        if resolved is None:
            logger.warning("Dispatch: route unavailable route_id=%s", node.route_id)
            return NodeResult(
                success=False,
                error=f"Route unavailable: {node.route_id} (tool not in registry)",
            )

        tool_name, param_key = resolved
        if node.params is not None:
            params = dict(node.params)
        elif param_key is not None:
            params = {param_key: node.instruction}
        else:
            logger.warning(
                "Dispatch: route %s requires structured parameters, cannot dispatch from instruction alone",
                node.route_id,
            )
            return NodeResult(
                success=False,
                error=(
                    f"Route {node.route_id} requires structured parameters; "
                    "instruction-only dispatch not supported"
                ),
            )

        logger.info(
            "Dispatch: planned_route=%s resolved_route=%s tool=%s",
            node.route_id, node.route_id, tool_name,
        )
        output = await context.tool_registry.execute(tool_name, params)
        output_str = str(output) if output is not None else ""
        if _is_error_output(output_str):
            return NodeResult(success=False, error=output_str, output=output_str)
        return NodeResult(success=True, output=output_str)

    async def _run_mcp(self, node: Any, context: RouterContext) -> NodeResult:
        """Dispatch an MCP atom through ToolRegistry using route resolution.

        MCP tools are registered under ``mcp_{server}_{tool}`` keys by
        MCPToolWrapper; this method resolves the planner's ``mcp.{server}.{tool}``
        route_id to that registry key and dispatches via ``ToolRegistry.execute()``.

        Note: ``context.mcp_client`` is retained for backward compatibility but is
        not used here — all MCP dispatch goes through the shared ToolRegistry.

        When ``fallback_route_ids`` is non-empty, delegates to
        :meth:`_dispatch_with_fallback` which chains through all alternatives
        before reporting failure.
        """
        if context.tool_registry is None:
            return NodeResult(success=False, error="No ToolRegistry configured for MCP dispatch")

        if node.route_id is None:
            logger.warning("Dispatch: no route_id on mcp atom, instruction=%r", node.instruction)
            return NodeResult(
                success=False,
                error=f"No route_id specified for mcp atom: {node.instruction!r}",
            )

        # Delegate to fallback chain when alternatives are declared
        if node.fallback_route_ids:
            return await self._dispatch_with_fallback(node, context)

        resolved = _resolve_route(node.route_id, context.tool_registry)
        if resolved is None:
            logger.warning("Dispatch: MCP route unavailable route_id=%s", node.route_id)
            return NodeResult(
                success=False,
                error=f"MCP route unavailable: {node.route_id} (tool not in registry)",
            )

        tool_name, param_key = resolved
        logger.info(
            "Dispatch: planned_route=%s resolved_route=%s tool=%s",
            node.route_id, node.route_id, tool_name,
        )
        if node.params is not None:
            params = dict(node.params)
        else:
            # MCP routes always have param_key="input"; fall back to instruction-based dispatch
            params = {param_key: node.instruction}
        output = await context.tool_registry.execute(tool_name, params)
        output_str = str(output) if output is not None else ""
        if _is_error_output(output_str):
            return NodeResult(success=False, error=output_str, output=output_str)
        return NodeResult(success=True, output=output_str)

    # ------------------------------------------------------------------
    # Replan helpers
    # ------------------------------------------------------------------

    async def _try_replan_and(
        self,
        node: Any,
        failed_index: int,
        failed_child: Any,
        context: RouterContext,
        all_outputs: list[str],
        all_traces: list[str],
    ) -> NodeResult | None:
        """Attempt to replan after an AND-child failure.

        Returns the result of executing the new plan, or *None* if replanning
        itself raised an exception.
        """
        self._replan_count += 1
        remaining = [
            c.instruction for c in node.children[failed_index + 1:] if str(c.node_type).lower() == "atom"
        ]
        failed_instruction = (
            failed_child.instruction
            if str(failed_child.node_type).lower() == "atom"
            else str(failed_child.to_dict())
        )
        try:
            new_plan = await self._planner.replan(
                context.task,
                completed=context.completed,
                failed=failed_instruction,
                remaining=remaining,
            )
            return await self.execute(new_plan, context)
        except Exception as exc:
            logger.warning("Replanning failed: %s", exc)
            return None
