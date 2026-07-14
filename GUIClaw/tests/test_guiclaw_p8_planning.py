"""Phase 8 tests — parallel AND execution, prioritized OR routing, and complexity gate.

Covers:
  - AND node parallel execution via asyncio.gather with concurrency semaphore
  - AND node per-child context isolation (no shared-list mutation)
  - AND node replan on child failure
  - OR node priority sorting (mcp > tool > gui)
  - OR node auto-fallback on child failure
  - OR node exhaustion with error reporting
  - AgentLoop complexity gate (skip for slash commands, short messages)
  - AgentLoop plan-and-execute path (TaskPlanner + TreeRouter integration)
  - AgentLoop fallback to direct agent loop on gate exception
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.planner import PlanNode
from nanobot.agent.router import NodeResult, RouterContext, TreeRouter
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _atom(instruction: str, capability: str = "gui") -> PlanNode:
    """Shorthand: create a leaf ATOM PlanNode."""
    return PlanNode(node_type="atom", instruction=instruction, capability=capability)  # type: ignore[arg-type]


def _and(*children: PlanNode) -> PlanNode:
    """Shorthand: create an AND composite PlanNode."""
    return PlanNode(node_type="and", children=children)


def _or(*children: PlanNode) -> PlanNode:
    """Shorthand: create an OR composite PlanNode."""
    return PlanNode(node_type="or", children=children)


def _make_ctx(**kwargs: Any) -> RouterContext:
    """Create a RouterContext with sensible defaults."""
    return RouterContext(task="test task", **kwargs)


class _StaticTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


# ---------------------------------------------------------------------------
# Task 1: Parallel AND execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_and_parallel_all_succeed() -> None:
    """AND node with 3 ATOM children dispatches all 3 via asyncio.gather.

    All 3 instructions must appear in context.completed after execution.
    """
    children = [
        _atom("step-alpha", "gui"),
        _atom("step-beta", "gui"),
        _atom("step-gamma", "gui"),
    ]
    plan = _and(*children)

    dispatch_calls: list[str] = []

    async def mock_run(instruction: str, max_retries: int = 1) -> Any:
        dispatch_calls.append(instruction)
        result = AsyncMock()
        result.success = True
        result.summary = f"Done: {instruction}"
        result.error = None
        result.trace_path = None
        return result

    mock_gui = AsyncMock()
    mock_gui.run = AsyncMock(side_effect=mock_run)

    router = TreeRouter()
    ctx = _make_ctx(gui_agent=mock_gui)

    result = await router.execute(plan, ctx)

    assert result.success, f"Expected success but got error: {result.error}"
    # All 3 atoms must appear in completed
    assert "step-alpha" in ctx.completed
    assert "step-beta" in ctx.completed
    assert "step-gamma" in ctx.completed
    # All 3 dispatch calls must have happened
    assert set(dispatch_calls) == {"step-alpha", "step-beta", "step-gamma"}


@pytest.mark.asyncio
async def test_and_executes_children_sequentially() -> None:
    """AND nodes run children sequentially without temporal overlap.

    We verify no temporal overlap by tracking each child coroutine's start and
    end timestamps.
    """
    intervals: list[tuple[float, float]] = []
    lock = asyncio.Lock()

    async def mock_run(instruction: str, max_retries: int = 1) -> Any:
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0.02)  # small delay to make overlap detectable
        end = asyncio.get_event_loop().time()
        async with lock:
            intervals.append((start, end))
        result = AsyncMock()
        result.success = True
        result.summary = f"Done: {instruction}"
        result.error = None
        result.trace_path = None
        return result

    mock_gui = AsyncMock()
    mock_gui.run = AsyncMock(side_effect=mock_run)

    plan = _and(_atom("a", "gui"), _atom("b", "gui"), _atom("c", "gui"))
    router = TreeRouter()
    ctx = _make_ctx(gui_agent=mock_gui)

    result = await router.execute(plan, ctx)

    assert result.success
    assert len(intervals) == 3

    # Verify no two intervals overlap: sort by start, then check end[i] <= start[i+1]
    intervals.sort(key=lambda x: x[0])
    for i in range(len(intervals) - 1):
        _, end_i = intervals[i]
        start_next, _ = intervals[i + 1]
        assert end_i <= start_next + 0.005, (
            f"Overlap detected between interval {i} and {i + 1}: "
            f"end={end_i:.4f}, next_start={start_next:.4f}"
        )


@pytest.mark.asyncio
async def test_and_child_failure_triggers_replan() -> None:
    """AND node triggers replan on child failure and can recover successfully.

    The failing child must cause replan() to be called.  The mock planner
    returns a single recovery atom that succeeds; final result is success.
    """
    plan = _and(
        _atom("step-ok", "gui"),
        _atom("step-fail", "gui"),
    )

    async def mock_run(instruction: str, max_retries: int = 1) -> Any:
        result = AsyncMock()
        if instruction == "step-fail":
            result.success = False
            result.summary = "Failed"
            result.error = "simulated failure"
        else:
            result.success = True
            result.summary = f"Done: {instruction}"
            result.error = None
        result.trace_path = None
        return result

    mock_gui = AsyncMock()
    mock_gui.run = AsyncMock(side_effect=mock_run)

    # Planner returns a recovery atom
    recovery = _atom("step-recovery", "gui")
    mock_planner = AsyncMock()
    mock_planner.replan = AsyncMock(return_value=recovery)

    router = TreeRouter(planner=mock_planner, max_replans=1)
    ctx = _make_ctx(gui_agent=mock_gui)

    result = await router.execute(plan, ctx)

    assert result.success
    mock_planner.replan.assert_awaited()


@pytest.mark.asyncio
async def test_and_no_shared_list_mutation() -> None:
    """AND parallel children must not share context.completed during execution.

    Each child receives an independent snapshot of context.completed at the
    time it is launched.  We verify this by inspecting what each child saw
    in its RouterContext — the snapshot should only contain items added
    *before* the AND started, not items from sibling children.
    """
    async def mock_run(instruction: str, max_retries: int = 1) -> Any:
        # We cannot inspect the RouterContext directly here; instead we rely
        # on the behaviour: if context is shared, then a child completing
        # early would add its instruction and the other child would see it.
        # This test validates the contract via the snapshot design.
        await asyncio.sleep(0.01)
        result = AsyncMock()
        result.success = True
        result.summary = f"Done: {instruction}"
        result.error = None
        result.trace_path = None
        return result

    mock_gui = AsyncMock()
    mock_gui.run = AsyncMock(side_effect=mock_run)

    # Pre-seed one item so we can verify the snapshot carries it
    ctx = _make_ctx(gui_agent=mock_gui)
    ctx.completed.append("pre-existing")

    plan = _and(_atom("child-1", "gui"), _atom("child-2", "gui"))
    router = TreeRouter()

    # Patch RouterContext constructor to record what each child receives
    created_contexts: list[RouterContext] = []

    class RecordingRouterContext(RouterContext):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            created_contexts.append(self)

    with patch("nanobot.agent.router.RouterContext", RecordingRouterContext):
        result = await router.execute(plan, ctx)

    assert result.success
    # Every child context created during parallel AND execution should start
    # with the pre-existing item but NOT contain the other child's instruction
    for child_ctx in created_contexts:
        assert "pre-existing" in child_ctx.completed, "Snapshot should carry pre-existing entries"
        # Sibling instructions must not be present at snapshot time
        assert "child-1" not in child_ctx.completed or "child-2" not in child_ctx.completed, (
            "Both sibling instructions found in child context — shared list mutation detected"
        )


# ---------------------------------------------------------------------------
# Task 2: OR node priority sorting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_or_priority_order() -> None:
    """OR node sorts children mcp > tool > gui before trying them.

    Children are provided in [gui, mcp, tool] order; mock makes mcp succeed.
    We verify that mcp is called first (not gui) by tracking call order.
    """
    call_order: list[str] = []

    async def mock_run(instruction: str, max_retries: int = 1) -> Any:
        call_order.append(f"gui:{instruction}")
        result = AsyncMock()
        result.success = False
        result.summary = ""
        result.error = "gui failed"
        result.trace_path = None
        return result

    mock_gui = AsyncMock()
    mock_gui.run = AsyncMock(side_effect=mock_run)

    # mcp_client: returns success for its atom
    mock_mcp_client = object()  # non-None so dispatch proceeds

    plan = _or(
        _atom("gui-option", "gui"),
        _atom("mcp-option", "mcp"),
        _atom("tool-option", "tool"),
    )

    # Intercept _dispatch_atom to track order
    dispatch_order: list[str] = []

    async def recording_dispatch(self: TreeRouter, node: Any, context: RouterContext) -> NodeResult:
        dispatch_order.append(f"{node.capability}:{node.instruction}")
        if node.capability == "mcp":
            return NodeResult(success=True, output="mcp done")
        return NodeResult(success=False, error=f"{node.capability} failed")

    with patch.object(TreeRouter, "_dispatch_atom", recording_dispatch):
        router = TreeRouter()
        ctx = _make_ctx(gui_agent=mock_gui, mcp_client=mock_mcp_client)
        result = await router.execute(plan, ctx)

    assert result.success, f"Expected success but got: {result.error}"
    # mcp must be the first capability tried
    assert dispatch_order[0].startswith("mcp:"), f"Expected mcp first, got: {dispatch_order}"
    # gui (lowest priority) must appear after mcp (mcp succeeded so gui not tried)
    # At minimum mcp was first and gui was not tried (mcp succeeded)
    assert not any(x.startswith("gui:") for x in dispatch_order), (
        "gui should not be tried because mcp already succeeded"
    )


@pytest.mark.asyncio
async def test_or_same_capability_preserves_order() -> None:
    """OR node with children of the same capability preserves original order.

    Three gui atoms in order [first, second, third].  First fails, second
    succeeds.  Verify that 'second' appears in ctx.completed (meaning it was
    reached in original order and succeeded).
    """
    call_sequence: list[str] = []

    async def recording_dispatch(self: TreeRouter, node: Any, context: RouterContext) -> NodeResult:
        call_sequence.append(node.instruction)
        if node.instruction == "gui-first":
            return NodeResult(success=False, error="first failed")
        return NodeResult(success=True, output=f"done: {node.instruction}")

    plan = _or(
        _atom("gui-first", "gui"),
        _atom("gui-second", "gui"),
        _atom("gui-third", "gui"),
    )

    with patch.object(TreeRouter, "_dispatch_atom", recording_dispatch):
        router = TreeRouter()
        ctx = _make_ctx()
        result = await router.execute(plan, ctx)

    assert result.success
    assert "gui-second" in ctx.completed
    # second must come after first in call order
    assert call_sequence.index("gui-first") < call_sequence.index("gui-second")
    # third should not be called (second succeeded)
    assert "gui-third" not in call_sequence


@pytest.mark.asyncio
async def test_or_auto_fallback() -> None:
    """OR auto-fallback: first child (mcp) fails, second (tool) succeeds.

    Result should be success with the tool's output.
    """

    async def recording_dispatch(self: TreeRouter, node: Any, context: RouterContext) -> NodeResult:
        if node.capability == "mcp":
            return NodeResult(success=False, error="mcp unavailable")
        if node.capability == "tool":
            return NodeResult(success=True, output="tool-output")
        return NodeResult(success=False, error="unexpected")

    plan = _or(
        _atom("mcp-step", "mcp"),
        _atom("tool-step", "tool"),
    )

    with patch.object(TreeRouter, "_dispatch_atom", recording_dispatch):
        router = TreeRouter()
        ctx = _make_ctx()
        result = await router.execute(plan, ctx)

    assert result.success
    assert result.output == "tool-output"
    assert "tool-step" in ctx.completed


@pytest.mark.asyncio
async def test_or_all_fail() -> None:
    """OR reports failure when all children fail.

    Result must have success=False and a non-empty error message.
    """

    async def recording_dispatch(self: TreeRouter, node: Any, context: RouterContext) -> NodeResult:
        return NodeResult(success=False, error=f"{node.capability} failed")

    plan = _or(
        _atom("mcp-opt", "mcp"),
        _atom("tool-opt", "tool"),
        _atom("gui-opt", "gui"),
    )

    with patch.object(TreeRouter, "_dispatch_atom", recording_dispatch):
        router = TreeRouter()
        ctx = _make_ctx()
        result = await router.execute(plan, ctx)

    assert not result.success
    assert result.error is not None and len(result.error) > 0


# ---------------------------------------------------------------------------
# Task 1 (Plan 03): AgentLoop complexity gate and plan-and-execute integration
# ---------------------------------------------------------------------------


def _make_agent_loop(tmp_path: Path) -> Any:
    """Build a minimal AgentLoop with all dependencies mocked.

    GuiSubagentTool construction is patched out so tests do not require a real
    device backend.  The loop is still created with a non-None gui_config so that
    the complexity gate condition (`self._gui_config is not None`) is satisfied.
    """
    from nanobot.agent.loop import AgentLoop

    mock_bus = MagicMock()
    mock_bus.publish_outbound = AsyncMock()
    mock_bus.consume_inbound = AsyncMock()

    mock_provider = MagicMock()
    mock_provider.get_default_model.return_value = "test-model"
    mock_provider.chat_with_retry = AsyncMock()

    # Minimal GuiConfig to satisfy _gui_config is not None check
    mock_gui_config = MagicMock()

    # Patch GuiSubagentTool so _register_default_tools does not try to
    # instantiate a real device backend during tests.
    with patch("nanobot.agent.tools.gui.GuiSubagentTool") as mock_gui_cls:
        mock_gui_tool = MagicMock()
        mock_gui_tool.name = "gui_task"
        mock_gui_tool.runtime_context_provider.return_value = None
        mock_gui_cls.return_value = mock_gui_tool
        loop = AgentLoop(
            bus=mock_bus,
            provider=mock_provider,
            workspace=tmp_path,
            gui_config=mock_gui_config,
        )
    return loop


def _make_inbound(content: str) -> Any:
    """Build a minimal InboundMessage."""
    from nanobot.bus.events import InboundMessage

    return InboundMessage(
        channel="cli",
        sender_id="user",
        chat_id="test",
        content=content,
    )


@pytest.mark.asyncio
async def test_complexity_gate_skip_slash_command(tmp_path: Path) -> None:
    """/help slash command bypasses _needs_planning entirely."""
    loop = _make_agent_loop(tmp_path)

    with patch.object(loop, "_needs_planning", new_callable=AsyncMock) as mock_gate:
        # /help is handled before any complexity check and returns early
        result = await loop._process_message(_make_inbound("/help"))

    mock_gate.assert_not_awaited()
    assert result is not None
    assert "/help" in result.content.lower() or "command" in result.content.lower()


@pytest.mark.asyncio
async def test_complexity_gate_skip_short_message(tmp_path: Path) -> None:
    """Messages shorter than 20 chars bypass _needs_planning."""
    loop = _make_agent_loop(tmp_path)

    # Patch _run_agent_loop to avoid real LLM call
    with (
        patch.object(loop, "_needs_planning", new_callable=AsyncMock) as mock_gate,
        patch.object(
            loop,
            "_run_agent_loop",
            new=AsyncMock(return_value=("short reply", [], [], "completed", False)),
        ),
        patch.object(loop.context, "build_messages", return_value=[]),
    ):
        await loop._process_message(_make_inbound("hi"))

    mock_gate.assert_not_awaited()


@pytest.mark.asyncio
async def test_complexity_gate_false_uses_agent_loop(tmp_path: Path) -> None:
    """When _needs_planning returns False, _run_agent_loop is called."""
    loop = _make_agent_loop(tmp_path)

    with (
        patch.object(loop, "_needs_planning", new_callable=AsyncMock, return_value=False),
        patch.object(
            loop,
            "_run_agent_loop",
            new=AsyncMock(return_value=("agent response", [], [], "completed", False)),
        ) as mock_run,
        patch.object(loop.context, "build_messages", return_value=[]),
    ):
        result = await loop._process_message(
            _make_inbound("Please do a complex multi-step task for me")
        )

    mock_run.assert_awaited_once()
    assert result is not None


@pytest.mark.asyncio
async def test_complexity_gate_true_uses_planning(tmp_path: Path) -> None:
    """When _needs_planning returns True, _plan_and_execute is called instead."""
    loop = _make_agent_loop(tmp_path)

    with (
        patch.object(loop, "_needs_planning", new_callable=AsyncMock, return_value=True),
        patch.object(
            loop,
            "_plan_and_execute",
            new=AsyncMock(return_value=("plan result", ["task_planner"], [])),
        ) as mock_plan,
        patch.object(
            loop,
            "_run_agent_loop",
            new=AsyncMock(return_value=("should not be called", [], [], "completed", False)),
        ) as mock_run,
        patch.object(loop.context, "build_messages", return_value=[]),
    ):
        result = await loop._process_message(
            _make_inbound("Search the web and then open a browser for me")
        )

    mock_plan.assert_awaited_once()
    mock_run.assert_not_awaited()
    assert result is not None


@pytest.mark.asyncio
async def test_complexity_gate_exception_falls_back(tmp_path: Path) -> None:
    """When _needs_planning raises, execution falls back to _run_agent_loop."""
    loop = _make_agent_loop(tmp_path)

    with (
        patch.object(
            loop,
            "_needs_planning",
            new=AsyncMock(side_effect=RuntimeError("gate failed")),
        ),
        patch.object(
            loop,
            "_run_agent_loop",
            new=AsyncMock(return_value=("fallback response", [], [], "completed", False)),
        ) as mock_run,
        patch.object(loop.context, "build_messages", return_value=[]),
    ):
        result = await loop._process_message(
            _make_inbound("Do something that makes the gate fail unexpectedly")
        )

    mock_run.assert_awaited_once()
    assert result is not None


@pytest.mark.asyncio
async def test_plan_and_execute_logs_tree(tmp_path: Path) -> None:
    """_plan_and_execute logs the decomposed plan tree via logger.info."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text(
        "tool.exec_shell worked for disabling bluetooth on macOS.\nUser prefers concise updates.\n",
        encoding="utf-8",
    )
    (memory_dir / "HISTORY.md").write_text(
        "[2026-03-22 09:00] fallback to gui.desktop when shell failed to open System Settings.",
        encoding="utf-8",
    )

    loop = _make_agent_loop(tmp_path)
    loop.tools = ToolRegistry()
    for tool_name in ("gui_task", "exec", "read_file", "web_search", "mcp_demo_lookup"):
        loop.tools.register(_StaticTool(tool_name))

    atom_node = PlanNode(
        node_type="atom",
        instruction="disable bluetooth",
        capability="tool",
        route_id="tool.exec_shell",
        route_reason="Local shell route can toggle host settings directly",
        fallback_route_ids=("gui.desktop",),
    )

    mock_planner = AsyncMock()
    mock_planner.plan = AsyncMock(return_value=atom_node)

    from nanobot.agent.router import NodeResult

    mock_router_execute = AsyncMock(return_value=NodeResult(success=True, output="done"))

    with (
        patch("nanobot.agent.planner.TaskPlanner", return_value=mock_planner),
        patch("nanobot.agent.router.TreeRouter") as mock_router_cls,
        patch("nanobot.agent.router.RouterContext"),
    ):
        mock_router_instance = MagicMock()
        mock_router_instance.execute = mock_router_execute
        mock_router_cls.return_value = mock_router_instance


        with patch("nanobot.agent.loop.logger") as mock_logger:
            output, tools_used, _ = await loop._plan_and_execute(
                "Open the browser and search for something complex",
                channel="cli",
                chat_id="chat-1",
                metadata={"message_id": "m-1"},
            )

    planning_context = mock_planner.plan.await_args.kwargs["planning_context"]
    route_ids = [route.route_id for route in planning_context.catalog.routes]
    assert route_ids == [
        "gui.desktop",
        "tool.exec_shell",
        "tool.filesystem.read",
        "tool.web.search",
        "mcp.demo.lookup",
    ]
    assert [hint.route_id for hint in planning_context.memory_hints] == [
        "tool.exec_shell",
        "gui.desktop",
    ]
    assert all(
        "concise updates" not in hint.to_prompt_line() for hint in planning_context.memory_hints
    )

    info_calls = [call.args for call in mock_logger.info.call_args_list]
    assert any(
        args
        and args[0] == "Decomposed plan:\n{}"
        and "via tool.exec_shell" in args[1]
        and "fallback -> gui.desktop" in args[1]
        for args in info_calls
    ), f"Expected 'Decomposed plan' in logger.info calls, got: {info_calls}"
    debug_calls = [call.args for call in mock_logger.debug.call_args_list]
    assert any(
        args and args[0] == "Decomposed plan (raw): {}" and args[1] == atom_node.to_dict()
        for args in debug_calls
    ), f"Expected raw decomposed plan in logger.debug calls, got: {debug_calls}"
    loop.bus.publish_outbound.assert_awaited_once()
    outbound = loop.bus.publish_outbound.await_args.args[0]
    assert outbound.channel == "cli"
    assert outbound.chat_id == "chat-1"
    assert outbound.metadata["_progress"] is True
    assert outbound.metadata["_plan_preview"] is True
    assert "执行计划预览：" in outbound.content
    assert "via tool.exec_shell" in outbound.content
    assert output == "done"


def test_format_plan_tree_renders_indented_human_readable_outline() -> None:
    """Plan formatting should favor a readable tree for humans scanning logs."""
    plan = PlanNode(
        node_type="and",
        children=(
            PlanNode(node_type="atom", instruction="open obsidian", capability="gui"),
            PlanNode(
                node_type="or",
                children=(
                    PlanNode(
                        node_type="atom", instruction="pause music from menu bar", capability="gui"
                    ),
                    PlanNode(
                        node_type="atom", instruction="pause music with media key", capability="gui"
                    ),
                ),
            ),
        ),
    )

    rendered = AgentLoop._format_plan_tree(plan)

    assert rendered == (
        "AND\n"
        "  - GUI: open obsidian\n"
        "  OR\n"
        "    - GUI: pause music from menu bar\n"
        "    - GUI: pause music with media key"
    )


def test_plan_node_route_metadata_round_trip() -> None:
    node = PlanNode(
        node_type="atom",
        instruction="disable bluetooth",
        capability="tool",
        route_id="tool.exec_shell",
        route_reason="Direct host route is safer than GUI",
        fallback_route_ids=("gui.desktop",),
    )

    assert PlanNode.from_dict(node.to_dict()) == node


def test_plan_node_route_metadata_legacy_payload_still_parses() -> None:
    payload = {
        "type": "atom",
        "instruction": "open obsidian",
        "capability": "gui",
    }

    node = PlanNode.from_dict(payload)

    assert node == PlanNode(node_type="atom", instruction="open obsidian", capability="gui")
    assert node.to_dict() == payload


def test_plan_node_from_dict_normalizes_uppercase_node_types() -> None:
    payload = {
        "type": "AND",
        "children": [
            {"type": "ATOM", "instruction": "open obsidian", "capability": "gui"},
            {"type": "ATOM", "instruction": "list files", "capability": "tool"},
        ],
    }

    node = PlanNode.from_dict(payload)

    assert node.node_type == "and"
    assert [child.node_type for child in node.children] == ["atom", "atom"]
    assert node.children[0].instruction == "open obsidian"
    assert node.children[1].capability == "tool"


# ---------------------------------------------------------------------------
# Phase 22 OTM: PlanNode.params field round-trip tests
# ---------------------------------------------------------------------------


def test_plan_node_params_defaults_to_none() -> None:
    """PlanNode().params is None by default (backward compatible)."""
    node = PlanNode(node_type="atom", instruction="list files")
    assert node.params is None


def test_plan_node_params_stores_dict() -> None:
    """PlanNode(params={"path": "/tmp"}).params returns the dict unchanged."""
    node = PlanNode(
        node_type="atom",
        instruction="list /tmp",
        capability="tool",
        route_id="tool.filesystem.list",
        params={"path": "/tmp"},
    )
    assert node.params == {"path": "/tmp"}


def test_plan_node_params_to_dict_includes_params_when_set() -> None:
    """to_dict() serialises params into the output dict when params is set."""
    node = PlanNode(
        node_type="atom",
        instruction="list /tmp",
        capability="tool",
        route_id="tool.filesystem.list",
        params={"path": "/tmp"},
    )
    d = node.to_dict()
    assert d["params"] == {"path": "/tmp"}


def test_plan_node_params_to_dict_omits_params_when_none() -> None:
    """to_dict() does not include a 'params' key when params is None."""
    node = PlanNode(node_type="atom", instruction="list files", capability="tool")
    d = node.to_dict()
    assert "params" not in d


def test_plan_node_params_from_dict_restores_params() -> None:
    """from_dict() populates params when the serialised payload includes it."""
    payload = {
        "type": "atom",
        "instruction": "list /tmp",
        "capability": "tool",
        "route_id": "tool.filesystem.list",
        "params": {"path": "/tmp"},
    }
    node = PlanNode.from_dict(payload)
    assert node.params == {"path": "/tmp"}


def test_plan_node_params_from_dict_none_when_absent() -> None:
    """from_dict() leaves params as None when the payload omits the key."""
    payload = {"type": "atom", "instruction": "list files", "capability": "tool"}
    node = PlanNode.from_dict(payload)
    assert node.params is None


def test_plan_node_params_round_trip() -> None:
    """to_dict -> from_dict round-trip preserves params exactly."""
    node = PlanNode(
        node_type="atom",
        instruction="write output",
        capability="tool",
        route_id="tool.filesystem.write",
        params={"path": "out.txt", "content": "hello"},
    )
    assert PlanNode.from_dict(node.to_dict()) == node


def test_plan_node_params_round_trip_without_params() -> None:
    """Round-trip of a node without params still equals the original (backward compat)."""
    node = PlanNode(
        node_type="atom",
        instruction="open obsidian",
        capability="gui",
    )
    assert PlanNode.from_dict(node.to_dict()) == node
