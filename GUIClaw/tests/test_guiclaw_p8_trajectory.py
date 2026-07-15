"""Phase 8 tests: TrajectorySummarizer wiring and nanobot.agent public API exports."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared helpers (reuse _MockNanobotProvider pattern from test_guiclaw_p3_nanobot)
# ---------------------------------------------------------------------------

try:
    from nanobot.config.schema import Config
    from nanobot.providers.base import LLMProvider as NanobotLLMProvider
    from nanobot.providers.base import LLMResponse as NanobotLLMResponse
    from nanobot.providers.base import ToolCallRequest
except Exception as exc:  # pragma: no cover
    Config = None
    NanobotLLMProvider = object
    NanobotLLMResponse = None
    ToolCallRequest = None
    _NANOBOT_IMPORT_ERROR: Exception | None = exc
else:
    _NANOBOT_IMPORT_ERROR = None


def _nanobot_tool_response(
    *,
    content: str,
    arguments: dict[str, Any],
    call_id: str,
) -> Any:
    return NanobotLLMResponse(
        content=_profile_content(content, arguments),
        tool_calls=[
            ToolCallRequest(
                id=call_id,
                name="computer_use",
                arguments=arguments,
            )
        ],
    )


def _profile_content(thought: str, arguments: dict[str, Any]) -> str:
    action_type = arguments.get("action_type")
    if action_type == "done":
        goal_status = "complete" if arguments.get("status", "success") == "success" else "failed"
        action = {"action_type": "status", "goal_status": goal_status}
    elif action_type == "wait":
        action = {"action_type": "wait"}
    elif action_type == "input_text":
        action = {"action_type": "input_text", "text": arguments.get("text", "")}
    elif action_type == "request_intervention":
        action = {"action_type": "request_intervention", "text": arguments.get("text", "")}
    else:
        action = arguments
    return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}"


if _NANOBOT_IMPORT_ERROR is None:

    class _MockNanobotProvider(NanobotLLMProvider):
        """Minimal scripted nanobot LLM provider for tests."""

        def __init__(self, responses: list[Any]) -> None:
            super().__init__(api_key="test-key")
            self._responses = list(responses)
            self.calls: list[dict[str, Any]] = []

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            reasoning_effort: str | None = None,
            tool_choice: str | dict[str, Any] | None = None,
        ) -> Any:
            return await self.chat_with_retry(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )

        async def chat_with_retry(self, messages, tools=None, model=None, **kwargs) -> Any:
            self.calls.append({"messages": messages, "tools": tools, "model": model, **kwargs})
            if not self._responses:
                raise AssertionError("No scripted nanobot responses left")
            return self._responses.pop(0)

        def get_default_model(self) -> str:
            return "test-model"

else:

    class _MockNanobotProvider:
        def __init__(self, responses: list[Any]) -> None:
            raise RuntimeError("nanobot imports unavailable") from _NANOBOT_IMPORT_ERROR


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    (tmp_path / "gui_runs").mkdir()
    (tmp_path / "gui_skills").mkdir()
    return tmp_path


def test_gui_subagent_run_storage_is_isolated_from_user_home(tmp_path: Path) -> None:
    from guiclaw.paths import resolve_guiclaw_data_dir

    assert resolve_guiclaw_data_dir("gui_runs") == tmp_path / ".guiclaw" / "gui_runs"


def _dry_run_tool(
    tmp_workspace: Path,
    extra_responses: list[Any] | None = None,
    gui_overrides: dict[str, Any] | None = None,
) -> Any:
    """Build a GuiSubagentTool with dry-run backend and two standard action responses."""
    from nanobot.agent.tools.gui import GuiSubagentTool

    responses = [
        NanobotLLMResponse(content='{"mode":"single","subtasks":[]}'),
        _nanobot_tool_response(
            content="Action: wait",
            arguments={"action_type": "wait", "duration_ms": 1},
            call_id="tc_wait",
        ),
        _nanobot_tool_response(
            content="Action: done",
            arguments={"action_type": "done", "status": "success"},
            call_id="tc_done",
        ),
    ]
    if extra_responses:
        responses.extend(extra_responses)

    provider = _MockNanobotProvider(responses)
    gui_config = {"backend": "dry-run"}
    if gui_overrides:
        gui_config.update(gui_overrides)
    return GuiSubagentTool(
        gui_config=Config(gui=gui_config).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )


# ---------------------------------------------------------------------------
# Task 1 tests: TrajectorySummarizer wiring into GuiSubagentTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarizer_called_post_run(
    tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """summarize_file should be awaited with a Path after a successful GUI run."""
    from nanobot.agent.tools.gui import GuiSubagentTool  # noqa: F401 — import check

    tool = _dry_run_tool(tmp_workspace)
    promote_mock = AsyncMock(return_value=None)
    with (
        patch(
            "guiclaw.trajectory.summarizer.TrajectorySummarizer.summarize_file",
            new_callable=AsyncMock,
            return_value="Summary text",
        ) as mock_summarize,
        patch.object(tool._postprocessor, "_extract_skill", new=promote_mock),
    ):
        await tool.execute(task="test task")
        await tool._wait_for_pending_postprocessing()

    mock_summarize.assert_awaited_once()
    promote_mock.assert_awaited_once()
    call_arg = mock_summarize.call_args[0][0]  # first positional arg
    assert isinstance(call_arg, Path), f"Expected Path, got {type(call_arg)}"


@pytest.mark.asyncio
async def test_summarizer_failure_non_fatal(
    tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If summarize_file raises, execute() should still return valid JSON and promotion still runs."""

    tool = _dry_run_tool(tmp_workspace)
    promote_mock = AsyncMock(return_value=None)
    with (
        patch(
            "guiclaw.trajectory.summarizer.TrajectorySummarizer.summarize_file",
            new_callable=AsyncMock,
            side_effect=RuntimeError("summarizer exploded"),
        ),
        patch.object(tool._postprocessor, "_extract_skill", new=promote_mock),
    ):
        raw = await tool.execute(task="test task")
        await tool._wait_for_pending_postprocessing()

    result = json.loads(raw)
    assert "success" in result, "execute() must return JSON with 'success' key"
    promote_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_gui_evaluation_runs_from_background_postprocessing(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_mock = AsyncMock(return_value={"success": True, "reason": "final state matches"})
    monkeypatch.setattr("guiclaw.evaluation.evaluate_gui_trajectory", eval_mock)

    tool = _dry_run_tool(
        tmp_workspace,
        gui_overrides={
            "evaluation": {
                "enabled": True,
                "judgeModel": "judge-model",
                "apiKey": "judge-key",
                "apiBase": "https://judge.example/v1",
            }
        },
    )
    promote_mock = AsyncMock(return_value=None)

    with patch.object(tool._postprocessor, "_extract_skill", new=promote_mock):
        await tool.execute(task="test task")
        await tool._wait_for_pending_postprocessing()

    eval_mock.assert_awaited_once()
    promote_mock.assert_awaited_once()
    kwargs = eval_mock.await_args.kwargs
    assert kwargs["instruction"] == "test task"
    assert kwargs["model"] == "judge-model"
    assert kwargs["api_key"] == "judge-key"
    assert kwargs["api_base"] == "https://judge.example/v1"
    assert isinstance(kwargs["trace_path"], Path)


@pytest.mark.asyncio
async def test_gui_tool_returns_post_run_state_from_latest_trace_step(
    tmp_workspace: Path,
) -> None:
    tool = _dry_run_tool(tmp_workspace)

    raw = await tool.execute(task="test task")
    await tool._wait_for_pending_postprocessing()

    result = json.loads(raw)
    post_run_state = result["post_run_state"]

    assert post_run_state["trace_read"] is True
    assert post_run_state["completion_assessment"] == "completed"
    assert post_run_state["latest_screenshot_path"] is not None
    assert post_run_state["last_action"]["action_type"] == "done"
    assert post_run_state["screen_resolution"] is None
    assert post_run_state["last_foreground_app"] == "DryRun"
    assert "Current: DryRun" in post_run_state["current_state"]


@pytest.mark.asyncio
async def test_gui_tool_returns_before_background_postprocessing_finishes_with_promotion(
    tmp_workspace: Path,
) -> None:
    tool = _dry_run_tool(tmp_workspace)
    release_postprocess = asyncio.Event()
    postprocess_started = asyncio.Event()
    captured: dict[str, Any] = {}

    async def fake_run_all(
        self_inner,
        trace_path: Path,
        *,
        is_success: bool,
        platform: str,
        task: str,
        subtask_index: int,
    ) -> None:
        captured["trace_path"] = trace_path
        captured["is_success"] = is_success
        captured["platform"] = platform
        captured["task"] = task
        captured["subtask_index"] = subtask_index
        postprocess_started.set()
        await release_postprocess.wait()

    with patch.object(type(tool._postprocessor), "_run_all", new=fake_run_all):
        raw = await tool.execute(task="test task")
        await asyncio.wait_for(postprocess_started.wait(), timeout=1.0)
        result = json.loads(raw)
        assert result["success"] is True
        assert captured["is_success"] is True
        assert captured["platform"] == "dry-run"
        assert captured["task"] == "test task"
        assert tool._postprocessor._pending
        release_postprocess.set()
        await tool._wait_for_pending_postprocessing()


@pytest.mark.asyncio
async def test_gui_evaluation_failure_is_non_fatal(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guiclaw.evaluation.evaluate_gui_trajectory",
        AsyncMock(side_effect=RuntimeError("judge unavailable")),
    )

    tool = _dry_run_tool(
        tmp_workspace,
        gui_overrides={
            "evaluation": {
                "enabled": True,
                "judgeModel": "judge-model",
                "apiKey": "judge-key",
            }
        },
    )
    promote_mock = AsyncMock(return_value=None)

    with patch.object(tool._postprocessor, "_extract_skill", new=promote_mock):
        raw = await tool.execute(task="test task")
        await tool._wait_for_pending_postprocessing()

    result = json.loads(raw)
    assert result["success"] is True
    promote_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_summarizer_skipped_when_no_trace(
    tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If trace_path is None, summarize_file must NOT be called."""
    from guiclaw.agent import AgentResult

    async def fake_run(self_inner, task: str, *, max_retries: int = 3, app_hint: str | None = None):
        return AgentResult(
            success=False,
            summary="no trace",
            trace_path=None,
            steps_taken=0,
            error="dry run no trace",
        )

    monkeypatch.setattr("guiclaw.agent.GuiAgent.run", fake_run)
    tool = _dry_run_tool(tmp_workspace)
    promote_mock = AsyncMock(return_value=None)

    # recorder.path will also be None because agent.run() didn't write anything
    # and we need to ensure the recorder itself returns None for its path.
    # Patch recorder.path to be None via monkeypatching the recorder class.
    from guiclaw.trajectory import recorder as rec_module

    monkeypatch.setattr(
        rec_module.TrajectoryRecorder,
        "path",
        property(lambda self: None),
    )

    with (
        patch(
            "guiclaw.trajectory.summarizer.TrajectorySummarizer.summarize_file",
            new_callable=AsyncMock,
        ) as mock_summarize,
        patch.object(tool._postprocessor, "_extract_skill", new=promote_mock),
    ):
        await tool.execute(task="test task")
        await tool._wait_for_pending_postprocessing()

    mock_summarize.assert_not_awaited()
    promote_mock.assert_not_awaited()


def test_evaluate_gui_trajectory_counts_only_step_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.utils.gui_evaluation import evaluate_gui_trajectory_sync

    trace_path = tmp_path / "traj.json"
    trace_path.write_text(
        json.dumps(
            {
                "instruction": "test instruction",
                "platform": "android",
                "subtasks": [{"subtask": 1, "task": "test instruction"}],
                "steps": [
                    {
                        "step": 1,
                        "subtask": 1,
                        "attempt": 1,
                        "phase": "agent",
                        "model_output": "tap",
                        "action": {"action_type": "tap"},
                        "screenshot": "step_001.png",
                    },
                    {
                        "step": 2,
                        "subtask": 1,
                        "attempt": 1,
                        "phase": "agent",
                        "model_output": "done",
                        "action": {"action_type": "done"},
                        "screenshot": "step_002.png",
                    },
                ],
                "screenshots": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for name in ("meta.png", "step_001.png", "step_002.png"):
        (tmp_path / name).write_bytes(b"png")

    captured: dict[str, Any] = {}

    def fake_judge_success(*, client, instruction, traj_rows, screenshots, model, task_id):
        captured["traj_rows"] = traj_rows
        captured["screenshots"] = screenshots
        return True, "ok"

    monkeypatch.setattr("guiclaw.evaluation.judge_success", fake_judge_success)
    monkeypatch.setattr("guiclaw.evaluation.OpenAI", lambda **kwargs: object())

    result = evaluate_gui_trajectory_sync(
        instruction="test instruction",
        trace_path=trace_path,
        model="judge-model",
        api_key="judge-key",
        api_base="https://judge.example/v1",
        task_id="task-1",
        output_path=None,
    )

    assert result["steps"] == 2
    assert [row.get("type") for row in captured["traj_rows"]] == ["step", "step"]
    assert len(captured["screenshots"]) == 2


def test_load_screenshots_for_judge_supports_screenshot_path_field(tmp_path: Path) -> None:
    from nanobot.utils.gui_evaluation import load_screenshots_for_judge

    trace_path = tmp_path / "traj.json"
    screenshot_path = tmp_path / "screenshots" / "001_tap.png"
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.write_bytes(b"png-bytes")
    trace_path.write_text("", encoding="utf-8")

    screenshots = load_screenshots_for_judge(
        trace_path,
        [
            {
                "type": "step",
                "step_index": 1,
                "screenshot_path": str(screenshot_path),
            }
        ],
    )

    assert screenshots == [b"png-bytes"]
