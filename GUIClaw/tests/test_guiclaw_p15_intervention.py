from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from guiclaw.action import ActionError, parse_action
from guiclaw.agent import GuiAgent
from guiclaw.agents.utils.prompts import GENERAL_E2E_PROMPT_TEMPLATE
from guiclaw.backends.dry_run import _TINY_PNG, DryRunBackend
from guiclaw.interfaces import LLMResponse, ToolCall
from guiclaw.observation import Observation
from guiclaw.tool_schemas import COMPUTER_USE_TOOL
from guiclaw.trajectory.recorder import TrajectoryRecorder


def _make_recorder(tmp_path: Path, task: str = "test task") -> TrajectoryRecorder:
    return TrajectoryRecorder(output_dir=tmp_path / "traj", task=task, platform="linux")


class _RecordingLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    async def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        model=None,
        max_tokens=None,
    ) -> LLMResponse:
        del tools, tool_choice, model, max_tokens
        self.calls.append(json.loads(json.dumps(messages)))
        if not self._responses:
            raise AssertionError("No scripted responses left.")
        return _profile_response(self._responses.pop(0))


def _profile_response(response: LLMResponse) -> LLMResponse:
    if not response.tool_calls:
        return response
    action = dict(response.tool_calls[0].arguments)
    action_type = action.get("action_type")
    if action_type == "done":
        goal_status = "complete" if action.get("status", "success") == "success" else "failed"
        profile_action = {"action_type": "status", "goal_status": goal_status}
    elif action_type == "wait":
        profile_action = {"action_type": "wait"}
    elif action_type == "request_intervention":
        profile_action = {"action_type": "request_intervention", "text": action.get("text", "")}
    elif action_type == "input_text":
        profile_action = {"action_type": "input_text", "text": action.get("text", "")}
    else:
        profile_action = action
    return replace(
        response,
        content=f"Thought: {response.content}\nAction: {json.dumps(profile_action, ensure_ascii=False)}",
    )


class _BackendDouble:
    platform = "linux"

    def __init__(self, observations: list[dict[str, object]]) -> None:
        self._observations = list(observations)
        self.preflight = AsyncMock()
        self.execute = AsyncMock(return_value="backend execute should not run")
        self.list_apps = AsyncMock(return_value=[])
        self.observe = AsyncMock(side_effect=self._observe)

    async def _observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        if not self._observations:
            raise AssertionError("No scripted observations left.")
        payload = self._observations.pop(0)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(_TINY_PNG)
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=1280,
            screen_height=720,
            foreground_app=payload.get("foreground_app"),
            platform=self.platform,
            extra=dict(payload.get("extra", {})),
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_action_accepts_request_intervention() -> None:
    action = parse_action(
        {
            "action_type": "request_intervention",
            "text": "Need the user to complete a login challenge.",
        }
    )

    assert action.action_type == "request_intervention"
    assert action.text == "Need the user to complete a login challenge."


def test_request_intervention_requires_reason_text() -> None:
    with pytest.raises(ActionError, match="requires a non-empty 'text' field"):
        parse_action({"action_type": "request_intervention"})

    with pytest.raises(ActionError, match="requires a non-empty 'text' field"):
        parse_action(
            {
                "action_type": "request_intervention",
                "text": "   ",
            }
        )


def test_active_general_e2e_prompt_lists_ask_user_action() -> None:
    prompt = GENERAL_E2E_PROMPT_TEMPLATE.render(
        tools="",
        scale_factor=1000,
        extra_action_rows="",
        decision_rules="",
        compact_skill_instructions="",
    )

    assert "`ask_user`" in prompt
    assert '"action_type":"ask_user"' in prompt
    assert "no enough information" in prompt


def test_agent_tool_schema_lists_request_intervention() -> None:
    action_types = COMPUTER_USE_TOOL["function"]["parameters"]["properties"]["action_type"]["enum"]

    assert "request_intervention" in action_types


@pytest.mark.asyncio
@pytest.mark.pauses_backend_io
async def test_intervention_request_pauses_backend_execute_and_observe(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content="Action: request intervention for login",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={
                            "action_type": "request_intervention",
                            "text": "Need the user to finish the OTP login challenge.",
                        },
                    )
                ],
            ),
            LLMResponse(
                content="Action: done",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "done", "status": "success"},
                    )
                ],
            ),
        ]
    )
    backend = _BackendDouble(
        [
            {
                "foreground_app": "Secure Login",
                "extra": {"display_id": "bg-desktop-1", "window_title": "Bank Login"},
            },
            {"foreground_app": "Bank Dashboard"},
        ]
    )
    handler = AsyncMock()

    async def _handle(request) -> SimpleNamespace:
        backend.execute.assert_not_awaited()
        assert backend.observe.await_count == 1
        assert request.platform == "linux"
        assert request.foreground_app == "Secure Login"
        assert request.target["display_id"] == "bg-desktop-1"
        return SimpleNamespace(resume_confirmed=True, note="user finished login")

    handler.request_intervention.side_effect = _handle

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "secure login"),
        intervention_handler=handler,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
    )

    result = await agent.run("Complete the secure login", max_retries=1)

    assert result.success
    backend.execute.assert_not_awaited()
    assert backend.observe.await_count == 2
    handler.request_intervention.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.explicit_resume_confirmation
async def test_intervention_waits_for_explicit_resume_confirmation(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content="Action: request intervention",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={
                            "action_type": "request_intervention",
                            "text": "The user must approve a payment step.",
                        },
                    )
                ],
            ),
            LLMResponse(
                content="Action: done",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "done", "status": "success"},
                    )
                ],
            ),
        ]
    )
    backend = _BackendDouble(
        [
            {"foreground_app": "Payment Approval"},
            {"foreground_app": "Payment Complete"},
        ]
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class _Handler:
        async def request_intervention(self, request) -> SimpleNamespace:
            assert request.reason == "The user must approve a payment step."
            entered.set()
            await release.wait()
            return SimpleNamespace(resume_confirmed=True, note="approved")

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "payment approval"),
        intervention_handler=_Handler(),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
    )

    task = asyncio.create_task(agent.run("Approve the payment", max_retries=1))
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    await asyncio.sleep(0)

    backend.execute.assert_not_awaited()
    assert backend.observe.await_count == 1
    assert not task.done()

    release.set()
    result = await asyncio.wait_for(task, timeout=1.0)

    assert result.success
    assert backend.observe.await_count == 2


@pytest.mark.asyncio
@pytest.mark.fresh_observation_after_intervention
async def test_resume_uses_fresh_observation_after_intervention(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content="Action: request intervention",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={
                            "action_type": "request_intervention",
                            "text": "User must complete MFA before continuing.",
                        },
                    )
                ],
            ),
            LLMResponse(
                content="Action: done",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "done", "status": "success"},
                    )
                ],
            ),
        ]
    )
    backend = _BackendDouble(
        [
            {"foreground_app": "Login Screen"},
            {"foreground_app": "Authenticated Workspace"},
        ]
    )

    class _Handler:
        async def request_intervention(self, request) -> SimpleNamespace:
            assert request.step_index == 1
            return SimpleNamespace(resume_confirmed=True, note="mfa complete")

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "mfa resume"),
        intervention_handler=_Handler(),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
    )

    result = await agent.run("Resume after MFA", max_retries=1)

    assert result.success
    assert Path(backend.observe.await_args_list[1].args[0]).name == "001_request_intervention.png"
    assert len(llm.calls) == 2


@pytest.mark.asyncio
@pytest.mark.scrub_sensitive_trace_fields
async def test_trace_and_trajectory_scrub_sensitive_intervention_fields(tmp_path: Path) -> None:
    reason = "Need the user to enter OTP 123456 for the payroll login."
    llm = _RecordingLLM(
        [
            LLMResponse(
                content="Action: request intervention",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={"action_type": "request_intervention", "text": reason},
                    )
                ],
            ),
            LLMResponse(
                content="Action: done",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "done", "status": "success"},
                    )
                ],
            ),
        ]
    )
    backend = _BackendDouble(
        [
            {
                "foreground_app": "Payroll Login",
                "extra": {"display_id": "desk-2", "session_token": "secret-session-token"},
            },
            {"foreground_app": "Payroll Home"},
        ]
    )
    recorder = _make_recorder(tmp_path, "scrub intervention")

    class _Handler:
        async def request_intervention(self, request) -> SimpleNamespace:
            assert request.target["session_token"] == "secret-session-token"
            return SimpleNamespace(resume_confirmed=True, note="user completed OTP")

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=recorder,
        intervention_handler=_Handler(),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
    )

    result = await agent.run("Handle the payroll OTP screen", max_retries=1)

    assert result.success
    assert not list(Path(result.trace_path).rglob("trace.jsonl"))
    trajectory_text = recorder.path.read_text(encoding="utf-8")

    assert "<redacted:intervention_reason>" in trajectory_text
    assert reason not in trajectory_text


@pytest.mark.asyncio
async def test_input_text_is_preserved_in_trace_artifacts(tmp_path: Path) -> None:
    typed_secret = "Sup3rS3cret OTP 123456"
    recorder = _make_recorder(tmp_path, "type secret")
    agent = GuiAgent(
        _RecordingLLM(
            [
                LLMResponse(
                    content="Action: input the code",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="computer_use",
                            arguments={"action_type": "input_text", "text": typed_secret},
                        )
                    ],
                ),
                LLMResponse(
                    content="Action: done",
                    tool_calls=[
                        ToolCall(
                            id="call-2",
                            name="computer_use",
                            arguments={"action_type": "done", "status": "success"},
                        )
                    ],
                ),
            ]
        ),
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
    )

    result = await agent.run("Enter the temporary code", max_retries=1)

    assert result.success
    assert not list(Path(result.trace_path).rglob("trace.jsonl"))
    trajectory = json.loads(recorder.path.read_text(encoding="utf-8"))
    serialized_trajectory = json.dumps(trajectory)

    assert typed_secret in serialized_trajectory
