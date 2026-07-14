"""Unit tests for the minimal flat GUIClaw skill stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from guiclaw.action import Action
from guiclaw.backends.dry_run import DryRunBackend
from guiclaw.interfaces import LLMResponse
from guiclaw.observation import Observation
from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.executor import ExecutionState, SkillExecutor, SubgoalResult
from guiclaw.skills.extractor import SkillExtractor
from guiclaw.skills.flat import FlatSkillLibrary, compile_flat_skills
from guiclaw.skills.normalization import normalize_app_identifier
from guiclaw.skills.state_contract import infer_focused_input_contract, infer_interaction_target
from guiclaw.skills.trajectory_codegen import CodegenResult, CodeStep, codegen_trajectory
from guiclaw.trajectory.recorder import TrajectoryRecorder


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = [LLMResponse(content=r) for r in responses]
        self.messages: list[list[dict[str, Any]]] = []
        self.max_tokens: list[int | None] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del tools, tool_choice, model
        self.messages.append(messages)
        self.max_tokens.append(max_tokens)
        if not self._responses:
            raise AssertionError("_ScriptedLLM has no remaining responses.")
        return self._responses.pop(0)


class _FakeValidator:
    def __init__(self, returns: list[bool]) -> None:
        self._returns = list(returns)
        self.calls: list[str] = []

    async def validate(
        self,
        valid_state: str,
        screenshot: Path | bytes | None = None,
    ) -> bool:
        del screenshot
        self.calls.append(valid_state)
        if not self._returns:
            raise AssertionError("_FakeValidator has no remaining results.")
        return self._returns.pop(0)


class _RecordingBackend(DryRunBackend):
    def __init__(self) -> None:
        super().__init__(screen_width=496, screen_height=1080)
        self.actions: list[Action] = []

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        del timeout
        self.actions.append(action)
        return action.action_type


class _ObservationProvider:
    def __init__(self, observations: list[Observation]) -> None:
        self._observations = list(observations)

    async def get_observation(self) -> Observation | None:
        if not self._observations:
            return None
        return self._observations.pop(0)


class _FakeSubgoalRunner:
    def __init__(self, result: SubgoalResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def run_subgoal(
        self,
        goal: str,
        screenshot: Path | bytes,
        *,
        max_steps: int = 3,
        current_observation: Observation | None = None,
    ) -> SubgoalResult:
        self.calls.append(
            {
                "goal": goal,
                "screenshot": screenshot,
                "max_steps": max_steps,
                "current_observation": current_observation,
            }
        )
        return self.result


class _RecordingEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("settings")),
            float(lowered.count("camera")),
            float(lowered.count("selfie")),
            float(lowered.count("browser")),
            1.0,
        ]


class _ConstantEmbeddingProvider:
    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0] for _text in texts], dtype=np.float32)


class _KeywordEmbeddingProvider:
    async def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float(lowered.count("messages")),
                    float(lowered.count("camera")),
                ]
            )
        return np.array(rows, dtype=np.float32)


def _make_skill(
    skill_id: str,
    name: str,
    description: str,
    *,
    app: str = "com.example.app",
    platform: str = "android",
    steps: tuple[SkillStep, ...] | None = None,
    success_count: int = 0,
    failure_count: int = 0,
) -> Skill:
    return Skill(
        skill_id=skill_id,
        name=name,
        description=description,
        app=app,
        platform=platform,
        steps=steps
        or (
            SkillStep(
                action_type="tap",
                target="Settings",
                valid_state="Settings icon is visible",
            ),
        ),
        created_at=1_700_000_000.0,
        success_count=success_count,
        failure_count=failure_count,
    )


def _focused_input_extra() -> dict[str, Any]:
    return {
        "ui_tree": [
            {
                "resource_id": "com.zhihu.android:id/input_text",
                "class": "android.widget.EditText",
                "content_desc": "Search query",
                "focused": True,
                "enabled": True,
                "bounds": "[10,20][300,80]",
            }
        ]
    }


def _write_compact_trajectory(path: Path, events: list[dict[str, Any]]) -> None:
    metadata = next((event for event in events if event.get("type") == "metadata"), {})
    platform = str(metadata.get("platform") or "unknown")
    steps: list[dict[str, Any]] = []
    pending_skill: dict[str, Any] = {}
    previous_observation: dict[str, Any] | None = None
    result_event = next((event for event in events if event.get("type") == "result"), {})

    for event in events:
        event_type = event.get("type")
        if event_type == "skill_step":
            pending_skill.update(
                {
                    key: event[key]
                    for key in ("skill_id", "skill_name")
                    if event.get(key) is not None
                }
            )
            pending_skill["failed_step"] = {
                key: event[key]
                for key in (
                    "step_index",
                    "target",
                    "valid_state",
                    "state_contract",
                    "observation",
                    "screenshot_path",
                    "error",
                )
                if event.get(key) is not None
            }
            continue
        if event_type == "skill_execution_result":
            pending_skill.update(
                {
                    key: event[key]
                    for key in ("skill_id", "skill_name", "state", "error")
                    if event.get(key) is not None
                }
            )
            continue
        if event_type != "step":
            continue

        action = event.get("action") or {}
        observation = event.get("observation") or {}
        app = str(observation.get("foreground_app") or observation.get("app") or "")

        interaction_target = event.get("interaction_target")
        action_type = action.get("action_type")
        if (
            not isinstance(interaction_target, dict)
            and action_type
            in {
                "tap",
                "long_press",
                "double_tap",
            }
            and previous_observation is not None
        ):
            interaction_target = infer_interaction_target(action, previous_observation)
        if not isinstance(interaction_target, dict) and action_type == "input_text":
            source = previous_observation or observation
            contract = infer_focused_input_contract(
                source.get("extra"),
                app=str(source.get("foreground_app") or source.get("app") or app),
            )
            if contract:
                interaction_target = {"state_contract": contract}

        compact_step: dict[str, Any] = {
            "step": len(steps) + 1,
            "subtask": 1,
            "attempt": 1,
            "phase": event.get("phase") or "agent",
            "model_output": event.get("model_output") or "",
            "action": action,
            "app": app,
        }
        if isinstance(interaction_target, dict):
            compact_step["interaction_target"] = interaction_target
        if pending_skill:
            compact_step["skill"] = dict(pending_skill)
            pending_skill.clear()
        steps.append(compact_step)
        previous_observation = observation

    if pending_skill:
        steps.append(
            {
                "step": len(steps) + 1,
                "subtask": 1,
                "attempt": 1,
                "phase": "skill",
                "model_output": "",
                "action": {"action_type": "use_skill"},
                "app": "",
                "skill": dict(pending_skill),
            }
        )

    task = str(metadata.get("task") or "test task")
    path.write_text(
        json.dumps(
            {
                "instruction": task,
                "platform": platform,
                "subtasks": [{"subtask": 1, "task": task, "app_hint": None}],
                "steps": steps,
                "screenshots": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    success = bool(result_event.get("success", True))
    result = {
        "run": {
            "success": success,
            "steps_taken": len(steps),
            "workflow_mode": "single",
            "token_usage": {},
        },
        "subtasks": [
            {
                "subtask": 1,
                "task": task,
                "success": success,
                "steps_taken": len(steps),
                "error": result_event.get("error"),
            }
        ],
    }
    (path.parent / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _codegen_result_from_steps(steps: list[dict[str, Any]]) -> CodegenResult:
    """Build focused extractor input without a production test-compat API."""
    first = steps[0]
    first_observation = first.get("observation") or {}
    platform = str(first_observation.get("platform") or "unknown")
    app = normalize_app_identifier(
        platform,
        str(first_observation.get("foreground_app") or first_observation.get("app") or ""),
    )
    app_candidates: list[str] = []
    for item in steps:
        action = item.get("action") or {}
        observation = item.get("observation") or {}
        raw_candidates = [
            action.get("text") if action.get("action_type") == "open_app" else None,
            observation.get("foreground_app") or observation.get("app"),
        ]
        for raw in raw_candidates:
            if not raw:
                continue
            candidate = normalize_app_identifier(platform, str(raw))
            if candidate not in {"", "unknown", "app"} and candidate not in app_candidates:
                app_candidates.append(candidate)
    if app in {"", "unknown", "app"} and app_candidates:
        app = app_candidates[0]

    code_steps: list[CodeStep] = []
    for index, item in enumerate(steps):
        action = item.get("action") or {}
        action_type = str(action.get("action_type") or "")
        code_steps.append(
            CodeStep(
                step_index=index,
                app=app,
                intent=str(item.get("action_intent") or item.get("action_summary") or action_type),
                action_type=action_type,
                action_params={
                    key: action[key]
                    for key in ("x", "y", "x2", "y2", "text", "key", "pixels")
                    if key in action and action[key] is not None
                },
                control_info="",
                contract_json="",
                screenshot_b64="",
            )
        )

    core_steps = [
        step
        for step in code_steps
        if step.action_type
        not in {
            "wait",
            "screenshot",
            "home",
            "back",
            "enter",
            "done",
            "app_switch",
            "request_intervention",
        }
    ]
    if len(core_steps) < 2:
        code_steps.append(
            CodeStep(
                step_index=len(code_steps),
                app=app,
                intent="continue reusable workflow",
                action_type="tap",
                action_params={"x": 1, "y": 1},
                control_info="",
                contract_json="",
                screenshot_b64="",
            )
        )

    return CodegenResult(
        task=str(first.get("task") or "test task"),
        platform=platform,
        app=app,
        app_candidates=tuple(app_candidates or (app,)),
        steps=code_steps,
        screenshots_b64=[],
    )


async def _extract_from_steps(
    extractor: SkillExtractor,
    steps: list[dict[str, Any]],
    *,
    is_success: bool = True,
) -> Skill | None:
    skills = await extractor.extract_from_codegen_result_multi(
        _codegen_result_from_steps(steps),
        is_success=is_success,
    )
    return skills[0] if skills else None


async def _extract_many_from_steps(
    extractor: SkillExtractor,
    steps: list[dict[str, Any]],
    *,
    is_success: bool = True,
) -> list[Skill]:
    return await extractor.extract_from_codegen_result_multi(
        _codegen_result_from_steps(steps),
        is_success=is_success,
    )


def test_compile_flat_skills_supports_helpers_contracts_and_parameters() -> None:
    source = """
from guiclaw.skills.flat import C, R, action, skill

async def search_box(device, query):
    await action("tap", target="Search", state_contract=C(app="com.example", required=[R(text="Search", visible=True)]))
    await action("input_text", target=query, text=query, valid_state="Search box focused")

@skill(app="com.example", platform="android", skill_id="flat:search")
async def search_example(device, query):
    await search_box(device, query)
"""
    result = compile_flat_skills(source)

    assert result.errors == ()
    assert len(result.skills) == 1
    skill = result.skills[0]
    assert skill.skill_id == "flat:search"
    assert skill.parameters == ("query",)
    assert [step.action_type for step in skill.steps] == ["tap", "input_text"]
    assert skill.steps[0].state_contract is not None
    assert skill.steps[1].target == "{{query}}"


def test_flat_skill_library_crud_persists_to_skills_py(tmp_path: Path) -> None:
    store = tmp_path / "skills"
    lib = FlatSkillLibrary(store_dir=store)
    skill = _make_skill("s1", "Open Settings", "Open Android Settings")

    lib.add(skill)

    assert lib.count() == 1
    assert (store / "skills.py").is_file()
    assert not list(store.rglob("skills.json"))
    reloaded = FlatSkillLibrary(store_dir=store)
    assert reloaded.get("s1") == skill

    updated = _make_skill("s1", "Open Settings", "Updated")
    assert reloaded.update("s1", updated) is True
    assert FlatSkillLibrary(store_dir=store).get("s1").description == "Updated"
    assert reloaded.remove("s1") is True
    assert FlatSkillLibrary(store_dir=store).count() == 0


def test_infer_interaction_target_uses_pre_action_ui_node() -> None:
    observation = Observation(
        screenshot_path=None,
        screen_width=1080,
        screen_height=2400,
        foreground_app="com.example",
        platform="android",
        extra={
            "ui_tree": [
                {"text": "Root", "bounds": "[0,0][1080,2400]", "class": "FrameLayout"},
                {
                    "text": "Settings",
                    "resource_id": "com.example:id/settings_btn",
                    "class": "Button",
                    "clickable": True,
                    "bounds": "[100,200][300,260]",
                },
            ],
        },
    )

    target = infer_interaction_target(
        Action(action_type="tap", x=150, y=230),
        observation,
    )

    assert target is not None
    assert target["selector"] == {"resource_id": "com.example:id/settings_btn"}
    contract = target["state_contract"]
    assert contract["anchor"]["app_package"] == "com.example"
    assert contract["signature"]["required"][0]["selector"] == target["selector"]


@pytest.mark.asyncio
async def test_flat_skill_library_search_returns_relevant_skill(tmp_path: Path) -> None:
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills")
    lib.add(_make_skill("settings", "Open Settings", "Navigate to Android settings"))
    lib.add(_make_skill("camera", "Open Camera", "Take a photo with the camera"))

    results = await lib.search("settings screen", platform="android", top_k=2)

    assert results
    assert results[0][0].skill_id == "settings"


@pytest.mark.asyncio
async def test_flat_skill_library_add_or_merge_deduplicates_semantic_conflict(
    tmp_path: Path,
) -> None:
    steps = (
        SkillStep(
            action_type="open_app", target="Launch WeChat", parameters={"text": "com.tencent.mm"}
        ),
        SkillStep(action_type="tap", target="Messages", valid_state="Messages tab is visible"),
        SkillStep(
            action_type="tap",
            target="Verification code",
            valid_state="Verification message is visible",
        ),
    )
    lib = FlatSkillLibrary(
        store_dir=tmp_path / "skills",
        embedding_provider=_RecordingEmbeddingProvider(),
        embedding_signature="sig-v1",
    )
    lib.add(
        _make_skill(
            "read-code", "read_verification_code", "Open WeChat and read a login code", steps=steps
        )
    )

    decision, skill_id = await lib.add_or_merge(
        _make_skill("otp", "get_otp_from_message", "Open WeChat and read a login code", steps=steps)
    )

    assert decision in {"MERGE", "KEEP_NEW"}
    assert skill_id is not None
    assert FlatSkillLibrary(store_dir=tmp_path / "skills").count() == 1


@pytest.mark.asyncio
async def test_flat_skill_library_rejects_unknown_app(tmp_path: Path) -> None:
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills")

    decision, skill_id = await lib.add_or_merge(
        _make_skill("unknown-app", "Open Calendar", "Open Calendar", app="unknown")
    )

    assert decision == "REJECT_UNKNOWN_APP"
    assert skill_id is None
    assert FlatSkillLibrary(store_dir=tmp_path / "skills").count() == 0


@pytest.mark.asyncio
async def test_flat_skill_library_add_or_merge_uses_description_when_names_differ(
    tmp_path: Path,
) -> None:
    steps = (
        SkillStep(
            action_type="open_app", target="Launch WeChat", parameters={"text": "com.tencent.mm"}
        ),
        SkillStep(action_type="tap", target="Messages", valid_state="Messages tab is visible"),
        SkillStep(
            action_type="tap",
            target="Verification code",
            valid_state="Verification message is visible",
        ),
    )
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills")
    lib.add(
        _make_skill(
            "read-code", "read_code_from_sms", "Open WeChat and read a login code", steps=steps
        )
    )

    decision, skill_id = await lib.add_or_merge(
        _make_skill("otp", "fetch_login_number", "Open WeChat and read a login code", steps=steps)
    )

    assert decision in {"MERGE", "KEEP_NEW"}
    assert skill_id is not None
    assert FlatSkillLibrary(store_dir=tmp_path / "skills").count() == 1


@pytest.mark.asyncio
async def test_flat_skill_library_search_includes_android_app_display_alias(tmp_path: Path) -> None:
    library = FlatSkillLibrary(store_dir=tmp_path / "skills")
    library.add(
        _make_skill(
            "bili-search",
            "search_and_play_bilibili_video",
            "Search for a video and play the first result",
            app="tv.danmaku.bili",
            platform="android",
            steps=(
                SkillStep(action_type="open_app", target="tv.danmaku.bili"),
                SkillStep(action_type="input_text", target="{{query}}"),
            ),
        )
    )

    results = await library.search("在哔哩哔哩搜索播放华强买瓜", platform="android", top_k=1)

    assert results
    assert results[0][0].skill_id == "bili-search"


@pytest.mark.asyncio
async def test_flat_skill_library_search_expands_common_bilingual_action_aliases(
    tmp_path: Path,
) -> None:
    library = FlatSkillLibrary(store_dir=tmp_path / "skills")
    library.add(
        _make_skill(
            "bili-open",
            "open_bilibili_app",
            "Open Bilibili",
            app="tv.danmaku.bili",
            platform="android",
            steps=(SkillStep(action_type="open_app", target="tv.danmaku.bili"),),
        )
    )
    library.add(
        _make_skill(
            "bili-search-play",
            "search_and_play_bilibili_video",
            "Search for a video and play the first result",
            app="tv.danmaku.bili",
            platform="android",
            steps=(
                SkillStep(action_type="open_app", target="tv.danmaku.bili"),
                SkillStep(action_type="input_text", target="{{query}}"),
                SkillStep(action_type="tap", target="first_video_result"),
            ),
        )
    )

    results = await library.search("在哔哩哔哩搜索播放华强买瓜", platform="android", top_k=1)

    assert results
    assert results[0][0].skill_id == "bili-search-play"


@pytest.mark.asyncio
async def test_flat_skill_library_does_not_merge_same_embedding_for_different_targets(
    tmp_path: Path,
) -> None:
    lib = FlatSkillLibrary(
        store_dir=tmp_path / "skills",
        embedding_provider=_ConstantEmbeddingProvider(),
        embedding_signature="constant",
    )
    lib.add(
        _make_skill(
            "settings",
            "open_entry",
            "Open the requested entry",
            steps=(
                SkillStep(
                    action_type="tap", target="Settings", valid_state="Settings icon visible"
                ),
            ),
        )
    )

    decision, skill_id = await lib.add_or_merge(
        _make_skill(
            "camera",
            "open_entry_variant",
            "Open the requested entry",
            steps=(
                SkillStep(action_type="tap", target="Camera", valid_state="Camera icon visible"),
            ),
        )
    )

    assert decision == "ADD"
    assert skill_id == "camera"
    assert FlatSkillLibrary(store_dir=tmp_path / "skills").count() == 2


@pytest.mark.asyncio
async def test_flat_skill_library_migrates_feedback_when_incoming_skill_merges(
    tmp_path: Path,
) -> None:
    steps = (SkillStep(action_type="tap", target="Messages", valid_state="Messages tab visible"),)
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills")
    lib.add(_make_skill("existing", "open_messages", "Open messages", steps=steps))
    lib.record_feedback(
        "incoming",
        task="Open messages",
        failure_case={"execution_error": "popup blocked action", "failed_target": "Messages"},
        status="failure_detected",
    )

    decision, skill_id = await lib.add_or_merge(
        _make_skill("incoming", "messages_shortcut", "Open messages", steps=steps)
    )

    assert decision == "MERGE"
    assert skill_id == "existing"
    feedback = lib.feedback_for_skill("existing")
    assert feedback["negative_tasks"] == ["Open messages"]
    assert feedback["failure_counts"]["popup blocked action"] == 1
    assert lib.feedback_for_skill("incoming") == {}


@pytest.mark.asyncio
async def test_flat_skill_library_keeps_proven_old_skill_for_weaker_unproven_conflict(
    tmp_path: Path,
) -> None:
    store = tmp_path / "skills"
    lib = FlatSkillLibrary(
        store_dir=store,
        embedding_provider=_ConstantEmbeddingProvider(),
        embedding_signature="constant",
    )
    old = _make_skill(
        "old",
        "open_settings",
        "Open settings",
        steps=(
            SkillStep(action_type="tap", target="Settings", valid_state="Settings icon visible"),
        ),
        success_count=3,
    )
    lib.add(old)

    decision, skill_id = await lib.add_or_merge(
        _make_skill(
            "new",
            "open_settings_variant",
            "Open settings",
            steps=(
                SkillStep(
                    action_type="tap", target="Settings gear", valid_state="Settings icon visible"
                ),
            ),
        )
    )

    assert decision == "KEEP_OLD"
    assert skill_id == "old"
    reloaded = FlatSkillLibrary(store_dir=store)
    assert reloaded.count() == 1
    assert reloaded.get("old") == old


@pytest.mark.asyncio
async def test_flat_skill_library_replaces_unproven_old_when_new_has_success(
    tmp_path: Path,
) -> None:
    store = tmp_path / "skills"
    lib = FlatSkillLibrary(store_dir=store)
    lib.add(
        _make_skill(
            "old",
            "open_messages",
            "Open messages",
            steps=(
                SkillStep(action_type="tap", target="Messages", valid_state="Messages visible"),
            ),
        )
    )

    decision, skill_id = await lib.add_or_merge(
        _make_skill(
            "new",
            "open_messages_verified",
            "Open messages",
            steps=(
                SkillStep(action_type="tap", target="Messages", valid_state="Messages visible"),
            ),
            success_count=2,
        )
    )

    assert decision == "KEEP_NEW"
    assert skill_id == "new"
    reloaded = FlatSkillLibrary(store_dir=store)
    assert reloaded.get("old") is None
    assert reloaded.get("new") is not None


@pytest.mark.asyncio
async def test_flat_skill_library_cleanup_removes_zero_success_superseded_prefix(
    tmp_path: Path,
) -> None:
    prefix = _make_skill(
        "prefix",
        "open_search",
        "Open search",
        steps=(
            SkillStep(
                action_type="open_app",
                target="Launch Store",
                parameters={"text": "com.example.app"},
            ),
            SkillStep(action_type="tap", target="Search", valid_state="Search button visible"),
        ),
    )
    longer = _make_skill(
        "longer",
        "search_store",
        "Search store for an item",
        steps=(
            *prefix.steps,
            SkillStep(
                action_type="input_text", target="{{query}}", valid_state="Search field focused"
            ),
        ),
        success_count=3,
    )
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills")
    lib.add(prefix)

    decision, skill_id = await lib.add_or_merge(longer)

    assert decision == "ADD"
    assert skill_id == "longer"
    reloaded = FlatSkillLibrary(store_dir=tmp_path / "skills")
    assert reloaded.get("prefix") is None
    assert reloaded.get("longer") is not None


@pytest.mark.asyncio
async def test_flat_skill_library_cleanup_prunes_feedback_for_removed_prefix(
    tmp_path: Path,
) -> None:
    prefix = _make_skill(
        "prefix",
        "open_search",
        "Open search",
        steps=(
            SkillStep(
                action_type="open_app",
                target="Launch Store",
                parameters={"text": "com.example.app"},
            ),
            SkillStep(action_type="tap", target="Search", valid_state="Search button visible"),
        ),
    )
    longer = _make_skill(
        "longer",
        "search_store",
        "Search store for an item",
        steps=(
            *prefix.steps,
            SkillStep(
                action_type="input_text", target="{{query}}", valid_state="Search field focused"
            ),
        ),
        success_count=3,
    )
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills")
    lib.add(prefix)
    lib.record_feedback(
        "prefix",
        task="Search store",
        failure_case={"execution_error": "stopped too early"},
        status="failure_detected",
    )

    await lib.add_or_merge(longer)

    assert lib.feedback_for_skill("prefix") == {}
    assert lib.feedback_for_skill("longer") == {}


@pytest.mark.asyncio
async def test_flat_skill_library_caches_skill_embeddings_and_only_embeds_query_on_hit(
    tmp_path: Path,
) -> None:
    store = tmp_path / "skills"
    embedder = _RecordingEmbeddingProvider()
    lib = FlatSkillLibrary(
        store_dir=store, embedding_provider=embedder, embedding_signature="sig-v1"
    )
    lib.add(_make_skill("settings", "Open Settings", "Navigate to Android settings"))
    lib.add(_make_skill("camera", "Open Camera", "Take a photo with the camera"))

    results = await lib.search("settings screen", platform="android", top_k=2)

    assert results[0][0].skill_id == "settings"
    assert (store / "skills_embeddings.npy").is_file()
    assert (store / "skills_embeddings_meta.json").is_file()
    assert len(embedder.calls) == 2
    assert len(embedder.calls[0]) == 2
    assert embedder.calls[1] == ["settings screen"]

    embedder.calls.clear()
    results = await lib.search("camera", platform="android", top_k=2)

    assert results[0][0].skill_id == "camera"
    assert embedder.calls == [["camera"]]


@pytest.mark.asyncio
async def test_flat_skill_library_reuses_unchanged_skill_embeddings_when_skills_py_changes(
    tmp_path: Path,
) -> None:
    store = tmp_path / "skills"
    embedder = _RecordingEmbeddingProvider()
    lib = FlatSkillLibrary(
        store_dir=store, embedding_provider=embedder, embedding_signature="sig-v1"
    )
    lib.add(_make_skill("settings", "Open Settings", "Navigate to Android settings"))
    lib.add(_make_skill("camera", "Open Camera", "Take a photo with the camera"))
    await lib.search("settings screen", platform="android", top_k=2)

    embedder.calls.clear()
    lib.update("camera", _make_skill("camera", "Open Camera", "Take a camera selfie"))
    results = await lib.search("camera selfie", platform="android", top_k=2)

    assert results[0][0].skill_id == "camera"
    assert len(embedder.calls) == 2
    rebuilt_skill_texts = embedder.calls[0]
    assert len(rebuilt_skill_texts) == 1
    assert "Take a camera selfie" in rebuilt_skill_texts[0]
    assert "Open Settings" not in rebuilt_skill_texts[0]
    assert embedder.calls[1] == ["camera selfie"]


@pytest.mark.asyncio
async def test_flat_skill_library_rebuilds_skill_embeddings_when_signature_changes(
    tmp_path: Path,
) -> None:
    store = tmp_path / "skills"
    first_embedder = _RecordingEmbeddingProvider()
    lib = FlatSkillLibrary(
        store_dir=store, embedding_provider=first_embedder, embedding_signature="sig-v1"
    )
    lib.add(_make_skill("settings", "Open Settings", "Navigate to Android settings"))
    lib.add(_make_skill("camera", "Open Camera", "Take a photo with the camera"))
    await lib.search("settings screen", platform="android", top_k=2)

    second_embedder = _RecordingEmbeddingProvider()
    reloaded = FlatSkillLibrary(
        store_dir=store, embedding_provider=second_embedder, embedding_signature="sig-v2"
    )
    await reloaded.search("camera", platform="android", top_k=2)

    assert len(second_embedder.calls) == 2
    assert len(second_embedder.calls[0]) == 2
    assert second_embedder.calls[1] == ["camera"]


@pytest.mark.asyncio
async def test_postprocessor_uses_add_or_merge_for_extracted_flat_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.postprocessing import PostRunProcessor

    store = tmp_path / "skills"
    existing = _make_skill(
        "existing",
        "open_settings",
        "Open Android settings",
        steps=(
            SkillStep(action_type="tap", target="Settings", valid_state="Settings icon visible"),
        ),
    )
    incoming = _make_skill(
        "incoming",
        "open_settings",
        "Open Android settings",
        steps=(
            SkillStep(action_type="tap", target="Settings", valid_state="Settings icon visible"),
        ),
    )
    FlatSkillLibrary(store_dir=store).add(existing)

    async def fake_induce_compact_skills_from_trace(
        extractor: object,
        trace_path: Path,
        *,
        is_success: bool = True,
        subtask_index: int = 1,
    ) -> list[Skill]:
        del extractor, trace_path, is_success, subtask_index
        return [incoming]

    monkeypatch.setattr(
        "guiclaw.postprocessing.induce_compact_skills_from_trace",
        fake_induce_compact_skills_from_trace,
    )
    recorder = TrajectoryRecorder(output_dir=tmp_path, task="Open settings", platform="android")
    trace_path = recorder.start()
    recorder.record_step(action={"action_type": "tap"})
    recorder.finish(success=True)

    processor = PostRunProcessor(
        llm=_ScriptedLLM([]),
        skill_store_root=store,
        enable_skill_extraction=True,
    )

    result_id = await processor._extract_skill(
        trace_path,
        True,
        "android",
        task="Open settings",
        subtask_index=1,
    )

    assert result_id == "existing"
    assert FlatSkillLibrary(store_dir=store).count() == 1
    run_result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    extraction_result = run_result["extraction"]["1"]
    assert extraction_result["skills"][0]["decision"] == "MERGE"
    assert extraction_result["compiled_skill_ids"] == ["existing"]


def _memory_response(title: str = "Search flow") -> str:
    return (
        "# Memory Item 1\n"
        f"## Title {title}\n"
        "## Description Use when searching inside an app.\n"
        "## Content Focus the search field before entering the query.\n"
    )


def _record_memory_trace(
    output_dir: Path,
    *,
    steps: int = 3,
    success: bool = True,
    error: str | None = None,
) -> Path:
    recorder = TrajectoryRecorder(
        output_dir=output_dir,
        task="Search for an item",
        platform="android",
    )
    trace_path = recorder.start()
    for index in range(steps):
        recorder.record_step(
            action={"action_type": "tap", "target": f"control-{index}"},
            model_output=f"Thought: interact with control {index}",
            foreground_app="com.example.app",
        )
    recorder.finish(success=success, error=error)
    return trace_path


@pytest.mark.asyncio
async def test_postprocessor_extracts_memory_to_configured_bank(tmp_path: Path) -> None:
    from guiclaw.postprocessing import PostRunProcessor

    trace_path = _record_memory_trace(tmp_path / "run")
    bank_path = tmp_path / "memory" / "gui_memory_bank.jsonl"
    processor = PostRunProcessor(
        llm=_ScriptedLLM([_memory_response()]),
        enable_memory_extraction=True,
        memory_bank_path=bank_path,
    )

    result = await processor._extract_memory(
        trace_path,
        is_success=True,
        task="Search for an item",
        subtask_index=1,
    )

    assert result == {"status": "processed", "added_count": 1, "duplicate_count": 0}
    stored = bank_path.read_text(encoding="utf-8")
    assert "Search flow" in stored
    run_result = json.loads((trace_path.parent / "result.json").read_text(encoding="utf-8"))
    assert run_result["memory_extraction"]["1"] == result


@pytest.mark.asyncio
async def test_postprocessor_skips_memory_when_disabled(tmp_path: Path) -> None:
    from guiclaw.postprocessing import PostRunProcessor

    trace_path = _record_memory_trace(tmp_path / "run")
    bank_path = tmp_path / "memory" / "gui_memory_bank.jsonl"
    processor = PostRunProcessor(
        llm=_ScriptedLLM([]),
        enable_memory_extraction=False,
        memory_bank_path=bank_path,
    )

    result = await processor._extract_memory(
        trace_path,
        is_success=True,
        task="Search for an item",
    )

    assert result is None
    assert not bank_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("steps", "error", "reason"),
    [
        (2, None, "short_trajectory"),
        (3, "intervention_cancelled", "abnormal_termination"),
    ],
)
async def test_postprocessor_skips_unusable_memory_traces(
    tmp_path: Path,
    steps: int,
    error: str | None,
    reason: str,
) -> None:
    from guiclaw.postprocessing import PostRunProcessor

    trace_path = _record_memory_trace(
        tmp_path / reason,
        steps=steps,
        success=False,
        error=error,
    )
    processor = PostRunProcessor(
        llm=_ScriptedLLM([]),
        enable_memory_extraction=True,
        memory_bank_path=tmp_path / "bank.jsonl",
    )

    result = await processor._extract_memory(
        trace_path,
        is_success=False,
        task="Search for an item",
    )

    assert result == {"status": "skipped", "reason": reason}


@pytest.mark.asyncio
async def test_memory_extraction_is_not_suppressed_by_skill_evolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.postprocessing import PostRunProcessor

    trace_path = _record_memory_trace(tmp_path / "run")
    processor = PostRunProcessor(
        llm=_ScriptedLLM([]),
        enable_skill_extraction=True,
        enable_memory_extraction=True,
    )
    memory = AsyncMock(return_value={"status": "processed", "added_count": 1})
    monkeypatch.setattr(processor, "_extract_memory", memory)
    monkeypatch.setattr(processor, "_summarize_trajectory", AsyncMock(return_value=""))
    monkeypatch.setattr(processor, "_run_evaluation", AsyncMock(return_value=None))
    monkeypatch.setattr(
        processor,
        "_evolve_failed_skill",
        AsyncMock(return_value={"status": "processed_evolution"}),
    )

    await processor._run_all(
        trace_path,
        is_success=True,
        platform="android",
        task="Search for an item",
    )

    memory.assert_awaited_once_with(
        trace_path,
        is_success=True,
        task="Search for an item",
        subtask_index=1,
    )


@pytest.mark.asyncio
async def test_postprocessor_evolves_failed_reused_skill_instead_of_extracting_new(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.postprocessing import PostRunProcessor
    from guiclaw.skills.extractor import SkillExtractor

    store = tmp_path / "skills"
    FlatSkillLibrary(store_dir=store).add(
        _make_skill(
            "skill-1",
            "open_messages",
            "Open messages",
            steps=(
                SkillStep(action_type="tap", target="Messages", valid_state="Messages tab visible"),
            ),
        )
    )
    evolved_payload = json.dumps(
        {
            "name": "open_messages",
            "description": "Open messages and dismiss popup when present",
            "app": "com.example.app",
            "platform": "android",
            "parameters": [],
            "steps": [
                {
                    "action_type": "tap",
                    "target": "Close",
                    "parameters": {"optional": True},
                    "valid_state": "popup close button is visible",
                },
                {
                    "action_type": "tap",
                    "target": "Messages",
                    "valid_state": "Messages tab visible",
                },
            ],
        }
    )
    extractor = monkeypatch.setattr(
        SkillExtractor,
        "extract_from_file_multi",
        pytest.fail,
    )
    del extractor
    recorder = TrajectoryRecorder(output_dir=tmp_path, task="Open messages", platform="android")
    trace_path = recorder.start()
    recorder.record_event("skill_execution_start", skill_id="skill-1", skill_name="open_messages")
    recorder.record_event(
        "skill_step",
        skill_id="skill-1",
        skill_name="open_messages",
        step_index=0,
        target="Messages",
        valid_state="Messages tab visible",
        valid_state_check=False,
        error="valid_state not reached: popup visible",
        observation={"foreground_app": "com.example.app", "platform": "android"},
    )
    recorder.record_event(
        "skill_execution_result",
        skill_id="skill-1",
        skill_name="open_messages",
        state="failed",
        error="Step 0 valid_state not reached",
    )
    recorder.record_step(action={"action_type": "use_skill", "text": "skill-1"})
    recorder.finish(success=True)
    processor = PostRunProcessor(
        llm=_ScriptedLLM([evolved_payload]),
        skill_store_root=store,
        enable_skill_extraction=True,
    )
    monkeypatch.setattr(processor, "_summarize_trajectory", AsyncMock(return_value=""))

    await processor._run_all(trace_path, is_success=True, platform="android", task="Open messages")

    evolved = FlatSkillLibrary(store_dir=store).get("skill-1")
    assert evolved is not None
    assert evolved.description == "Open messages and dismiss popup when present"
    assert evolved.steps[0].parameters["optional"] is True
    run_result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    evolution_result = run_result["evolution"]["1"]
    assert evolution_result["status"] == "processed_evolution"
    extraction_result = run_result["extraction"]["1"]
    assert extraction_result["status"] == "processed_evolution"
    assert extraction_result["ordinary_code_extraction_skipped"] is True
    feedback = FlatSkillLibrary(store_dir=store).feedback_for_skill("skill-1")
    assert feedback["negative_tasks"] == ["Open messages"]
    assert feedback["failure_counts"]["Step 0 valid_state not reached"] == 1
    assert feedback["last_evolution_status"] == "processed_evolution"
    assert feedback["evolution_count"] == 1


@pytest.mark.asyncio
async def test_postprocessor_evolution_injects_focused_input_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.postprocessing import PostRunProcessor
    from guiclaw.skills.extractor import SkillExtractor

    store = tmp_path / "skills"
    FlatSkillLibrary(store_dir=store).add(
        _make_skill(
            "skill-1",
            "search_zhihu",
            "Search Zhihu",
            app="com.zhihu.android",
            steps=(
                SkillStep(
                    action_type="input_text", target="{{query}}", valid_state="Search field focused"
                ),
            ),
        )
    )
    evolved_payload = json.dumps(
        {
            "name": "search_zhihu",
            "description": "Search Zhihu",
            "app": "com.zhihu.android",
            "platform": "android",
            "parameters": ["query"],
            "steps": [
                {
                    "action_type": "input_text",
                    "target": "{{query}}",
                    "valid_state": "Search field focused",
                }
            ],
        }
    )
    monkeypatch.setattr(SkillExtractor, "extract_from_file_multi", pytest.fail)
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {
                "type": "skill_step",
                "skill_id": "skill-1",
                "skill_name": "search_zhihu",
                "step_index": 0,
                "target": "{{query}}",
                "valid_state": "Search field focused",
                "valid_state_check": False,
                "error": "valid_state not reached: Search field focused",
            },
            {
                "type": "skill_execution_result",
                "skill_id": "skill-1",
                "skill_name": "search_zhihu",
                "state": "failed",
                "error": "Step 0 valid_state not reached",
            },
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 100, "y": 40},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.zhihu.android",
                    "extra": _focused_input_extra(),
                },
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "input_text", "text": "强化学习"},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.zhihu.android",
                    "extra": {"ui_tree": []},
                },
            },
            {"type": "result", "success": True, "total_steps": 2},
        ],
    )
    processor = PostRunProcessor(
        llm=_ScriptedLLM([evolved_payload]),
        skill_store_root=store,
        enable_skill_extraction=True,
    )
    monkeypatch.setattr(processor, "_summarize_trajectory", AsyncMock(return_value=""))

    await processor._run_all(trace_path, is_success=True, platform="android", task="Search Zhihu")

    evolved = FlatSkillLibrary(store_dir=store).get("skill-1")
    assert evolved is not None
    required = evolved.steps[0].state_contract["signature"]["required"]
    assert required[0]["selector"] == {
        "resource_id": "com.zhihu.android:id/input_text",
        "class": "android.widget.EditText",
    }
    assert required[0]["state"] == ["visible", "enabled", "focused"]


@pytest.mark.asyncio
async def test_postprocessor_rejects_evolved_skill_that_drifts_from_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.postprocessing import PostRunProcessor
    from guiclaw.skills.extractor import SkillExtractor

    store = tmp_path / "skills"
    original = _make_skill(
        "skill-1",
        "open_messages",
        "Open messages",
        steps=(
            SkillStep(action_type="tap", target="Messages", valid_state="Messages tab visible"),
        ),
    )
    FlatSkillLibrary(store_dir=store).add(original)
    drifted_payload = json.dumps(
        {
            "name": "open_camera",
            "description": "Open camera",
            "app": "com.example.app",
            "platform": "android",
            "parameters": [],
            "steps": [
                {
                    "action_type": "tap",
                    "target": "Camera",
                    "valid_state": "Camera icon visible",
                },
            ],
        }
    )
    monkeypatch.setattr(SkillExtractor, "extract_from_file_multi", pytest.fail)
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {
                "type": "skill_step",
                "skill_id": "skill-1",
                "skill_name": "open_messages",
                "step_index": 0,
                "target": "Messages",
                "valid_state": "Messages tab visible",
                "valid_state_check": False,
                "error": "valid_state not reached: camera visible",
            },
            {
                "type": "skill_execution_result",
                "skill_id": "skill-1",
                "skill_name": "open_messages",
                "state": "failed",
                "error": "Step 0 valid_state not reached",
            },
            {"type": "result", "success": True, "total_steps": 2},
        ],
    )
    processor = PostRunProcessor(
        llm=_ScriptedLLM([drifted_payload]),
        skill_store_root=store,
        enable_skill_extraction=True,
        embedding_provider=_KeywordEmbeddingProvider(),
        embedding_signature="keywords",
    )
    monkeypatch.setattr(processor, "_summarize_trajectory", AsyncMock(return_value=""))

    await processor._run_all(trace_path, is_success=True, platform="android", task="Open messages")

    unchanged = FlatSkillLibrary(store_dir=store).get("skill-1")
    assert unchanged == original
    run_result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    evolution_result = run_result["evolution"]["1"]
    assert evolution_result["status"] == "evolution_rejected"
    assert evolution_result["reason"] == "low_embedding_similarity"
    extraction_result = run_result["extraction"]["1"]
    assert extraction_result["status"] == "evolution_error"
    feedback = FlatSkillLibrary(store_dir=store).feedback_for_skill("skill-1")
    assert feedback["last_evolution_status"] == "rejected:low_embedding_similarity"


def test_build_failure_case_ignores_failed_result_without_skill_id(tmp_path: Path) -> None:
    from guiclaw.skills.evolution import _build_failure_case

    failure_case = _build_failure_case(
        [
            {
                "type": "skill_execution_result",
                "state": "failed",
                "error": "Skill failed without metadata",
            },
        ],
        trace_path=tmp_path / "traj.json",
        task="Open messages",
        platform="android",
    )

    assert failure_case is None


@pytest.mark.asyncio
async def test_skill_executor_runs_validated_steps() -> None:
    skill = _make_skill(
        "s1",
        "Wait",
        "Wait for a moment",
        steps=(
            SkillStep(
                action_type="wait",
                target="pause",
                parameters={"duration_ms": 1},
                valid_state="ready",
            ),
        ),
    )
    executor = SkillExecutor(
        backend=DryRunBackend(),
        state_validator=_FakeValidator([True]),
    )

    result = await executor.execute(skill)

    assert result.state == ExecutionState.SUCCEEDED
    assert result.step_results[0].valid_state_check is True


@pytest.mark.asyncio
async def test_skill_executor_rejects_required_step_missing_valid_state() -> None:
    backend = _RecordingBackend()
    validator = _FakeValidator([True])
    skill = _make_skill(
        "s1",
        "Tap Settings",
        "Tap Settings",
        steps=(SkillStep(action_type="tap", target="Settings"),),
    )
    executor = SkillExecutor(backend=backend, state_validator=validator)

    result = await executor.execute(skill)

    assert result.state == ExecutionState.FAILED
    assert backend.actions == []
    assert validator.calls == []
    assert result.step_results[0].error == "valid_state not reached: state_contract"


@pytest.mark.asyncio
async def test_skill_executor_falls_back_to_valid_state_when_contract_unknown() -> None:
    validator = _FakeValidator([True])
    skill = _make_skill(
        "s1",
        "Tap Settings",
        "Tap Settings",
        steps=(
            SkillStep(
                action_type="wait",
                target="Settings",
                valid_state="Settings button is visible",
                state_contract={
                    "anchor": {"app_package": "com.example"},
                    "signature": {
                        "required": [
                            {
                                "selector": {"resource_id": "com.example:id/settings_btn"},
                                "state": ["visible", "clickable"],
                            }
                        ],
                        "forbidden": [],
                    },
                    "mask_rules": [],
                },
            ),
        ),
    )
    executor = SkillExecutor(backend=DryRunBackend(), state_validator=validator)

    result = await executor.execute(skill)

    assert result.state == ExecutionState.SUCCEEDED
    assert validator.calls == ["Settings button is visible"]


@pytest.mark.asyncio
async def test_skill_executor_passes_observation_to_subgoal_and_skips_revalidation(
    tmp_path: Path,
) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"png")
    observation = Observation(
        screenshot_path=str(screenshot),
        screen_width=496,
        screen_height=1080,
        foreground_app="com.example",
        platform="android",
    )
    validator = _FakeValidator([False])
    subgoal = _FakeSubgoalRunner(
        SubgoalResult(
            success=True,
            steps_taken=1,
            action_summaries=["goal reached"],
            final_screenshot=screenshot,
            final_observation=observation,
            done_judgment=True,
        )
    )
    backend = _RecordingBackend()
    skill = _make_skill(
        "s1",
        "Tap Settings",
        "Tap Settings",
        steps=(
            SkillStep(
                action_type="tap",
                target="Settings",
                valid_state="Settings button is visible",
                fixed=True,
                fixed_values={"x": 10, "y": 20},
            ),
        ),
    )
    executor = SkillExecutor(
        backend=backend,
        state_validator=validator,
        subgoal_runner=subgoal,
        screenshot_provider=_ObservationProvider([observation]),
    )

    result = await executor.execute(skill)

    assert result.state == ExecutionState.SUCCEEDED
    assert validator.calls == ["Settings button is visible"]
    assert len(subgoal.calls) == 1
    assert subgoal.calls[0]["current_observation"] is observation
    assert backend.actions[0].action_type == "tap"
    assert result.step_results[0].recovery_result is subgoal.result


@pytest.mark.asyncio
async def test_skill_executor_dismisses_post_open_app_skip_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import guiclaw.skills.executor as executor_module

    monkeypatch.setattr(executor_module, "_OPEN_APP_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(executor_module, "_POST_ACTION_SETTLE_SECONDS", 0.0)
    backend = _RecordingBackend()
    skill = _make_skill(
        "s1",
        "Open Bilibili",
        "Open Bilibili",
        steps=(
            SkillStep(
                action_type="open_app", target="tv.danmaku.bili", valid_state="No need to verify"
            ),
        ),
    )
    provider = _ObservationProvider(
        [
            Observation(None, 496, 1080, foreground_app="tv.danmaku.bili", platform="android"),
            Observation(
                None,
                496,
                1080,
                foreground_app="tv.danmaku.bili",
                platform="android",
                extra={
                    "ui_tree": [
                        {"class": "FrameLayout", "enabled": True, "bounds": "[0,0][1440,3120]"},
                        {
                            "text": "Skip 2",
                            "class": "TextView",
                            "clickable": True,
                            "enabled": True,
                            "bounds": "[1062,2775][1384,2936]",
                        },
                    ],
                },
            ),
        ]
    )
    executor = SkillExecutor(backend=backend, screenshot_provider=provider)

    result = await executor.execute(skill)

    assert result.state == ExecutionState.SUCCEEDED
    assert [action.action_type for action in backend.actions] == ["open_app", "tap"]
    assert backend.actions[1].x == pytest.approx(421.4, abs=1.0)
    assert backend.actions[1].y == pytest.approx(988.4, abs=1.0)


@pytest.mark.asyncio
async def test_skill_executor_ignores_center_close_after_open_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import guiclaw.skills.executor as executor_module

    monkeypatch.setattr(executor_module, "_OPEN_APP_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(executor_module, "_POST_ACTION_SETTLE_SECONDS", 0.0)
    backend = _RecordingBackend()
    skill = _make_skill(
        "s1",
        "Open App",
        "Open App",
        steps=(
            SkillStep(
                action_type="open_app", target="com.example", valid_state="No need to verify"
            ),
        ),
    )
    provider = _ObservationProvider(
        [
            Observation(None, 496, 1080, foreground_app="com.example", platform="android"),
            Observation(
                None,
                496,
                1080,
                foreground_app="com.example",
                platform="android",
                extra={
                    "ui_tree": [
                        {"class": "FrameLayout", "enabled": True, "bounds": "[0,0][1440,3120]"},
                        {
                            "text": "关闭",
                            "class": "TextView",
                            "clickable": True,
                            "enabled": True,
                            "bounds": "[620,1400][820,1500]",
                        },
                    ],
                },
            ),
        ]
    )
    executor = SkillExecutor(backend=backend, screenshot_provider=provider)

    result = await executor.execute(skill)

    assert result.state == ExecutionState.SUCCEEDED
    assert [action.action_type for action in backend.actions] == ["open_app"]


@pytest.mark.asyncio
async def test_skill_extractor_parses_llm_json_response() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.android.settings", platform="android", name="open_settings", description="Open settings")
async def open_settings(device):
    await action("open_app", target="Settings", fixed=True, fixed_values={"text": "com.android.settings"}, valid_state="No need to verify")
    await action("tap", target="Search", fixed=True, fixed_values={"x": 100, "y": 200}, valid_state="Search button is visible")
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "com.android.settings"},
                "observation": {"platform": "android", "foreground_app": "com.android.settings"},
            },
            {
                "action": {"action_type": "done"},
                "observation": {"platform": "android", "foreground_app": "com.android.settings"},
            },
        ],
        is_success=True,
    )

    assert skill is not None
    assert skill.name == "open_settings"
    assert skill.steps[0].fixed is True
    assert skill.steps[1].action_type == "tap"


@pytest.mark.asyncio
async def test_skill_extractor_accepts_python_fence_after_explanation() -> None:
    response = """I will extract the reusable interaction as one compact skill.

```python
from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.android.settings", platform="android", name="open_settings", description="Open settings")
async def open_settings(device):
    await action("open_app", target="Settings", fixed=True, fixed_values={"text": "com.android.settings"}, valid_state="No need to verify")
```
"""
    llm = _ScriptedLLM([response])
    extractor = SkillExtractor(llm)

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "com.android.settings"},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.android.settings",
                },
            },
        ],
    )

    assert skill is not None
    assert skill.name == "open_settings"
    assert len(llm.messages) == 1


@pytest.mark.asyncio
async def test_skill_extractor_prompt_requests_stable_targets_and_generic_description() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="tv.danmaku.bili", platform="android", name="search_bilibili", description="In Bilibili, search for a video and start playback through the search results.")
async def search_bilibili(device, query):
    await action("open_app", target="Bilibili", fixed=True, fixed_values={"text": "tv.danmaku.bili"}, valid_state="No need to verify")
    await action("input_text", target=query, valid_state="search field is focused")
"""
    llm = _ScriptedLLM([response])
    extractor = SkillExtractor(llm)

    await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "tv.danmaku.bili"},
                "observation": {"platform": "android", "foreground_app": "tv.danmaku.bili"},
            },
            {
                "action": {"action_type": "input_text", "text": "敢杀我的马"},
                "observation": {"platform": "android", "foreground_app": "tv.danmaku.bili"},
            },
        ],
        is_success=False,
    )

    prompt = llm.messages[0][0]["content"][0]["text"]
    assert prompt.startswith("Return executable Python immediately.")
    assert "contains task-specific dynamic text" in prompt
    assert "omit state_contract from the generated code" in prompt
    assert "MUST be generic and reusable" in prompt
    assert "app name, capability, and broad feature-level route" in prompt
    assert "Use parameter roles" in prompt
    assert "NEVER include literal values/entities" in prompt
    assert "specific, official, first result, or top result" in prompt
    assert "Avoid tap-by-tap UI actions" in prompt
    assert "concise natural-language grounding hint" in prompt
    assert "Do not use raw class/resource_id as target" in prompt
    assert "prefer the trajectory app package" in prompt
    assert (
        "Every required interactive step must have a natural-language target and valid_state"
        in prompt
    )
    assert "input field is focused" in prompt
    assert "do not invent selectors" in prompt
    assert "postprocess will align contracts from codegen" in prompt
    assert "guarded optional=True steps" in prompt
    assert "at most one non-fixed corrective step" in prompt


@pytest.mark.asyncio
async def test_skill_extractor_fills_missing_valid_state_for_interactive_steps() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="search_example", description="Search in Example")
async def search_example(device, query):
    await action("tap", target="Search", fixed=True, fixed_values={"x": 100, "y": 200})
    await action("input_text", target=query)
"""
    llm = _ScriptedLLM([response])
    extractor = SkillExtractor(llm)

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 100, "y": 200},
                "observation": {"platform": "android", "foreground_app": "com.example"},
            },
            {
                "action": {"action_type": "input_text", "text": "query"},
                "observation": {"platform": "android", "foreground_app": "com.example"},
            },
        ],
    )

    assert skill is not None
    assert len(llm.messages) == 1
    assert skill.steps[0].valid_state == "Search is visible and enabled"
    assert skill.steps[1].valid_state == "input field is focused"


@pytest.mark.asyncio
async def test_skill_extractor_retries_when_required_target_is_missing() -> None:
    bad_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device):
    await action("tap", fixed=True, fixed_values={"x": 150, "y": 230})
"""
    fixed_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device):
    await action("tap", target="Details", fixed=True, fixed_values={"x": 150, "y": 230}, valid_state="Details is visible and enabled")
"""
    llm = _ScriptedLLM([bad_response, fixed_response])
    extractor = SkillExtractor(llm)

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 150, "y": 230},
                "observation": {"platform": "android", "foreground_app": "com.example"},
            },
        ],
    )

    assert skill is not None
    assert len(llm.messages) == 2
    assert "missing target" in llm.messages[1][0]["content"][0]["text"]
    assert skill.steps[0].target == "Details"
    assert skill.steps[0].valid_state == "Details is visible and enabled"


@pytest.mark.asyncio
async def test_skill_extractor_drops_skill_when_retry_still_has_missing_target() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device):
    await action("tap", fixed=True, fixed_values={"x": 150, "y": 230})
"""
    extractor = SkillExtractor(_ScriptedLLM([response, response]))

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 150, "y": 230},
                "observation": {"platform": "android", "foreground_app": "com.example"},
            },
        ],
    )

    assert skill is None


@pytest.mark.asyncio
async def test_skill_extractor_retries_when_fixed_action_is_invalid() -> None:
    bad_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_search", description="Open search")
async def open_search(device):
    await action("tap", target="Search", fixed=True, fixed_values={"text": "Search"}, valid_state="Search button is visible")
"""
    fixed_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_search", description="Open search")
async def open_search(device):
    await action("tap", target="Search", fixed=True, fixed_values={"x": 150, "y": 230}, valid_state="Search button is visible")
"""
    llm = _ScriptedLLM([bad_response, fixed_response])
    extractor = SkillExtractor(llm)

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 150, "y": 230},
                "observation": {"platform": "android", "foreground_app": "com.example"},
            },
        ],
    )

    assert skill is not None
    assert len(llm.messages) == 2
    retry_prompt = llm.messages[1][0]["content"][0]["text"]
    assert "fixed tap is invalid" in retry_prompt
    assert skill.steps[0].fixed_values == {"x": 150, "y": 230}


@pytest.mark.asyncio
async def test_skill_extractor_retries_when_flat_code_does_not_compile() -> None:
    bad_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device, item_name):
    await action("tap", target=f"{item_name} details", valid_state="details button is visible")
"""
    fixed_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device, item_name):
    await action("tap", target="{{item_name}} details", valid_state="details button is visible")
"""
    llm = _ScriptedLLM([bad_response, fixed_response])
    extractor = SkillExtractor(llm)

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 150, "y": 230},
                "observation": {"platform": "android", "foreground_app": "com.example"},
            },
        ],
    )

    assert skill is not None
    assert len(llm.messages) == 2
    retry_prompt = llm.messages[1][0]["content"][0]["text"]
    assert "did not compile as a flat GUI skill" in retry_prompt
    assert "Do not use f-strings" in retry_prompt
    assert skill.steps[0].target == "{{item_name}} details"


@pytest.mark.asyncio
async def test_skill_extractor_uses_8192_tokens_for_initial_and_retry_calls() -> None:
    compile_error = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device, item_name):
    await action("tap", target=f"{item_name} details", valid_state="details button is visible")
"""
    quality_error = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device):
    await action("tap", fixed=True, fixed_values={"x": 150, "y": 230})
"""
    valid_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device):
    await action("tap", target="Details", fixed=True, fixed_values={"x": 150, "y": 230}, valid_state="Details is visible and enabled")
"""
    llm = _ScriptedLLM([compile_error, quality_error, valid_response])
    extractor = SkillExtractor(llm)

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 150, "y": 230},
                "observation": {"platform": "android", "foreground_app": "com.example"},
            }
        ],
    )

    assert skill is not None
    assert llm.max_tokens == [8192, 8192, 8192]


@pytest.mark.asyncio
async def test_skill_extractor_generalizes_narrow_description_terms() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.google.android.youtube", platform="android", name="search_youtube", description="Opens YouTube, searches for a specific video query, selects the top result, and skips ads.")
async def search_youtube(device, query):
    await action("open_app", target="YouTube", fixed=True, fixed_values={"text": "YouTube"}, valid_state="No need to verify")
    await action("input_text", target=query, valid_state="search field is focused")
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "YouTube"},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.google.android.youtube",
                },
            },
            {
                "action": {"action_type": "input_text", "text": "Never Gonna Give You Up"},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.google.android.youtube",
                },
            },
        ],
    )

    assert skill is not None
    assert (
        skill.description
        == "Opens YouTube, searches for a video by query, selects the matching result, and skips ads."
    )


@pytest.mark.asyncio
async def test_skill_extractor_splits_segments_by_foreground_app() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.netease.cloudmusic", platform="android", name="open_netease", description="Open Netease")
async def open_netease(device):
    await action("open_app", target="Netease", fixed=True, fixed_values={"text": "com.netease.cloudmusic"}, valid_state="No need to verify")
    await action("tap", target="Daily", fixed=True, fixed_values={"x": 100, "y": 200}, valid_state="Daily icon is visible")
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skills = await _extract_many_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "com.netease.cloudmusic"},
                "observation": {"platform": "android", "foreground_app": "com.netease.cloudmusic"},
            },
            {
                "action": {"action_type": "tap", "x": 100, "y": 200},
                "observation": {"platform": "android", "foreground_app": "com.netease.cloudmusic"},
            },
        ],
        is_success=True,
    )

    assert len(skills) == 1
    assert skills[0].app == "com.netease.cloudmusic"
    assert skills[0].steps[1].action_type == "tap"


@pytest.mark.asyncio
async def test_skill_extractor_splits_file_trace_into_foreground_app_segments(
    tmp_path: Path,
) -> None:
    chrome_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.android.chrome", platform="android", name="search_web", description="Search in Chrome")
async def search_web(device):
    await action("tap", target="search field", valid_state="search field is visible")
"""
    calendar_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="org.fossify.calendar", platform="android", name="open_day", description="Open a calendar day")
async def open_day(device):
    await action("tap", target="calendar day", valid_state="calendar day is visible")
"""
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Search then schedule", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 100, "y": 100},
                "observation": {"platform": "android", "foreground_app": "com.android.chrome"},
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "input_text", "text": "graduation"},
                "observation": {"platform": "android", "foreground_app": "com.android.chrome"},
            },
            {
                "type": "step",
                "step_index": 2,
                "action": {"action_type": "tap", "x": 200, "y": 200},
                "observation": {"platform": "android", "foreground_app": "org.fossify.calendar"},
            },
            {
                "type": "step",
                "step_index": 3,
                "action": {"action_type": "tap", "x": 240, "y": 240},
                "observation": {"platform": "android", "foreground_app": "org.fossify.calendar"},
            },
        ],
    )
    llm = _ScriptedLLM([chrome_response, calendar_response])
    extractor = SkillExtractor(llm)

    skills = await extractor.extract_from_file_multi(trace_path)

    assert [skill.app for skill in skills] == ["com.android.chrome", "org.fossify.calendar"]
    assert len(llm.messages) == 2
    first_prompt = llm.messages[0][0]["content"][0]["text"]
    second_prompt = llm.messages[1][0]["content"][0]["text"]
    assert "App: com.android.chrome" in first_prompt
    assert "[0] tap" in first_prompt
    assert "[1] input_text" in first_prompt
    assert "[2] tap" not in first_prompt
    assert "App: org.fossify.calendar" in second_prompt
    assert "[2] tap" in second_prompt
    assert "[0] tap" not in second_prompt


@pytest.mark.asyncio
async def test_skill_extractor_skips_single_step_file_segments_before_llm(
    tmp_path: Path,
) -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="org.fossify.calendar", platform="android", name="open_day", description="Open a calendar day")
async def open_day(device):
    await action("tap", target="calendar day", valid_state="calendar day is visible")
"""
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Search then schedule", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 100, "y": 100},
                "observation": {"platform": "android", "foreground_app": "com.android.chrome"},
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "tap", "x": 200, "y": 200},
                "observation": {"platform": "android", "foreground_app": "org.fossify.calendar"},
            },
            {
                "type": "step",
                "step_index": 2,
                "action": {"action_type": "tap", "x": 240, "y": 240},
                "observation": {"platform": "android", "foreground_app": "org.fossify.calendar"},
            },
        ],
    )
    llm = _ScriptedLLM([response])
    extractor = SkillExtractor(llm)

    skills = await extractor.extract_from_file_multi(trace_path)

    assert [skill.app for skill in skills] == ["org.fossify.calendar"]
    assert len(llm.messages) == 1
    prompt = llm.messages[0][0]["content"][0]["text"]
    assert "App: org.fossify.calendar" in prompt
    assert "[0] tap" not in prompt
    assert extractor.last_diagnostics == []


@pytest.mark.asyncio
async def test_skill_extractor_does_not_carry_contracts_across_app_segments(
    tmp_path: Path,
) -> None:
    chrome_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.android.chrome", platform="android", name="search_web", description="Search in Chrome")
async def search_web(device):
    await action("tap", target="search field", valid_state="search field is visible")
"""
    calendar_response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="org.fossify.calendar", platform="android", name="open_day", description="Open a calendar day")
async def open_day(device):
    await action("tap", target="calendar day", valid_state="calendar day is visible",
                 state_contract=C(app="com.android.chrome", required=[R(resource_id="com.android.chrome:id/search_box", visible=True)]))
"""
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Search then schedule", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 100, "y": 100},
                "observation": {"platform": "android", "foreground_app": "com.android.chrome"},
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "tap", "x": 140, "y": 140},
                "observation": {"platform": "android", "foreground_app": "com.android.chrome"},
            },
            {
                "type": "step",
                "step_index": 2,
                "action": {"action_type": "tap", "x": 120, "y": 120},
                "observation": {
                    "platform": "android",
                    "foreground_app": "org.fossify.calendar",
                    "screen_width": 200,
                    "screen_height": 200,
                    "extra": {
                        "ui_tree": [
                            {
                                "resource_id": "org.fossify.calendar:id/day_button",
                                "class": "android.widget.TextView",
                                "clickable": True,
                                "enabled": True,
                                "bounds": "[80,80][200,200]",
                            }
                        ]
                    },
                },
            },
            {
                "type": "step",
                "step_index": 3,
                "action": {"action_type": "tap", "x": 180, "y": 180},
                "observation": {"platform": "android", "foreground_app": "org.fossify.calendar"},
            },
        ],
    )
    extractor = SkillExtractor(_ScriptedLLM([chrome_response, calendar_response]))

    skills = await extractor.extract_from_file_multi(trace_path)

    calendar_skill = [skill for skill in skills if skill.app == "org.fossify.calendar"][0]
    contract = calendar_skill.steps[0].state_contract
    assert contract is not None
    assert contract["anchor"]["app_package"] == "org.fossify.calendar"
    required = contract["signature"]["required"]
    assert required[0]["selector"] == {"resource_id": "org.fossify.calendar:id/day_button"}


@pytest.mark.asyncio
async def test_skill_extractor_resolves_unknown_app_from_open_app_text() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="unknown", platform="android", name="open_calendar", description="Open Calendar")
async def open_calendar(device):
    await action("open_app", target="Calendar", fixed=True, fixed_values={"text": "Calendar"}, valid_state="No need to verify")
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skills = await _extract_many_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "Calendar"},
                "observation": {"platform": "android"},
            },
        ],
        is_success=True,
    )

    assert len(skills) == 1
    assert skills[0].app == "org.fossify.calendar"


@pytest.mark.asyncio
async def test_skill_extractor_falls_back_to_trace_app_when_open_app_is_unknown() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="unknown", platform="android", name="open_calendar", description="Open Calendar")
async def open_calendar(device):
    await action("open_app", target="unknown", fixed=True, fixed_values={"text": "unknown"}, valid_state="No need to verify")
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skills = await _extract_many_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "unknown"},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.google.android.calendar",
                },
            },
        ],
        is_success=True,
    )

    assert len(skills) == 1
    assert skills[0].app == "org.fossify.calendar"


@pytest.mark.asyncio
async def test_skill_extractor_trace_app_overrides_llm_app_and_open_app_valid_state() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.android.settings", platform="android", name="open_calendar", description="Open Calendar")
async def open_calendar(device):
    await action("open_app", target="Settings", fixed=True, fixed_values={"text": "com.android.settings"}, valid_state="Settings screen is visible")
    await action("tap", target="Add event", fixed=True, fixed_values={"x": 100, "y": 200}, valid_state="Add event button is visible")
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skills = await _extract_many_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "org.fossify.calendar"},
                "observation": {"platform": "android", "foreground_app": "org.fossify.calendar"},
            },
            {
                "action": {"action_type": "tap", "x": 100, "y": 200},
                "observation": {"platform": "android", "foreground_app": "org.fossify.calendar"},
            },
        ],
        is_success=True,
    )

    assert len(skills) == 1
    assert skills[0].app == "org.fossify.calendar"
    assert skills[0].steps[0].action_type == "open_app"
    assert skills[0].steps[0].target == "org.fossify.calendar"
    assert skills[0].steps[0].fixed is True
    assert skills[0].steps[0].fixed_values["text"] == "org.fossify.calendar"
    assert skills[0].steps[0].valid_state == "No need to verify"


@pytest.mark.asyncio
async def test_skill_extractor_skips_trace_when_all_steps_share_foreground_app(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Search Zhihu", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 100, "y": 200},
                "observation": {"platform": "android", "foreground_app": "com.zhihu.android"},
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "input_text", "text": "query"},
                "observation": {"platform": "android", "foreground_app": "com.zhihu.android"},
            },
        ],
    )
    llm = _ScriptedLLM([])
    extractor = SkillExtractor(llm)

    skills = await extractor.extract_from_file_multi(trace_path)

    assert skills == []
    assert llm.messages == []
    assert extractor.last_diagnostics == []


@pytest.mark.asyncio
async def test_skill_extractor_rejects_unknown_app_without_fallback() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="unknown", platform="android", name="open_unknown", description="Open unknown")
async def open_unknown(device):
    await action("open_app", target="unknown", fixed=True, fixed_values={"text": "unknown"}, valid_state="No need to verify")
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skills = await _extract_many_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "open_app", "text": "unknown"},
                "observation": {"platform": "android"},
            },
        ],
        is_success=True,
    )

    assert skills == []


@pytest.mark.asyncio
async def test_skill_extractor_removes_llm_contract_without_codegen_contract() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.example", platform="android", name="open_details", description="Open details")
async def open_details(device):
    await action("tap", target="Details", fixed=True, fixed_values={"x": 150, "y": 230},
                 valid_state="Details button is visible",
                 state_contract=C(app="com.example", required=[R(resource_id="com.example:id/details_btn", visible=True, clickable=True)]))
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 150, "y": 230},
                "observation": {"platform": "android", "foreground_app": "com.example"},
            }
        ],
        is_success=True,
    )

    assert skill is not None
    assert skill.steps[0].state_contract is None


@pytest.mark.asyncio
async def test_skill_extractor_drops_llm_contract_when_step_trace_has_no_codegen_selector() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.zhihu.android", platform="android", name="search_box", description="Tap Zhihu search box")
async def search_box(device):
    await action("tap", target="Search box", fixed=True, fixed_values={"x": 436, "y": 76, "relative": True},
                 valid_state="Home screen is visible with search bar at the top",
                 state_contract=C(app="com.zhihu.android", required=[R(resource_id="com.zhihu.android:id/query_container", visible=True, clickable=True)]))
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 436.0, "y": 76.0, "relative": True},
                "observation": {"platform": "android", "foreground_app": "com.zhihu.android"},
            }
        ],
        is_success=True,
    )

    assert skill is not None
    assert skill.steps[0].state_contract is None


@pytest.mark.asyncio
async def test_skill_extractor_drops_later_llm_contract_without_codegen_selector() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.zhihu.android", platform="android", name="search_box_recover", description="Tap search box")
async def search_box_recover(device):
    await action("tap", target="Home", fixed=True, fixed_values={"x": 607, "y": 243, "relative": True}, valid_state="No need to verify")
    await action("tap", target="Search box", fixed=True, fixed_values={"x": 436, "y": 76, "relative": True},
                 valid_state="Search bar is visible",
                 state_contract=C(app="com.zhihu.android", required=[R(resource_id="com.zhihu.android:id/query_container", visible=True, clickable=True)]))
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "tap", "x": 607.0, "y": 243.0, "relative": True},
                "observation": {"platform": "android", "foreground_app": "com.zhihu.android"},
            },
            {
                "action": {"action_type": "tap", "x": 436.0, "y": 76.0, "relative": True},
                "observation": {"platform": "android", "foreground_app": "com.zhihu.android"},
            },
        ],
        is_success=True,
    )

    assert skill is not None
    assert skill.steps[1].state_contract is None


@pytest.mark.asyncio
async def test_skill_extractor_drops_input_llm_contract_without_codegen_selector() -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.zhihu.android", platform="android", name="zhihu_search_text", description="Enter search text")
async def zhihu_search_text(device):
    await action("input_text", target="com.zhihu.android:id/input_text",
                 fixed_values={"text": "强化学习"},
                 valid_state="Search input field is visible and focused",
                 state_contract=C(app="com.zhihu.android", required=[R(resource_id="com.zhihu.android:id/input_text", visible=True, enabled=True)]))
"""
    extractor = SkillExtractor(_ScriptedLLM([response]))

    skill = await _extract_from_steps(
        extractor,
        [
            {
                "action": {"action_type": "input_text", "text": "强化学习"},
                "observation": {"platform": "android", "foreground_app": "com.zhihu.android"},
            },
        ],
        is_success=True,
    )

    assert skill is not None
    assert skill.steps[0].action_type == "input_text"
    assert skill.steps[0].state_contract is None


def test_infer_focused_input_contract_prefers_resource_id_and_class() -> None:
    contract = infer_focused_input_contract(_focused_input_extra(), app="com.zhihu.android")

    assert contract is not None
    required = contract["signature"]["required"]
    assert required[0]["selector"] == {
        "resource_id": "com.zhihu.android:id/input_text",
        "class": "android.widget.EditText",
    }
    assert required[0]["state"] == ["visible", "enabled", "focused"]


def test_codegen_scales_coordinates_and_prefers_previous_observation_for_tap(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Open first result", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "wait"},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.example",
                    "screen_width": 496,
                    "screen_height": 1080,
                    "extra": {
                        "ui_tree": [
                            {
                                "class": "android.widget.FrameLayout",
                                "enabled": True,
                                "bounds": "[0,0][1440,3120]",
                            },
                            {
                                "resource_id": "com.example:id/search_results_list",
                                "class": "androidx.recyclerview.widget.RecyclerView",
                                "enabled": True,
                                "bounds": "[0,460][1440,3036]",
                            },
                            {
                                "class": "android.view.ViewGroup",
                                "clickable": True,
                                "enabled": True,
                                "bounds": "[0,460][1440,1024]",
                            },
                        ],
                    },
                },
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "tap", "x": 248, "y": 235},
                "interaction_target": {
                    "state_contract": {
                        "anchor": {"app_package": "com.example"},
                        "signature": {
                            "required": [
                                {
                                    "selector": {
                                        "resource_id": "com.example:id/search_results_list"
                                    },
                                    "state": ["visible", "clickable"],
                                }
                            ],
                            "forbidden": [],
                        },
                        "mask_rules": [],
                    }
                },
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.example",
                    "screen_width": 496,
                    "screen_height": 1080,
                    "extra": {
                        "ui_tree": [
                            {
                                "class": "android.widget.FrameLayout",
                                "enabled": True,
                                "bounds": "[0,0][1440,3120]",
                            },
                            {
                                "text": "query",
                                "resource_id": "com.example:id/search_fake_text",
                                "class": "android.widget.TextView",
                                "clickable": True,
                                "enabled": True,
                                "bounds": "[225,179][1030,284]",
                            },
                        ],
                    },
                },
            },
        ],
    )

    result = codegen_trajectory(trace_path)

    assert result is not None
    contract = json.loads(result.steps[1].contract_json)
    required = contract["signature"]["required"]
    assert required[0]["selector"] == {"resource_id": "com.example:id/search_results_list"}


def test_codegen_ignores_launcher_when_inferring_trace_app(tmp_path: Path) -> None:
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Send Mail", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "home"},
                "observation": {"platform": "android", "foreground_app": "nexuslauncher"},
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "tap", "x": 100, "y": 200},
                "observation": {"platform": "android", "foreground_app": "gmailclone"},
            },
        ],
    )

    result = codegen_trajectory(trace_path)

    assert result is not None
    assert result.app == "com.gmailclone"
    assert result.app_candidates == ("com.gmailclone",)


def test_codegen_uses_resolved_app_from_compact_step(tmp_path: Path) -> None:
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "CartManagementTask", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 100, "y": 200},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.testmall.app",
                },
            },
        ],
    )

    result = codegen_trajectory(trace_path)

    assert result is not None
    assert result.app == "com.testmall.app"
    assert result.app_candidates == ("com.testmall.app",)


def test_compact_codegen_does_not_reconstruct_target_from_post_action_ui_tree(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Tap search", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 206, "y": 81},
                "observation": {
                    "platform": "android",
                    "screen_width": 496,
                    "screen_height": 1080,
                    "extra": {
                        "ui_tree": [
                            {
                                "class": "android.widget.FrameLayout",
                                "enabled": True,
                                "bounds": "[0,0][1440,3120]",
                            },
                            {
                                "resource_id": "tv.danmaku.bili:id/search_bar",
                                "class": "android.widget.FrameLayout",
                                "enabled": True,
                                "bounds": "[225,172][1156,291]",
                            },
                            {
                                "text": "Search for videos, series, or UPs",
                                "content_desc": "Search query",
                                "resource_id": "tv.danmaku.bili:id/search_src_text",
                                "class": "android.widget.EditText",
                                "clickable": True,
                                "focused": True,
                                "enabled": True,
                                "bounds": "[225,179][1156,284]",
                            },
                        ],
                    },
                },
            },
        ],
    )

    result = codegen_trajectory(trace_path)

    assert result is not None
    assert result.steps[0].control_info == ""
    assert result.steps[0].contract_json == ""


@pytest.mark.asyncio
async def test_skill_extractor_overrides_llm_contract_with_codegen_contract(
    tmp_path: Path,
) -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="tv.danmaku.bili", platform="android", name="search_bilibili", description="Search Bilibili")
async def search_bilibili(device, query):
    await action("open_app", target="Bilibili", valid_state="No need to verify")
    await action("tap", target="search bar", valid_state="search bar is visible",
                 state_contract=C(app="tv.danmaku.bili", required=[R(resource_id="tv.danmaku.bili:id/hallucinated", visible=True, clickable=True)]))
    await action("input_text", target=query, valid_state="search input is focused")
"""
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Search Bilibili", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 400, "y": 800},
                "interaction_target": {
                    "state_contract": {
                        "anchor": {"app_package": "tv.danmaku.bili"},
                        "signature": {
                            "required": [
                                {
                                    "selector": {"resource_id": "tv.danmaku.bili:id/expand_search"},
                                    "state": ["visible", "clickable"],
                                }
                            ],
                            "forbidden": [],
                        },
                        "mask_rules": [],
                    }
                },
                "observation": {
                    "platform": "android",
                    "foreground_app": "tv.danmaku.bili",
                    "screen_width": 496,
                    "screen_height": 1080,
                    "extra": {
                        "ui_tree": [
                            {
                                "class": "android.widget.FrameLayout",
                                "enabled": True,
                                "bounds": "[0,0][1440,3120]",
                            },
                            {
                                "content_desc": "Search bar, button",
                                "resource_id": "tv.danmaku.bili:id/expand_search",
                                "class": "android.widget.LinearLayout",
                                "clickable": True,
                                "enabled": True,
                                "bounds": "[249,182][1076,301]",
                            },
                        ],
                    },
                },
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "tap", "x": 206, "y": 81},
                "observation": {
                    "platform": "android",
                    "foreground_app": "tv.danmaku.bili",
                    "screen_width": 496,
                    "screen_height": 1080,
                    "extra": {
                        "ui_tree": [
                            {
                                "class": "android.widget.FrameLayout",
                                "enabled": True,
                                "bounds": "[0,0][1440,3120]",
                            },
                            {
                                "resource_id": "tv.danmaku.bili:id/search_src_text",
                                "class": "android.widget.EditText",
                                "clickable": True,
                                "focused": True,
                                "enabled": True,
                                "bounds": "[225,179][1156,284]",
                            },
                        ],
                    },
                },
            },
            {
                "type": "step",
                "step_index": 2,
                "action": {"action_type": "input_text", "text": "Never Gonna Give You Up MV"},
                "observation": {
                    "platform": "android",
                    "extra": {"ui_tree": []},
                },
            },
        ],
    )
    extractor = SkillExtractor(_ScriptedLLM([response]))

    codegen = codegen_trajectory(trace_path)
    assert codegen is not None
    skills = await extractor.extract_from_codegen_result_multi(codegen)
    skill = skills[0] if skills else None

    assert skill is not None
    tap_step = [step for step in skill.steps if step.action_type == "tap"][0]
    assert tap_step.state_contract is not None
    tap_required = tap_step.state_contract["signature"]["required"]
    assert tap_required[0]["selector"] == {"resource_id": "tv.danmaku.bili:id/expand_search"}
    input_step = [step for step in skill.steps if step.action_type == "input_text"][0]
    assert input_step.state_contract is not None
    input_required = input_step.state_contract["signature"]["required"]
    assert input_required[0]["selector"]["resource_id"] == "tv.danmaku.bili:id/search_src_text"
    assert "focused" in input_required[0]["state"]


@pytest.mark.asyncio
async def test_skill_extractor_injects_focused_input_contract_when_llm_omits_it(
    tmp_path: Path,
) -> None:
    response = """from guiclaw.skills.flat import C, R, action, skill

@skill(app="com.zhihu.android", platform="android", name="zhihu_search_text", description="Enter search text")
async def zhihu_search_text(device, query):
    await action("tap", target="Search box", valid_state="Search box is visible")
    await action("input_text", target=query, valid_state="Search input field is focused")
"""
    trace_path = tmp_path / "traj.json"
    _write_compact_trajectory(
        trace_path,
        [
            {"type": "metadata", "task": "Search Zhihu", "platform": "android"},
            {
                "type": "step",
                "step_index": 0,
                "action": {"action_type": "tap", "x": 100, "y": 40},
                "observation": {
                    "platform": "android",
                    "foreground_app": "com.zhihu.android",
                    "extra": _focused_input_extra(),
                },
            },
            {
                "type": "step",
                "step_index": 1,
                "action": {"action_type": "input_text", "text": "强化学习"},
                "observation": {
                    "platform": "android",
                    "extra": {"ui_tree": []},
                },
            },
        ],
    )

    extractor = SkillExtractor(_ScriptedLLM([response]))
    codegen = codegen_trajectory(trace_path)
    assert codegen is not None
    skills = await extractor.extract_from_codegen_result_multi(codegen)
    skill = skills[0] if skills else None

    assert skill is not None
    input_step = [step for step in skill.steps if step.action_type == "input_text"][0]
    required = input_step.state_contract["signature"]["required"]
    assert required[0]["selector"] == {
        "resource_id": "com.zhihu.android:id/input_text",
        "class": "android.widget.EditText",
    }
    assert required[0]["state"] == ["visible", "enabled", "focused"]
