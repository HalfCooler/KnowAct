"""Unit tests for SkillExecutor wiring into GuiSubagentTool via enable_skill_execution config.

Tests verify:
- GuiConfig accepts enable_skill_execution=True and defaults to False.
- When enable_skill_execution=False, GuiAgent receives skill_executor=None.
- When enable_skill_execution=True, GuiAgent receives a SkillExecutor instance.
- The SkillExecutor is built with backend=active_backend and
  state_validator=LLMStateValidator(llm_adapter).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import GuiConfig

if TYPE_CHECKING:
    from nanobot.agent.tools.gui import GuiSubagentTool


# ---------------------------------------------------------------------------
# GuiConfig field tests
# ---------------------------------------------------------------------------


class TestGuiConfigSkillExecutionField:
    """GuiConfig.enable_skill_execution must default to False and accept True."""

    def test_defaults_to_false(self) -> None:
        config = GuiConfig()
        assert config.enable_skill_execution is False

    def test_accepts_true(self) -> None:
        config = GuiConfig(enable_skill_execution=True)
        assert config.enable_skill_execution is True

    def test_accepts_false_explicitly(self) -> None:
        config = GuiConfig(enable_skill_execution=False)
        assert config.enable_skill_execution is False

    def test_accepts_camel_case_key(self) -> None:
        """GuiConfig uses camelCase aliases; enableSkillExecution must be accepted."""
        config = GuiConfig.model_validate({"enableSkillExecution": True})
        assert config.enable_skill_execution is True


class TestGuiConfigSkillExtractionField:
    """GuiConfig.enable_skill_extraction must default to False and accept True."""

    def test_defaults_to_false(self) -> None:
        config = GuiConfig()
        assert config.enable_skill_extraction is False

    def test_accepts_true(self) -> None:
        config = GuiConfig(enable_skill_extraction=True)
        assert config.enable_skill_extraction is True

    def test_accepts_false_explicitly(self) -> None:
        config = GuiConfig(enable_skill_extraction=False)
        assert config.enable_skill_extraction is False

    def test_accepts_camel_case_key(self) -> None:
        """GuiConfig uses camelCase aliases; enableSkillExtraction must be accepted."""
        config = GuiConfig.model_validate({"enableSkillExtraction": True})
        assert config.enable_skill_extraction is True


class TestGuiConfigHasNoDesktopSkillOverride:
    """Desktop skill support is a platform constraint, not a user switch."""

    def test_schema_has_no_desktop_skill_override(self) -> None:
        assert "enable_desktop_skills" not in GuiConfig.model_fields

    def test_legacy_camel_case_key_is_ignored(self) -> None:
        config = GuiConfig.model_validate({"enableDesktopSkills": True})
        assert not hasattr(config, "enable_desktop_skills")


class TestGuiConfigMemoryExtractionField:
    """GuiConfig.enable_memory_extraction must be independently opt-in."""

    def test_defaults_to_false(self) -> None:
        config = GuiConfig()
        assert config.enable_memory_extraction is False

    def test_accepts_true(self) -> None:
        config = GuiConfig(enable_memory_extraction=True)
        assert config.enable_memory_extraction is True

    def test_accepts_camel_case_key(self) -> None:
        config = GuiConfig.model_validate({"enableMemoryExtraction": True})
        assert config.enable_memory_extraction is True

    def test_is_wired_to_postprocessor(self) -> None:
        tool = _make_tool(GuiConfig(backend="dry-run", enable_memory_extraction=True))
        assert tool._postprocessor._enable_memory_extraction is True


# ---------------------------------------------------------------------------
# GuiSubagentTool wiring tests
# ---------------------------------------------------------------------------


def _make_gui_config(enable_skill_execution: bool) -> GuiConfig:
    return GuiConfig(backend="dry-run", enable_skill_execution=enable_skill_execution)


def _make_tool(gui_config: GuiConfig) -> "GuiSubagentTool":
    """Build a GuiSubagentTool with mocked provider/model/workspace."""
    from nanobot.agent.tools.gui import GuiSubagentTool

    provider = MagicMock()
    provider.api_key = "test-key"
    provider.api_base = None
    provider.extra_headers = None
    # Simulate the _client attribute used by NanobotLLMAdapter
    provider._client = None

    return GuiSubagentTool(
        gui_config=gui_config,
        provider=provider,
        model="test/model",
        workspace=Path("/tmp/test_workspace"),
    )


class TestDesktopSkillGate:
    def test_host_disables_desktop_skill_runtime_and_extraction_by_default(self) -> None:
        from nanobot.agent.tools.gui import GuiSubagentTool

        backend = MagicMock(platform="macos")
        captured_kwargs: dict = {}
        config = GuiConfig(
            backend="local",
            enable_skill_execution=True,
            enable_prompt_skill_selection=True,
            enable_skill_extraction=True,
        )
        with (
            patch.object(GuiSubagentTool, "_build_backend", return_value=backend),
            patch.object(GuiSubagentTool, "_get_skill_library") as get_library,
        ):
            tool = _make_tool(config)

            async def _run() -> None:
                with (
                    patch(
                        "nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None
                    ) as mock_init,
                    patch(
                        "nanobot.agent.tools.gui.TrajectoryRecorder",
                        return_value=MagicMock(path=None),
                    ),
                    patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run,
                ):
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    try:
                        await tool._run_task(backend, "open notes")
                    except (TypeError, ValueError):
                        pass

            asyncio.run(_run())

        get_library.assert_not_called()
        assert tool._skill_library is None
        assert tool._postprocessor._enable_skill_extraction is False
        assert captured_kwargs["skill_executor"] is None
        assert captured_kwargs["enable_prompt_skill_selection"] is False

    def test_legacy_desktop_skill_opt_in_cannot_enable_host_skills(self) -> None:
        from nanobot.agent.tools.gui import GuiSubagentTool

        backend = MagicMock(platform="windows")
        config = GuiConfig.model_validate(
            {
                "backend": "local",
                "enableSkillExecution": True,
                "enableSkillExtraction": True,
                "enableDesktopSkills": True,
            }
        )
        with (
            patch.object(GuiSubagentTool, "_build_backend", return_value=backend),
            patch.object(GuiSubagentTool, "_get_skill_library") as get_library,
        ):
            tool = _make_tool(config)

        get_library.assert_not_called()
        assert tool._skill_library is None
        assert tool._postprocessor._enable_skill_extraction is False


class TestSkillExecutorWiringDisabled:
    """When enable_skill_execution=False, GuiAgent must receive skill_executor=None."""

    def test_skill_executor_is_none_when_disabled(self) -> None:
        gui_config = _make_gui_config(enable_skill_execution=False)
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
            ):
                # Patch run so we don't actually execute the agent.
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(tool._backend, "open settings")
                    except Exception:
                        pass  # GuiAgent.__init__ is mocked, run may raise

        asyncio.run(_run())

        # skill_executor must be None (or absent, which defaults to None)
        assert captured_kwargs.get("skill_executor") is None

    def test_skill_library_lookup_is_skipped_when_execution_disabled(self) -> None:
        gui_config = _make_gui_config(enable_skill_execution=False)
        tool = _make_tool(gui_config)

        async def _run() -> None:
            with (
                patch.object(tool, "_refresh_cached_skill_stores") as refresh_mock,
                patch.object(tool, "_get_skill_library") as get_lib_mock,
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None),
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
                patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run,
            ):
                mock_run.return_value = MagicMock(
                    success=True,
                    summary="ok",
                    model_summary="",
                    trace_path=None,
                    steps_taken=0,
                    error=None,
                )
                try:
                    await tool._run_task(tool._backend, "open settings")
                except Exception:
                    pass
                refresh_mock.assert_not_called()
                get_lib_mock.assert_not_called()

        asyncio.run(_run())


class TestShortcutDiscoveryBackend:
    """Shortcut discovery requires an ADB-like command runner."""

    def test_adb_like_backend_is_still_allowed_for_runtime_shortcut_discovery(self) -> None:
        from nanobot.agent.tools.gui import GuiSubagentTool

        class AdbLikeBackend:
            async def _run(self, *args: str, timeout: float = 10.0) -> str:
                return ""

        backend = AdbLikeBackend()

        assert GuiSubagentTool._shortcut_discovery_backend(backend) is backend


class TestSkillExecutorWiringEnabled:
    """When enable_skill_execution=True, GuiAgent must receive a SkillExecutor instance."""

    def test_skill_executor_is_passed_when_enabled(self) -> None:
        from guiclaw.skills.executor import SkillExecutor

        gui_config = _make_gui_config(enable_skill_execution=True)
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
            ):
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(tool._backend, "open settings")
                    except Exception:
                        pass

        asyncio.run(_run())

        assert "skill_executor" in captured_kwargs
        assert isinstance(captured_kwargs["skill_executor"], SkillExecutor)

    def test_skill_executor_built_with_correct_backend(self) -> None:
        """SkillExecutor.backend must be the active_backend passed to _run_task."""
        from guiclaw.skills.executor import SkillExecutor

        gui_config = _make_gui_config(enable_skill_execution=True)
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}
        active_backend = tool._backend  # dry-run backend

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
            ):
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(active_backend, "open settings")
                    except Exception:
                        pass

        asyncio.run(_run())

        skill_executor = captured_kwargs.get("skill_executor")
        assert isinstance(skill_executor, SkillExecutor)
        assert skill_executor.backend is active_backend

    def test_skill_executor_built_with_llm_state_validator(self) -> None:
        """SkillExecutor.state_validator must be an LLMStateValidator backed by the LLM adapter."""
        from guiclaw.skills.executor import LLMStateValidator, SkillExecutor

        gui_config = _make_gui_config(enable_skill_execution=True)
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
            ):
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(tool._backend, "open settings")
                    except Exception:
                        pass

        asyncio.run(_run())

        skill_executor = captured_kwargs.get("skill_executor")
        assert isinstance(skill_executor, SkillExecutor)
        assert isinstance(skill_executor.state_validator, LLMStateValidator)
        # Verify the state_validator is backed by the tool's LLM adapter
        assert skill_executor.state_validator._llm is tool._llm_adapter

    def test_skill_executor_and_subgoal_runner_share_live_recorder(self) -> None:
        from guiclaw.skills.executor import SkillExecutor

        gui_config = _make_gui_config(enable_skill_execution=True)
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}
        recorder = MagicMock(path=None)

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch("nanobot.agent.tools.gui.TrajectoryRecorder", return_value=recorder),
            ):
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(tool._backend, "open settings")
                    except Exception:
                        pass

        asyncio.run(_run())

        skill_executor = captured_kwargs.get("skill_executor")
        assert isinstance(skill_executor, SkillExecutor)
        assert skill_executor.trajectory_recorder is recorder
        assert getattr(skill_executor.subgoal_runner, "_trajectory_recorder", None) is recorder

    def test_image_scale_ratio_is_forwarded_to_skill_components(self) -> None:
        from guiclaw.skills.executor import SkillExecutor

        gui_config = GuiConfig(
            backend="dry-run",
            enable_skill_execution=True,
            image_scale_ratio=0.25,
        )
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
            ):
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(tool._backend, "open settings")
                    except Exception:
                        pass

        asyncio.run(_run())

        skill_executor = captured_kwargs.get("skill_executor")
        assert isinstance(skill_executor, SkillExecutor)
        assert getattr(skill_executor.state_validator, "_image_scale_ratio", None) == pytest.approx(0.25)
        assert getattr(skill_executor.action_grounder, "_image_scale_ratio", None) == pytest.approx(0.25)
        assert getattr(skill_executor.subgoal_runner, "_image_scale_ratio", None) == pytest.approx(0.25)


class TestGuiAgentProfileWiring:
    """Configured gui.agent_profile must flow through to the GUI agent chain."""

    def test_agent_profile_is_forwarded_to_agent(self) -> None:
        gui_config = GuiConfig(
            backend="dry-run",
            enable_skill_execution=True,
            agent_profile="qwen3vl",
        )
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
            ):
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(tool._backend, "open settings")
                    except Exception:
                        pass

        asyncio.run(_run())

        assert captured_kwargs["agent_profile"] == "qwen3vl"

    def test_image_scale_ratio_is_forwarded_to_agent(self) -> None:
        gui_config = GuiConfig(
            backend="dry-run",
            enable_skill_execution=True,
            image_scale_ratio=0.4,
        )
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
            ):
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(tool._backend, "open settings")
                    except Exception:
                        pass

        asyncio.run(_run())

        assert captured_kwargs["image_scale_ratio"] == pytest.approx(0.4)

    def test_stagnation_limit_is_forwarded_to_agent(self) -> None:
        gui_config = GuiConfig(
            backend="dry-run",
            enable_skill_execution=True,
            stagnation_limit=3,
        )
        tool = _make_tool(gui_config)

        captured_kwargs: dict = {}

        async def _run() -> None:
            with (
                patch("nanobot.agent.tools.gui.GuiAgent.__init__", return_value=None) as mock_init,
                patch(
                    "nanobot.agent.tools.gui.TrajectoryRecorder",
                    return_value=MagicMock(path=None),
                ),
            ):
                with patch("guiclaw.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = MagicMock(
                        success=True,
                        summary="ok",
                        model_summary="",
                        trace_path=None,
                        steps_taken=0,
                        error=None,
                    )
                    mock_init.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
                    try:
                        await tool._run_task(tool._backend, "open settings")
                    except Exception:
                        pass

        asyncio.run(_run())

        assert captured_kwargs["stagnation_limit"] == 3
