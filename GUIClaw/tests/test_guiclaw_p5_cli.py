"""Phase 5 CLI regression tests.

Task 1 seeds the CLI contract as xfailed tests. Task 2 removes the xfail mark
and promotes this file to passing coverage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import runpy
import sys
import textwrap
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from guiclaw.backends.dry_run import _TINY_PNG


@pytest.fixture(autouse=True)
def _isolate_default_gui_runs_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import guiclaw.cli as cli

    monkeypatch.setattr(cli, "DEFAULT_GUI_RUNS_DIR", tmp_path / "gui_runs")


def test_cli_tests_isolate_default_gui_runs_dir(tmp_path: Path) -> None:
    import guiclaw.cli as cli

    assert cli.DEFAULT_GUI_RUNS_DIR == tmp_path / "gui_runs"


def _write_config(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


class _FakeBackend:
    def __init__(self, platform: str = "dry-run") -> None:
        self._platform = platform
        self.preflight_calls = 0

    @property
    def platform(self) -> str:
        return self._platform

    async def preflight(self) -> None:
        self.preflight_calls += 1


class _ScriptedCliProvider:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        del messages, tools, tool_choice, model, max_tokens
        if not self._responses:
            raise AssertionError("No scripted CLI responses left.")
        return _profile_response(self._responses.pop(0))


def _profile_response(response: Any) -> Any:
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return response
    action = dict(tool_calls[0].arguments)
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


class _InterventionCliBackend:
    def __init__(self) -> None:
        from guiclaw.observation import Observation

        self._Observation = Observation
        self.platform = "linux"
        self.observe_calls: list[Path] = []
        self._observations = [
            {
                "foreground_app": "Secure Login",
                "extra": {"display_id": ":77", "session_token": "secret-session-token"},
            },
            {
                "foreground_app": "Authenticated Workspace",
                "extra": {"display_id": ":77"},
            },
        ]

    async def preflight(self) -> None:
        return None

    async def execute(self, *_: Any, **__: Any) -> str:
        return "execute should not run"

    async def list_apps(self) -> list[str]:
        return []

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Any:
        del timeout
        self.observe_calls.append(screenshot_path)
        if not self._observations:
            raise AssertionError("No scripted observations left.")
        payload = self._observations.pop(0)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(_TINY_PNG)
        return self._Observation(
            screenshot_path=str(screenshot_path),
            screen_width=1280,
            screen_height=720,
            foreground_app=payload["foreground_app"],
            platform=self.platform,
            extra=payload["extra"],
        )

    def get_intervention_target(self) -> dict[str, Any]:
        return {
            "display_id": ":77",
            "monitor_index": 2,
            "width": 1440,
            "height": 900,
            "platform": "linux",
            "session_token": "secret-session-token",
        }


def test_cli_parses_task_and_backend_flags() -> None:
    import guiclaw.cli as cli

    positional = cli.parse_args(["Open Settings"])
    assert cli.resolve_task(positional) == "Open Settings"
    assert cli.resolve_backend_name(positional) == "local"

    flagged = cli.parse_args(["--task", "Open Settings", "--backend", "adb"])
    assert cli.resolve_task(flagged) == "Open Settings"
    assert cli.resolve_backend_name(flagged) == "adb"

    dry_run = cli.parse_args(["--task", "Open Settings", "--backend", "adb", "--dry-run"])
    assert cli.resolve_backend_name(dry_run) == "dry-run"

    profiled = cli.parse_args(["--task", "Open Settings", "--agent-profile", "qwen3vl"])
    assert profiled.agent_profile == "qwen3vl"

    with pytest.raises(SystemExit):
        cli.parse_args([])


def test_load_config_env_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import guiclaw.cli as cli

    default_config = _write_config(
        tmp_path / ".guiclaw" / "config.yaml",
        """
        agent_profile: gelab
        provider:
          base_url: http://localhost:1234/v1
          model: qwen-gui
        """,
    )
    monkeypatch.setattr(cli, "DEFAULT_CONFIG_PATH", default_config)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    cfg = cli.load_config()
    assert cfg.agent_profile == "gelab"
    assert cfg.provider.base_url == "http://localhost:1234/v1"
    assert cfg.provider.model == "qwen-gui"
    assert cfg.provider.api_key == "env-key"
    assert cfg.image_scale_ratio == pytest.approx(0.5)
    assert cfg.stagnation_limit == 0
    assert cfg.enable_skill_execution is False
    assert cfg.enable_skill_extraction is False
    assert cfg.enable_memory_extraction is False

    custom_config = _write_config(
        tmp_path / "custom.yaml",
        """
        provider:
          base_url: http://localhost:9999/v1
          model: qwen-custom
          api_key: inline-key
        adb:
          serial: emulator-5554
          adb_path: /tmp/adb
        """,
    )
    override = cli.load_config(custom_config)
    assert override.provider.base_url == "http://localhost:9999/v1"
    assert override.provider.model == "qwen-custom"
    assert override.provider.api_key == "inline-key"
    assert override.adb.serial == "emulator-5554"
    assert override.adb.adb_path == "/tmp/adb"

    scaled_config = _write_config(
        tmp_path / "scaled.yaml",
        """
        provider:
          base_url: http://localhost:9999/v1
          model: qwen-custom
        image_scale_ratio: 0.25
        stagnation_limit: 3
        enable_skill_execution: true
        enable_skill_extraction: true
        enable_memory_extraction: true
        """,
    )
    scaled = cli.load_config(scaled_config)
    assert scaled.image_scale_ratio == pytest.approx(0.25)
    assert scaled.stagnation_limit == 3
    assert scaled.enable_skill_execution is True
    assert scaled.enable_skill_extraction is True
    assert scaled.enable_memory_extraction is True


def test_build_backend_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    import guiclaw.cli as cli

    calls: dict[str, list[dict[str, Any]]] = {"adb": [], "local": [], "dry-run": []}

    class FakeAdbBackend:
        def __init__(self, serial: str | None = None, adb_path: str = "adb") -> None:
            calls["adb"].append({"serial": serial, "adb_path": adb_path})

    class FakeLocalDesktopBackend:
        def __init__(self) -> None:
            calls["local"].append({})

    class FakeDryRunBackend:
        def __init__(self) -> None:
            calls["dry-run"].append({})

    monkeypatch.setattr(cli, "AdbBackend", FakeAdbBackend)
    monkeypatch.setattr(cli, "LocalDesktopBackend", FakeLocalDesktopBackend)
    monkeypatch.setattr(cli, "DryRunBackend", FakeDryRunBackend)

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        ),
        adb=cli.AdbConfig(serial="emulator-5554", adb_path="/tmp/adb"),
    )

    assert isinstance(cli.build_backend("adb", config), FakeAdbBackend)
    assert isinstance(cli.build_backend("local", config), FakeLocalDesktopBackend)
    assert isinstance(cli.build_backend("dry-run", config), FakeDryRunBackend)
    assert calls["adb"] == [{"serial": "emulator-5554", "adb_path": "/tmp/adb"}]


def test_cli_runs_dry_run_agent_loop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import guiclaw.cli as cli
    from guiclaw.agent import AgentResult

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    backend = _FakeBackend()
    recorder_state: dict[str, Any] = {}
    agent_state: dict[str, Any] = {}

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir
            self.task = task
            self.platform = platform
            recorder_state["output_dir"] = output_dir
            recorder_state["task"] = task
            recorder_state["platform"] = platform

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            agent_state.update(kwargs)
            self._progress_callback = kwargs["progress_callback"]

        async def run(self, task: str, **_: Any) -> AgentResult:
            await self._progress_callback("GUI step 1/15: inspect screen")
            return AgentResult(
                success=True,
                summary=f"Completed {task}",
                model_summary="Opened app and confirmed final state.",
                trace_path="trace.jsonl",
                steps_taken=2,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_backend", lambda backend_name, cfg: backend)
    monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)

    exit_code = cli.main(["--dry-run", "--task", "Open Settings", "--agent-profile", "seed"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "GUI step 1/15: inspect screen" in captured.out
    assert "status: success" in captured.out
    assert "success: true" in captured.out
    assert "summary: Completed Open Settings" in captured.out
    assert "model_summary: Opened app and confirmed final state." in captured.out
    assert "trace_path: trace.jsonl" in captured.out
    assert "steps_taken: 2" in captured.out
    assert recorder_state["task"] == "Open Settings"
    assert recorder_state["platform"] == "dry-run"
    assert recorder_state["output_dir"].parent == cli.DEFAULT_GUI_RUNS_DIR
    assert agent_state["backend"] is backend
    assert agent_state["model"] == "qwen-gui"
    assert agent_state["agent_profile"] == "seed"
    assert agent_state["artifacts_root"] == recorder_state["output_dir"]
    assert agent_state["stagnation_limit"] == 0
    assert agent_state["enable_prompt_skill_selection"] is False


def test_standalone_cli_runs_enabled_postprocessing_before_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import guiclaw.cli as cli
    from guiclaw.agent import AgentResult

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        ),
        embedding=cli.EmbeddingConfig(
            base_url="http://localhost:5678/v1",
            model="embed-model",
            api_key="embed-key",
        ),
        memory_dir=tmp_path / "memory",
        skills_dir=tmp_path / "skill",
        enable_skill_execution=True,
        enable_skill_extraction=True,
        enable_memory_extraction=True,
    )
    backend = _FakeBackend(platform="android")
    provider = object()
    embedding_provider = object()
    trace_path = tmp_path / "gui_runs" / "run" / "trace.jsonl"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text("{}\n", encoding="utf-8")
    agent_state: dict[str, Any] = {}
    postprocess_state: dict[str, Any] = {}

    class FakeRecorder:
        def __init__(self, **_: Any) -> None:
            pass

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            agent_state.update(kwargs)

        async def run(self, task: str, **_: Any) -> AgentResult:
            return AgentResult(
                success=True,
                summary=f"Completed {task}",
                model_summary=None,
                trace_path=str(trace_path),
                steps_taken=2,
                error=None,
            )

    class FakePostRunProcessor:
        def __init__(self, **kwargs: Any) -> None:
            postprocess_state["init"] = kwargs

        def schedule(self, path: Path | None, **kwargs: Any) -> None:
            postprocess_state["schedule"] = {"trace_path": path, **kwargs}

        async def drain(self) -> None:
            postprocess_state["drained"] = True

    async def fake_build_optional_components(*_: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        postprocess_state["components_embedding"] = kwargs.get("embedding_provider")
        return None, object(), types.SimpleNamespace(subgoal_runner=None)

    args = cli.parse_args(["--dry-run", "--task", "Open Contacts"])
    monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(cli, "PostRunProcessor", FakePostRunProcessor, raising=False)
    monkeypatch.setattr(
        cli,
        "build_embedding_provider",
        lambda _: embedding_provider,
        raising=False,
    )
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)

    result = asyncio.run(cli._execute_agent(args, config, backend, provider, "Open Contacts"))

    assert result.success is True
    assert postprocess_state["components_embedding"] is embedding_provider
    assert agent_state["enable_prompt_skill_selection"] is True
    assert postprocess_state["init"] == {
        "llm": provider,
        "merge_llm": provider,
        "embedding_provider": embedding_provider,
        "embedding_signature": "embed-model",
        "skill_store_root": config.skills_dir,
        "enable_skill_extraction": True,
        "enable_memory_extraction": True,
        "memory_bank_path": config.memory_dir / "gui_memory_bank.jsonl",
    }
    assert postprocess_state["schedule"] == {
        "trace_path": trace_path,
        "is_success": True,
        "platform": "android",
        "task": "Open Contacts",
    }
    assert postprocess_state["drained"] is True


def test_cli_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    backend = _FakeBackend()

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir
            self.task = task
            self.platform = platform

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            self._progress_callback = kwargs["progress_callback"]

        async def run(self, task: str, **_: Any) -> AgentResult:
            await self._progress_callback("GUI step 1/15: inspect screen")
            return AgentResult(
                success=True,
                summary=f"Completed {task}",
                model_summary="Opened app and confirmed final state.",
                trace_path="trace.jsonl",
                steps_taken=3,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_backend", lambda backend_name, cfg: backend)
    monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)

    exit_code = cli.main(["--json", "--dry-run", "--task", "Open Settings"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload == {
        "success": True,
        "summary": "Completed Open Settings",
        "model_summary": "Opened app and confirmed final state.",
        "trace_path": "trace.jsonl",
        "steps_taken": 3,
        "error": None,
    }
    assert "GUI step 1/15: inspect screen" not in captured.out


def test_cli_main_catches_runtime_exception_and_prints_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import guiclaw.cli as cli

    monkeypatch.setattr(cli, "run_cli", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))

    exit_code = cli.main(["--dry-run", "--task", "Open Settings"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "status: failure" in captured.out
    assert "success: false" in captured.out
    assert "summary: CLI execution failed." in captured.out
    assert "error: RuntimeError: boom" in captured.out


def test_package_main_delegates_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    fake_cli = types.ModuleType("guiclaw.cli")

    def fake_main() -> int:
        called.append("main")
        return 0

    fake_cli.main = fake_main
    monkeypatch.setitem(sys.modules, "guiclaw.cli", fake_cli)
    sys.modules.pop("guiclaw.__main__", None)

    runpy.run_module("guiclaw.__main__", run_name="__main__")

    assert called == ["main"]


def test_cli_embedding_batch_splits_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import guiclaw.cli as cli

    calls: list[dict[str, Any]] = []

    class FakeEmbeddingsAPI:
        async def create(self, *, model: str, input: list[str]) -> Any:
            calls.append({"model": model, "input": list(input)})
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[float(int(text[1:]))]) for text in input]
            )

    class FakeClient:
        def __init__(self) -> None:
            self.embeddings = FakeEmbeddingsAPI()

    monkeypatch.setattr(
        cli,
        "AsyncOpenAI",
        lambda **_: FakeClient(),
    )

    provider = cli.OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:5678/v1",
        model="embed-model",
        api_key="embed-key",
    )
    texts = [f"t{i}" for i in range(23)]

    vectors = asyncio.run(provider.embed(texts))

    assert [len(call["input"]) for call in calls] == [10, 10, 3]
    assert vectors.shape == (23, 1)
    assert str(vectors.dtype) == "float32"
    assert [row[0] for row in vectors.tolist()] == [float(i) for i in range(23)]


def test_cli_enables_memory_and_skill_bundle_when_embedding_config_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import guiclaw.cli as cli

    calls: dict[str, list[Any]] = {
        "embedding": [],
        "memory_store": [],
        "retriever": [],
        "indexed_entries": [],
        "skill_library": [],
        "validator": [],
        "skill_executor": [],
        "grounder": [],
        "runner": [],
        "screenshots": [],
    }

    class FakeEmbeddingProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["embedding"].append(kwargs)

    class FakeMemoryStore:
        def __init__(self, store_dir: Path) -> None:
            calls["memory_store"].append(store_dir)

        def list_all(self, **_: Any) -> list[str]:
            return ["entry-1", "entry-2"]

    class FakeMemoryRetriever:
        def __init__(self, embedding_provider: Any, top_k: int = 5) -> None:
            self.embedding_provider = embedding_provider
            self.top_k = top_k
            calls["retriever"].append({"embedding_provider": embedding_provider, "top_k": top_k})

        async def index(self, entries: list[str]) -> None:
            calls["indexed_entries"].append(entries)

    class FakeFlatSkillLibrary:
        def __init__(
            self, store_dir: Path, embedding_provider: Any = None, merge_llm: Any = None
        ) -> None:
            calls["skill_library"].append(
                {
                    "store_dir": store_dir,
                    "embedding_provider": embedding_provider,
                    "merge_llm": merge_llm,
                }
            )

    class FakeValidator:
        def __init__(self, provider: Any, **kwargs: Any) -> None:
            calls["validator"].append({"provider": provider, **kwargs})

    class FakeGrounder:
        def __init__(self, **kwargs: Any) -> None:
            calls["grounder"].append(kwargs)

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            calls["runner"].append(kwargs)

    class FakeScreenshots:
        def __init__(self, **kwargs: Any) -> None:
            calls["screenshots"].append(kwargs)

    class FakeSkillExecutor:
        def __init__(self, backend: Any, state_validator: Any = None, **kwargs: Any) -> None:
            calls["skill_executor"].append(
                {"backend": backend, "state_validator": state_validator, **kwargs}
            )

    monkeypatch.setattr(cli, "OpenAICompatibleEmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr(cli, "MemoryStore", FakeMemoryStore)
    monkeypatch.setattr(cli, "MemoryRetriever", FakeMemoryRetriever)
    monkeypatch.setattr(cli, "FlatSkillLibrary", FakeFlatSkillLibrary)
    monkeypatch.setattr(cli, "LLMStateValidator", FakeValidator)
    monkeypatch.setattr(cli, "_AgentActionGrounder", FakeGrounder)
    monkeypatch.setattr(cli, "_AgentSubgoalRunner", FakeRunner)
    monkeypatch.setattr(cli, "_AgentScreenshotProvider", FakeScreenshots)
    monkeypatch.setattr(cli, "SkillExecutor", FakeSkillExecutor)

    provider = object()
    backend = _FakeBackend(platform="macos")
    with_embedding = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        ),
        embedding=cli.EmbeddingConfig(
            base_url="http://localhost:5678/v1",
            model="embed-model",
            api_key="embed-key",
        ),
        enable_skill_execution=True,
        agent_profile="qwen3vl",
    )
    artifacts_root = Path("/tmp/guiclaw-skill-artifacts")

    memory_retriever, skill_library, skill_executor = asyncio.run(
        cli.build_optional_components(
            with_embedding,
            provider=provider,
            backend=backend,
            model_name=with_embedding.provider.model,
            artifacts_root=artifacts_root,
        )
    )

    assert memory_retriever is not None
    assert skill_library is not None
    assert skill_executor is not None
    assert calls["memory_store"] == [cli.DEFAULT_MEMORY_DIR]
    assert calls["indexed_entries"] == [["entry-1", "entry-2"]]
    assert calls["retriever"][0]["top_k"] == 5
    assert calls["skill_library"][0]["store_dir"] == cli.DEFAULT_SKILLS_DIR
    assert calls["skill_library"][0]["merge_llm"] is provider
    assert calls["validator"][0]["provider"] is provider
    assert calls["validator"][0]["image_scale_ratio"] == pytest.approx(0.5)
    assert calls["skill_executor"][0]["backend"] is backend
    assert calls["grounder"][0]["agent_profile"] == "qwen3vl"
    assert calls["grounder"][0]["image_scale_ratio"] == pytest.approx(0.5)
    assert calls["runner"][0]["agent_profile"] == "qwen3vl"
    assert calls["runner"][0]["image_scale_ratio"] == pytest.approx(0.5)
    assert calls["screenshots"][0]["artifacts_root"] == artifacts_root

    memory_only = replace(with_embedding, enable_skill_execution=False)
    memory_retriever_only, disabled_library, disabled_executor = asyncio.run(
        cli.build_optional_components(
            memory_only,
            provider=provider,
            backend=backend,
            model_name=memory_only.provider.model,
            artifacts_root=artifacts_root,
        )
    )
    assert memory_retriever_only is not None
    assert disabled_library is None
    assert disabled_executor is None

    no_embedding = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        ),
        enable_skill_execution=True,
    )

    no_memory_retriever, bm25_library, bm25_executor = asyncio.run(
        cli.build_optional_components(
            no_embedding,
            provider=provider,
            backend=backend,
            model_name=no_embedding.provider.model,
            artifacts_root=artifacts_root,
        )
    )

    assert no_memory_retriever is None
    assert bm25_library is not None
    assert bm25_executor is not None


# ---------------------------------------------------------------------------
# Phase 11 Plan 01 — --background CLI flag tests
# ---------------------------------------------------------------------------


def test_cli_parses_background_flags() -> None:
    """parse_args accepts --background and optional display geometry flags."""
    import guiclaw.cli as cli

    # Basic --background flag
    args = cli.parse_args(["--background", "--task", "t"])
    assert args.background is True
    assert args.display_num is None
    assert args.width is None
    assert args.height is None

    # With all display geometry flags
    args2 = cli.parse_args(
        [
            "--background",
            "--display-num",
            "42",
            "--width",
            "1920",
            "--height",
            "1080",
            "--task",
            "t",
        ]
    )
    assert args2.background is True
    assert args2.display_num == 42
    assert args2.width == 1920
    assert args2.height == 1080


def test_cli_background_rejects_adb() -> None:
    """--background combined with --backend adb exits with an error."""
    import guiclaw.cli as cli

    with pytest.raises(SystemExit):
        cli.parse_args(["--background", "--backend", "adb", "--task", "t"])


def test_cli_background_rejects_dry_run() -> None:
    """--background combined with --dry-run exits with an error."""
    import guiclaw.cli as cli

    with pytest.raises(SystemExit):
        cli.parse_args(["--background", "--dry-run", "--task", "t"])


def test_cli_background_implies_local() -> None:
    """--background without --backend should resolve to 'local'."""
    import guiclaw.cli as cli

    args = cli.parse_args(["--background", "--task", "t"])
    assert cli.resolve_backend_name(args) == "local"


def test_run_cli_logs_resolved_background_mode_before_agent_start() -> None:
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    inner_backend = _FakeBackend(platform="linux")
    events: list[str] = []
    log_messages: list[str] = []

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir
            self.task = task
            self.platform = platform

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["backend"] is inner_backend

        async def run(self, task: str, **_: Any) -> AgentResult:
            events.append("agent_started")
            return AgentResult(
                success=True,
                summary=f"Completed {task}",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    probe_result = runtime.IsolationProbeResult(
        supported=False,
        reason_code="xvfb_missing",
        retryable=True,
        host_platform="linux",
        backend_name="xvfb",
        sys_platform="linux",
    )

    class _EventHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "background runtime resolved:" in record.getMessage():
                events.append("mode_logged")
                log_messages.append(record.getMessage())

    handler = _EventHandler()
    logger = logging.getLogger("guiclaw.cli")
    logger.addHandler(handler)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(cli, "load_config", lambda path=None: config)
        monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
        monkeypatch.setattr(cli, "build_backend", lambda backend_name, cfg: inner_backend)
        monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
        monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
        monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
        monkeypatch.setattr(cli, "probe_isolated_background_support", lambda **_: probe_result)
        asyncio.run(cli.run_cli(cli.parse_args(["--background", "--task", "Open Settings"])))
    finally:
        logger.removeHandler(handler)
        monkeypatch.undo()

    assert events[0] == "mode_logged"
    assert events[1] == "agent_started"
    assert "mode=fallback" in log_messages[0]
    assert "reason=xvfb_missing" in log_messages[0]
    assert "Install Xvfb to enable isolated background execution." in log_messages[0]


def test_run_cli_blocks_when_isolation_required_but_unavailable() -> None:
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    probe_result = runtime.IsolationProbeResult(
        supported=False,
        reason_code="xvfb_missing",
        retryable=True,
        host_platform="linux",
        backend_name="xvfb",
        sys_platform="linux",
    )

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(cli, "load_config", lambda path=None: config)
        monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
        monkeypatch.setattr(
            cli, "build_backend", lambda backend_name, cfg: _FakeBackend(platform="linux")
        )
        monkeypatch.setattr(cli, "probe_isolated_background_support", lambda **_: probe_result)
        monkeypatch.setattr(
            cli,
            "GuiAgent",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("GuiAgent should not be constructed")
            ),
        )

        args = cli.parse_args(["--background", "--require-isolation", "--task", "open settings"])
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(cli.run_cli(args))
    finally:
        monkeypatch.undo()

    assert "blocked" in str(exc_info.value)
    assert "xvfb_missing" in str(exc_info.value)
    assert "Install Xvfb to enable isolated background execution." in str(exc_info.value)


def test_run_cli_background_wraps_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux, run_cli wraps the inner backend in BackgroundDesktopBackend."""
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    inner_backend = _FakeBackend(platform="linux")
    wrapped_backend_ref: list[Any] = []
    agent_backend_ref: list[Any] = []

    class FakeXvfbDisplayManager:
        def __init__(self, display_num: int = 99, width: int = 1280, height: int = 720) -> None:
            self.display_num = display_num
            self.width = width
            self.height = height

        async def start(self) -> Any:
            from guiclaw.backends.virtual_display import DisplayInfo

            return DisplayInfo(display_id=":99", width=self.width, height=self.height)

        async def stop(self) -> None:
            pass

    class FakeBackgroundBackend:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            self._inner = inner
            self._manager = manager
            self._run_metadata = run_metadata
            wrapped_backend_ref.append(self)
            self.shutdown_calls = 0

        @property
        def platform(self) -> str:
            return self._inner.platform

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir
            self.task = task
            self.platform = platform

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            agent_backend_ref.append(kwargs["backend"])

        async def run(self, task: str, **_: Any) -> AgentResult:
            return AgentResult(
                success=True,
                summary="done",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_backend", lambda name, cfg: inner_backend)
    monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
    monkeypatch.setattr(cli, "BackgroundDesktopBackend", FakeBackgroundBackend)
    monkeypatch.setattr(
        cli,
        "probe_isolated_background_support",
        lambda **_: runtime.IsolationProbeResult(
            supported=True,
            reason_code="xvfb_available",
            retryable=False,
            host_platform="linux",
            backend_name="xvfb",
            sys_platform="linux",
        ),
    )
    monkeypatch.setattr(sys, "platform", "linux")

    import guiclaw.backends.displays.xvfb as xvfb_mod

    monkeypatch.setattr(xvfb_mod, "XvfbDisplayManager", FakeXvfbDisplayManager)

    # Also patch the xvfb import inside run_cli's local scope by patching the module reference
    import importlib
    import guiclaw.backends.displays.xvfb as _xvfb

    monkeypatch.setattr(_xvfb, "XvfbDisplayManager", FakeXvfbDisplayManager)

    args = cli.parse_args(["--background", "--task", "open settings"])
    asyncio.run(cli.run_cli(args))

    # FakeBackgroundBackend was constructed and the agent received the wrapped backend
    assert len(wrapped_backend_ref) == 1
    assert wrapped_backend_ref[0]._inner is inner_backend
    assert wrapped_backend_ref[0]._run_metadata == {"owner": "cli", "task": "open settings"}
    assert wrapped_backend_ref[0].shutdown_calls == 1
    assert len(agent_backend_ref) == 1
    assert agent_backend_ref[0] is wrapped_backend_ref[0]


def test_run_cli_background_nonlinux_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unsupported isolation falls back on the raw backend and logs the resolved mode."""
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    inner_backend = _FakeBackend(platform="macos")
    agent_backend_ref: list[Any] = []
    bg_created: list[Any] = []

    class FakeBackgroundBackend:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            bg_created.append(self)

        @property
        def platform(self) -> str:
            return "linux"

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir
            self.task = task
            self.platform = platform

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            agent_backend_ref.append(kwargs["backend"])

        async def run(self, task: str, **_: Any) -> AgentResult:
            return AgentResult(
                success=True,
                summary="done",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_backend", lambda name, cfg: inner_backend)
    monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
    monkeypatch.setattr(cli, "BackgroundDesktopBackend", FakeBackgroundBackend)
    monkeypatch.setattr(
        cli,
        "probe_isolated_background_support",
        lambda **_: runtime.IsolationProbeResult(
            supported=False,
            reason_code="platform_unsupported",
            retryable=False,
            host_platform="macos",
            backend_name=None,
            sys_platform="darwin",
        ),
    )
    monkeypatch.setattr(sys, "platform", "darwin")

    args = cli.parse_args(["--background", "--task", "open settings"])
    with caplog.at_level(logging.WARNING, logger="guiclaw.cli"):
        asyncio.run(cli.run_cli(args))

    # BackgroundDesktopBackend was NOT constructed
    assert len(bg_created) == 0
    # Agent received raw backend
    assert len(agent_backend_ref) == 1
    assert agent_backend_ref[0] is inner_backend
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("background runtime resolved:" in msg for msg in warning_messages)
    assert any(
        "mode=fallback" in msg and "reason=platform_unsupported" in msg for msg in warning_messages
    )


def test_run_cli_background_uses_cli_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """XvfbDisplayManager is constructed with --display-num, --width, --height values."""
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    inner_backend = _FakeBackend(platform="linux")
    xvfb_init_kwargs: list[dict[str, Any]] = []

    class FakeXvfbDisplayManager:
        def __init__(self, display_num: int = 99, width: int = 1280, height: int = 720) -> None:
            xvfb_init_kwargs.append({"display_num": display_num, "width": width, "height": height})

        async def start(self) -> Any:
            from guiclaw.backends.virtual_display import DisplayInfo

            return DisplayInfo(display_id=":42", width=1920, height=1080)

        async def stop(self) -> None:
            pass

    class FakeBackgroundBackend:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            self._inner = inner
            self._manager = manager

        @property
        def platform(self) -> str:
            return self._inner.platform

        async def shutdown(self) -> None:
            return None

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run(self, task: str, **_: Any) -> AgentResult:
            return AgentResult(
                success=True,
                summary="done",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_backend", lambda name, cfg: inner_backend)
    monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
    monkeypatch.setattr(cli, "BackgroundDesktopBackend", FakeBackgroundBackend)
    monkeypatch.setattr(
        cli,
        "probe_isolated_background_support",
        lambda **_: runtime.IsolationProbeResult(
            supported=True,
            reason_code="xvfb_available",
            retryable=False,
            host_platform="linux",
            backend_name="xvfb",
            sys_platform="linux",
        ),
    )
    monkeypatch.setattr(sys, "platform", "linux")

    import guiclaw.backends.displays.xvfb as _xvfb

    monkeypatch.setattr(_xvfb, "XvfbDisplayManager", FakeXvfbDisplayManager)

    args = cli.parse_args(
        [
            "--background",
            "--display-num",
            "42",
            "--width",
            "1920",
            "--height",
            "1080",
            "--task",
            "t",
        ]
    )
    asyncio.run(cli.run_cli(args))

    assert len(xvfb_init_kwargs) == 1
    assert xvfb_init_kwargs[0] == {"display_num": 42, "width": 1920, "height": 1080}


def test_run_cli_uses_windows_isolated_desktop_backend_for_windows_isolated_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    inner_backend = _FakeBackend(platform="windows")
    manager_init_kwargs: list[dict[str, Any]] = []
    wrapped_backend_ref: list[Any] = []
    agent_backend_ref: list[Any] = []

    class FakeWin32DesktopManager:
        def __init__(self, width: int = 1280, height: int = 720) -> None:
            manager_init_kwargs.append({"width": width, "height": height})

    class FakeWindowsIsolatedBackend:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            self._inner = inner
            self._manager = manager
            self._run_metadata = run_metadata
            self.shutdown_calls = 0
            wrapped_backend_ref.append(self)

        @property
        def platform(self) -> str:
            return self._inner.platform

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir
            self.task = task
            self.platform = platform

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            agent_backend_ref.append(kwargs["backend"])

        async def run(self, task: str, **_: Any) -> AgentResult:
            return AgentResult(
                success=True,
                summary="done",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_backend", lambda name, cfg: inner_backend)
    monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
    monkeypatch.setattr(cli, "WindowsIsolatedBackend", FakeWindowsIsolatedBackend, raising=False)
    monkeypatch.setattr(
        cli,
        "probe_isolated_background_support",
        lambda **_: runtime.IsolationProbeResult(
            supported=True,
            reason_code="windows_isolated_desktop_available",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        ),
    )
    monkeypatch.setattr(sys, "platform", "win32")

    import guiclaw.backends.displays.win32desktop as win32_mod

    monkeypatch.setattr(win32_mod, "Win32DesktopManager", FakeWin32DesktopManager)

    args = cli.parse_args(
        ["--background", "--width", "1600", "--height", "900", "--task", "open settings"]
    )
    asyncio.run(cli.run_cli(args))

    assert manager_init_kwargs == [{"width": 1600, "height": 900}]
    assert len(wrapped_backend_ref) == 1
    assert wrapped_backend_ref[0]._inner is inner_backend
    assert wrapped_backend_ref[0]._run_metadata == {"owner": "cli", "task": "open settings"}
    assert wrapped_backend_ref[0].shutdown_calls == 1
    assert agent_backend_ref == [wrapped_backend_ref[0]]


def test_run_cli_passes_target_app_class_to_windows_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    inner_backend = _FakeBackend(platform="windows")
    probe_calls: list[dict[str, Any]] = []

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    async def fake_execute_agent(
        args: Any,
        loaded_config: Any,
        backend: Any,
        provider: Any,
        task: str,
    ) -> AgentResult:
        assert backend is inner_backend
        assert loaded_config is config
        assert task == "Open Settings"
        return AgentResult(
            success=True,
            summary="done",
            model_summary=None,
            trace_path=None,
            steps_taken=1,
            error=None,
        )

    def fake_probe(**kwargs: Any) -> runtime.IsolationProbeResult:
        probe_calls.append(kwargs)
        return runtime.IsolationProbeResult(
            supported=False,
            reason_code="platform_unsupported",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        )

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_backend", lambda name, cfg: inner_backend)
    monkeypatch.setattr(cli, "_execute_agent", fake_execute_agent)
    monkeypatch.setattr(cli, "probe_isolated_background_support", fake_probe)
    monkeypatch.setattr(sys, "platform", "win32")

    asyncio.run(
        cli.run_cli(
            cli.parse_args(["--background", "--target-app-class", "uwp", "--task", "Open Settings"])
        )
    )
    asyncio.run(cli.run_cli(cli.parse_args(["--background", "--task", "Open Settings"])))

    assert probe_calls[0] == {"sys_platform": "win32", "target_app_class": "uwp"}
    assert probe_calls[1] == {"sys_platform": "win32", "target_app_class": "classic-win32"}


def test_run_cli_warns_for_windows_unsupported_app_class_before_agent_start() -> None:
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    inner_backend = _FakeBackend(platform="windows")
    events: list[str] = []
    log_messages: list[str] = []

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir
            self.task = task
            self.platform = platform

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["backend"] is inner_backend

        async def run(self, task: str, **_: Any) -> AgentResult:
            events.append("agent_started")
            return AgentResult(
                success=True,
                summary=f"Completed {task}",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    class _EventHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "background runtime resolved:" in record.getMessage():
                events.append("mode_logged")
                log_messages.append(record.getMessage())

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    handler = _EventHandler()
    cli_logger = logging.getLogger("guiclaw.cli")
    cli_logger.addHandler(handler)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(cli, "load_config", lambda path=None: config)
        monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
        monkeypatch.setattr(cli, "build_backend", lambda name, cfg: inner_backend)
        monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
        monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
        monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
        monkeypatch.setattr(
            cli,
            "WindowsIsolatedBackend",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("WindowsIsolatedBackend should not be constructed for fallback")
            ),
            raising=False,
        )
        monkeypatch.setattr(
            cli,
            "probe_isolated_background_support",
            lambda **_: runtime.IsolationProbeResult(
                supported=False,
                reason_code="windows_app_class_unsupported",
                retryable=False,
                host_platform="windows",
                backend_name="windows_isolated_desktop",
                sys_platform="win32",
            ),
        )
        monkeypatch.setattr(sys, "platform", "win32")

        import guiclaw.backends.displays.win32desktop as win32_mod

        monkeypatch.setattr(
            win32_mod,
            "Win32DesktopManager",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Win32DesktopManager should not be constructed for fallback")
            ),
        )

        asyncio.run(cli.run_cli(cli.parse_args(["--background", "--task", "Open Settings"])))
    finally:
        cli_logger.removeHandler(handler)
        monkeypatch.undo()

    assert events[:2] == ["mode_logged", "agent_started"]
    assert "windows_app_class_unsupported" in log_messages[0]
    assert "classic Win32/GDI" in log_messages[0]
    assert "UWP, DirectX, and GPU-heavy surfaces" in log_messages[0]


def test_run_cli_background_decision_tokens_stay_consistent_across_supported_hosts() -> None:
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    log_messages: list[str] = []

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    class FakeBackgroundBackend:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            self._inner = inner

        @property
        def platform(self) -> str:
            return self._inner.platform

        async def shutdown(self) -> None:
            return None

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            self._backend = kwargs["backend"]

        async def run(self, task: str, **_: Any) -> AgentResult:
            return AgentResult(
                success=True,
                summary=f"done {task}",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    class _EventHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "background runtime resolved:" in record.getMessage():
                log_messages.append(record.getMessage())

    def run_case(
        *,
        sys_platform: str,
        probe_result: runtime.IsolationProbeResult,
        argv: list[str],
    ) -> str:
        monkeypatch = pytest.MonkeyPatch()
        handler = _EventHandler()
        cli_logger = logging.getLogger("guiclaw.cli")
        previous_level = cli_logger.level
        cli_logger.setLevel(logging.INFO)
        cli_logger.addHandler(handler)
        try:
            monkeypatch.setattr(cli, "load_config", lambda path=None: config)
            monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
            monkeypatch.setattr(
                cli,
                "build_backend",
                lambda backend_name, cfg: _FakeBackend(
                    platform=(
                        "macos"
                        if sys_platform == "darwin"
                        else "windows"
                        if sys_platform == "win32"
                        else "linux"
                    )
                ),
            )
            monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
            monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
            monkeypatch.setattr(cli, "BackgroundDesktopBackend", FakeBackgroundBackend)
            monkeypatch.setattr(cli, "probe_isolated_background_support", lambda **_: probe_result)
            monkeypatch.setattr(
                cli, "TrajectoryRecorder", lambda *args, **kwargs: types.SimpleNamespace(path=None)
            )
            monkeypatch.setattr(sys, "platform", sys_platform)

            if probe_result.backend_name == "windows_isolated_desktop":
                import guiclaw.backends.displays.win32desktop as win32_mod

                monkeypatch.setattr(
                    win32_mod, "Win32DesktopManager", lambda *args, **kwargs: object()
                )

            try:
                asyncio.run(cli.run_cli(cli.parse_args(argv)))
            except RuntimeError as exc:
                return str(exc)
            return ""
        finally:
            cli_logger.removeHandler(handler)
            cli_logger.setLevel(previous_level)
            monkeypatch.undo()

    blocked_error = run_case(
        sys_platform="linux",
        probe_result=runtime.IsolationProbeResult(
            supported=False,
            reason_code="xvfb_missing",
            retryable=True,
            host_platform="linux",
            backend_name="xvfb",
            sys_platform="linux",
        ),
        argv=["--background", "--require-isolation", "--task", "Open Settings"],
    )
    run_case(
        sys_platform="win32",
        probe_result=runtime.IsolationProbeResult(
            supported=False,
            reason_code="windows_app_class_unsupported",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        ),
        argv=["--background", "--task", "Open Settings"],
    )

    assert "xvfb_missing" in blocked_error
    assert any(
        "owner=cli" in message and "mode=blocked" in message and "reason=xvfb_missing" in message
        for message in log_messages
    )
    assert any(
        "owner=cli" in message
        and "mode=fallback" in message
        and "reason=windows_app_class_unsupported" in message
        for message in log_messages
    )


def test_run_cli_logs_windows_target_surface_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from guiclaw.agent import AgentResult
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime
    import guiclaw.backends.windows_isolated as win_iso
    from guiclaw.backends.virtual_display import DisplayInfo

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    inner_backend = _FakeBackend(platform="windows")

    class FakeWin32DesktopManager:
        def __init__(self, width: int = 1280, height: int = 720) -> None:
            self._width = width
            self._height = height
            self._desktop_name = "GUIClaw-Background-1"

        @property
        def desktop_name(self) -> str:
            return self._desktop_name

        async def start(self) -> DisplayInfo:
            return DisplayInfo(
                display_id="windows_isolated_desktop:GUIClaw-Background-1",
                width=self._width,
                height=self._height,
                offset_x=0,
                offset_y=0,
                monitor_index=1,
            )

        async def stop(self) -> None:
            return None

    class FakeWorkerProcess:
        def __init__(self) -> None:
            self.stdin = self._FakeStdin()
            self.stdout = self._FakeStdout(['{"ok": true, "result": "shutdown"}\n'])
            self.stderr = self._FakePipe()

        class _FakePipe:
            def close(self) -> None:
                return None

        class _FakeStdin(_FakePipe):
            def write(self, data: str) -> None:
                return None

            def flush(self) -> None:
                return None

        class _FakeStdout(_FakePipe):
            def __init__(self, responses: list[str]) -> None:
                self._responses = responses

            def readline(self) -> str:
                if self._responses:
                    return self._responses.pop(0)
                return ""

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float = 1) -> int:
            return 0

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeRecorder:
        def __init__(self, output_dir: Path, task: str, platform: str = "unknown") -> None:
            self.output_dir = output_dir
            self.task = task
            self.platform = platform

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            self._backend = kwargs["backend"]

        async def run(self, task: str, **_: Any) -> AgentResult:
            await self._backend.preflight()
            return AgentResult(
                success=True,
                summary="done",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_backend", lambda name, cfg: inner_backend)
    monkeypatch.setattr(cli, "TrajectoryRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
    monkeypatch.setattr(
        cli,
        "probe_isolated_background_support",
        lambda **_: runtime.IsolationProbeResult(
            supported=True,
            reason_code="windows_isolated_desktop_available",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        ),
    )
    monkeypatch.setattr(sys, "platform", "win32")

    import guiclaw.backends.displays.win32desktop as win32_mod

    monkeypatch.setattr(win32_mod, "Win32DesktopManager", FakeWin32DesktopManager)
    monkeypatch.setattr(win_iso, "launch_windows_worker", lambda **kwargs: FakeWorkerProcess())

    with caplog.at_level(logging.INFO):
        result = asyncio.run(
            cli.run_cli(cli.parse_args(["--background", "--task", "Open Settings"]))
        )

    assert result.success is True
    assert "backend_name=windows_isolated_desktop" in caplog.text
    assert "display_id=windows_isolated_desktop:GUIClaw-Background-1" in caplog.text


def test_run_cli_handoff_and_cleanup_tokens_stay_visible_without_leaking_sensitive_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from guiclaw.interfaces import InterventionRequest
    import guiclaw.cli as cli

    class _FakeInterventionBackend:
        def get_intervention_target(self) -> dict[str, Any]:
            return {
                "display_id": "windows_isolated_desktop:GUIClaw-Background-1",
                "desktop_name": "GUIClaw-Background-1",
                "session_token": "top-secret",
            }

    handler = cli._CliInterventionHandler(_FakeInterventionBackend())
    request = InterventionRequest(
        task="Complete payroll login",
        reason="Need the user to enter OTP 123456 for the payroll login.",
        step_index=1,
        platform="windows",
        foreground_app=None,
        target={
            "display_id": "windows_isolated_desktop:GUIClaw-Background-1",
            "desktop_name": "GUIClaw-Background-1",
            "session_token": "top-secret",
        },
    )

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("builtins.input", lambda prompt="": "cancel")
        resolution = asyncio.run(handler.request_intervention(request))
    finally:
        monkeypatch.undo()

    cli._print_human_result(
        cli.AgentResult(
            success=False,
            summary="CLI execution failed.",
            model_summary=None,
            trace_path=None,
            steps_taken=0,
            error=(
                "RuntimeError: worker startup failed cleanup_reason=startup_failed "
                "display_id=windows_isolated_desktop:GUIClaw-Background-1 "
                "desktop_name=GUIClaw-Background-1"
            ),
        )
    )

    captured = capsys.readouterr()
    assert resolution.resume_confirmed is False
    assert "<redacted:intervention_reason>" in captured.out
    assert "OTP 123456" not in captured.out
    assert "display_id" in captured.out
    assert "desktop_name" in captured.out
    assert "session_token" not in captured.out
    assert "cleanup_reason=startup_failed" in captured.out
    assert "display_id=windows_isolated_desktop:GUIClaw-Background-1" in captured.out


def test_run_cli_intervention_flow_resumes_after_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from guiclaw.agent import AgentResult
    from guiclaw.interfaces import LLMResponse, ToolCall
    import guiclaw.cli as cli

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    backend = _InterventionCliBackend()
    reason = "Need the user to enter OTP 123456 for the payroll login."

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(
        cli,
        "OpenAICompatibleLLMProvider",
        lambda **_: _ScriptedCliProvider(
            [
                LLMResponse(
                    content="request intervention",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="computer_use",
                            arguments={"action_type": "request_intervention", "text": reason},
                        )
                    ],
                ),
                LLMResponse(
                    content="done",
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
    )
    monkeypatch.setattr(cli, "build_backend", lambda backend_name, loaded_config: backend)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
    monkeypatch.setattr("builtins.input", lambda prompt="": "resume")

    result = asyncio.run(cli.run_cli(cli.parse_args(["--task", "Complete payroll login"])))

    captured = capsys.readouterr()
    assert isinstance(result, AgentResult)
    assert result.success is True
    assert Path(backend.observe_calls[1]).name == "001_request_intervention.png"
    assert "<redacted:intervention_reason>" in captured.out
    assert reason not in captured.out
    assert "display_id" in captured.out
    assert "monitor_index" in captured.out
    assert "session_token" not in captured.out


def test_run_cli_intervention_logs_are_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from guiclaw.interfaces import LLMResponse, ToolCall
    import guiclaw.cli as cli

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )
    backend = _InterventionCliBackend()
    reason = "Need the user to enter OTP 123456 for the payroll login."

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(
        cli,
        "OpenAICompatibleLLMProvider",
        lambda **_: _ScriptedCliProvider(
            [
                LLMResponse(
                    content="request intervention",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="computer_use",
                            arguments={"action_type": "request_intervention", "text": reason},
                        )
                    ],
                )
            ]
        ),
    )
    monkeypatch.setattr(cli, "build_backend", lambda backend_name, loaded_config: backend)
    monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
    monkeypatch.setattr("builtins.input", lambda prompt="": "cancel")

    result = asyncio.run(cli.run_cli(cli.parse_args(["--task", "Handle OTP"])))

    trace_text = (Path(result.trace_path).parent / "traj.json").read_text(encoding="utf-8")
    captured = capsys.readouterr()

    assert result.success is False
    assert "<redacted:intervention_reason>" in captured.out
    assert reason not in captured.out
    assert "secret-session-token" not in captured.out
    assert "<redacted:intervention_reason>" in trace_text
    assert reason not in trace_text
    assert "secret-session-token" not in trace_text
