from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import MagicMock, patch

from nanobot.config.loader import load_config
from nanobot.config.schema import Config
from nanobot.tui.config import resolve_tui_runtime_config


def test_tui_defaults_bind_to_localhost() -> None:
    config = Config()

    assert config.tui.host == "127.0.0.1"
    assert config.tui.port == 18791
    assert config.tui.reload is False


def test_load_config_accepts_explicit_tui_section(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tui": {
                    "host": "127.0.0.1",
                    "port": 29999,
                    "reload": True,
                    "logLevel": "debug",
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.tui.host == "127.0.0.1"
    assert loaded.tui.port == 29999
    assert loaded.tui.reload is True
    assert loaded.tui.log_level == "debug"


def test_tui_runtime_normalization_does_not_reuse_gateway_defaults() -> None:
    config = Config.model_validate(
        {
            "gateway": {"host": "0.0.0.0", "port": 24567},
        }
    )

    runtime = resolve_tui_runtime_config(config)

    assert runtime.host == "127.0.0.1"
    assert runtime.port == 18791
    assert runtime.host != config.gateway.host
    assert runtime.port != config.gateway.port


def test_tui_startup_wiring_uses_create_app_and_local_runtime_config() -> None:
    from nanobot.tui import __main__ as tui_main

    fake_app = object()
    config = Config.model_validate(
        {
            "tui": {
                "host": "127.0.0.1",
                "port": 29999,
                "reload": True,
                "logLevel": "debug",
            }
        }
    )

    with (
        patch("nanobot.tui.__main__.load_config", return_value=config) as load_config_mock,
        patch("nanobot.tui.__main__.create_app", return_value=fake_app) as create_app_mock,
        patch("nanobot.tui.__main__.uvicorn.run") as uvicorn_run,
    ):
        tui_main.main()

    load_config_mock.assert_called_once_with(None)
    create_app_mock.assert_called_once_with(
        config=config,
        include_runtime_routes=True,
        serve_frontend=True,
    )
    uvicorn_run.assert_called_once_with(
        fake_app,
        host="127.0.0.1",
        port=29999,
        reload=True,
        log_level="debug",
    )


def test_tui_module_import_does_not_boot_cli_runtime() -> None:
    module_name = "nanobot.tui.__main__"
    sys.modules.pop(module_name, None)

    with (
        patch("nanobot.cli.commands.sync_workspace_templates") as sync_templates,
        patch("nanobot.agent.loop.ChannelManager", create=True) as channel_manager,
        patch("nanobot.agent.loop.AgentLoop") as agent_loop,
    ):
        module = importlib.import_module(module_name)

    assert hasattr(module, "main")
    sync_templates.assert_not_called()
    channel_manager.assert_not_called()
    agent_loop.assert_not_called()


def test_chat_runtime_factory_uses_agent_loop_from_config(tmp_path) -> None:
    from nanobot.tui.dependencies import get_chat_runtime_factory

    config = Config.model_validate({"agents": {"defaults": {"workspace": str(tmp_path)}}})
    sessions = MagicMock()
    runtime = object()

    with patch(
        "nanobot.tui.dependencies.AgentLoop.from_config",
        return_value=runtime,
    ) as from_config:
        created = get_chat_runtime_factory(
            config=config,
            session_manager=sessions,
        )()

    assert created is runtime
    assert from_config.call_args.args[0] is config
    assert from_config.call_args.kwargs["session_manager"] is sessions
