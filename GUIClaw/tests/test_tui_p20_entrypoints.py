from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from unittest.mock import patch

from nanobot.config.schema import Config


def _load_pyproject() -> dict:
    root = Path(__file__).resolve().parents[1]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_tui_main_enables_frontend_serving_for_the_canonical_entrypoint() -> None:
    from nanobot.tui import __main__ as tui_main

    fake_app = object()
    config = Config.model_validate(
        {
            "tui": {
                "host": "127.0.0.1",
                "port": 29999,
                "reload": False,
                "logLevel": "info",
            }
        }
    )

    with (
        patch("nanobot.tui.__main__.load_config", return_value=config),
        patch("nanobot.tui.__main__.create_app", return_value=fake_app) as create_app_mock,
        patch("nanobot.tui.__main__.uvicorn.run"),
    ):
        tui_main.main()

    create_app_mock.assert_called_once_with(
        config=config,
        include_runtime_routes=True,
        serve_frontend=True,
    )


def test_pyproject_packages_the_tui_console_script_and_frontend_assets() -> None:
    pyproject = _load_pyproject()

    scripts = pyproject["project"]["scripts"]
    build = pyproject["tool"]["hatch"]["build"]

    assert scripts["nanobot-tui"] == "nanobot.tui.__main__:main"
    assert "nanobot/tui/web/index.html" in build["include"]
    assert "nanobot/tui/web/dist/**/*" in build["artifacts"]


def test_tui_web_package_is_importable_for_resource_lookup() -> None:
    module = importlib.import_module("nanobot.tui.web")

    assert getattr(module, "__file__", "").endswith("__init__.py")
