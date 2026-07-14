"""Regression contract for the public GUIClaw backend surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from guiclaw.backends import adb, background_runtime, displays, read_png_size, virtual_display
from nanobot.config.schema import GuiConfig, IosConfig

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_and_nonfunctional_backends_are_removed() -> None:
    assert not (ROOT / "guiclaw/backends/mobileworld.py").exists()
    assert not (ROOT / "guiclaw/backends/displays/cgvirtualdisplay.py").exists()


def test_supported_isolated_backend_implementations_are_retained() -> None:
    assert (ROOT / "guiclaw/backends/displays/xvfb.py").is_file()
    assert (ROOT / "guiclaw/backends/windows_isolated.py").is_file()
    assert (ROOT / "guiclaw/backends/displays/win32desktop.py").is_file()


def test_virtual_display_module_has_no_noop_manager() -> None:
    assert not hasattr(virtual_display, "NoOpDisplayManager")


def test_displays_package_has_no_manager_reexports() -> None:
    assert not hasattr(displays, "CGVirtualDisplayManager")
    assert not hasattr(displays, "Win32DesktopManager")
    assert not hasattr(displays, "XvfbDisplayManager")


def test_mobileworld_is_not_a_configurable_backend() -> None:
    with pytest.raises(ValidationError):
        GuiConfig(backend="mobileworld")  # type: ignore[arg-type]
    assert "mobileworld" not in GuiConfig.model_fields


def test_ios_config_has_no_legacy_mjpeg_fields() -> None:
    assert "mjpeg_url" not in IosConfig.model_fields
    assert "mjpeg_frame_timeout_ms" not in IosConfig.model_fields


def test_adb_uses_shared_png_size_parser() -> None:
    assert not hasattr(adb, "_read_png_size")
    assert adb.read_png_size is read_png_size


def test_macos_isolated_background_is_explicitly_unsupported() -> None:
    result = background_runtime.probe_isolated_background_support(sys_platform="darwin")
    assert result.supported is False
    assert result.reason_code == "platform_unsupported"
    assert result.backend_name is None


def test_dead_background_remediations_are_removed() -> None:
    removed = {
        "macos_virtual_display_available",
        "windows_isolated_desktop_available",
        "windows_attach_desktop_failed",
    }
    assert removed.isdisjoint(background_runtime._REMEDIATIONS)
    assert not any(key.startswith("macos_") for key in background_runtime._REMEDIATIONS)
