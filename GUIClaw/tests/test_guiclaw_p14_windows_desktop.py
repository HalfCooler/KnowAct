"""Phase 14 Windows isolated desktop contract coverage."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import pathlib
from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call

import pytest

from guiclaw.action import Action
import guiclaw.backends.background_runtime as runtime
from guiclaw.backends.virtual_display import DisplayInfo
from guiclaw.observation import Observation


def _load_win32desktop_module() -> Any:
    try:
        return import_module("guiclaw.backends.displays.win32desktop")
    except ModuleNotFoundError as exc:  # pragma: no cover - red phase expectation
        pytest.fail(f"win32desktop module missing: {exc}")


def _load_windows_isolated_module() -> Any:
    try:
        return import_module("guiclaw.backends.windows_isolated")
    except ModuleNotFoundError as exc:  # pragma: no cover - red phase expectation
        pytest.fail(f"windows_isolated module missing: {exc}")


def _windows_probe_from_support(support: dict[str, object]) -> runtime.IsolationProbeResult:
    return runtime.IsolationProbeResult(
        supported=bool(support["supported"]),
        reason_code=str(support["reason_code"]),
        retryable=bool(support["retryable"]),
        host_platform="windows",
        backend_name="windows_isolated_desktop",
        sys_platform="win32",
    )


def _make_windows_display_info() -> DisplayInfo:
    return DisplayInfo(
        display_id="windows_isolated_desktop:GUIClaw-Background-1",
        width=1280,
        height=720,
        offset_x=0,
        offset_y=0,
        monitor_index=1,
    )


def _make_windows_manager(display_info: DisplayInfo | None = None) -> AsyncMock:
    manager = AsyncMock()
    manager.start = AsyncMock(return_value=display_info or _make_windows_display_info())
    manager.stop = AsyncMock()
    manager.desktop_name = "GUIClaw-Background-1"
    return manager


def _make_windows_inner() -> AsyncMock:
    inner = AsyncMock()
    type(inner).platform = PropertyMock(return_value="windows")
    inner.preflight = AsyncMock()
    inner.observe = AsyncMock()
    inner.execute = AsyncMock()
    inner.list_apps = AsyncMock(return_value=[])
    inner.configure_target_display = MagicMock()
    return inner


def test_probe_windows_isolated_desktop_available(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = runtime.IsolationProbeResult(
        supported=True,
        reason_code="windows_isolated_desktop_available",
        retryable=False,
        host_platform="windows",
        backend_name="windows_isolated_desktop",
        sys_platform="win32",
    )

    def _probe(raw_platform: str, *, target_app_class: str | None = None) -> runtime.IsolationProbeResult:
        assert raw_platform == "win32"
        assert target_app_class == "classic-win32"
        return expected

    monkeypatch.setattr(runtime, "_probe_windows_isolated_support", _probe)

    result = runtime.probe_isolated_background_support(
        sys_platform="win32",
        target_app_class="classic-win32",
    )

    assert result == expected


def test_probe_reports_non_interactive_windows_context(monkeypatch: pytest.MonkeyPatch) -> None:
    win32desktop = _load_win32desktop_module()
    support_state = win32desktop.WindowsIsolatedDesktopSupport(
        interactive_session=False,
        input_desktop_available=True,
        create_desktop_available=True,
        reason_code="windows_non_interactive_session",
    )
    monkeypatch.setattr(
        win32desktop,
        "_collect_windows_isolated_desktop_support",
        lambda: support_state,
    )

    support = win32desktop.probe_windows_isolated_desktop_support()
    decision = runtime.resolve_run_mode(
        _windows_probe_from_support(support),
        require_isolation=True,
        require_acknowledgement_for_fallback=False,
    )

    assert support["supported"] is False
    assert support["reason_code"] == "windows_non_interactive_session"
    assert "Session 0" in decision.message


def test_probe_reports_unsupported_app_class_with_actionable_message() -> None:
    win32desktop = _load_win32desktop_module()

    support = win32desktop.probe_windows_isolated_desktop_support(target_app_class="uwp")
    decision = runtime.resolve_run_mode(
        _windows_probe_from_support(support),
        require_isolation=True,
        require_acknowledgement_for_fallback=False,
    )

    assert support["supported"] is False
    assert support["reason_code"] == "windows_app_class_unsupported"
    assert "classic Win32" in decision.message
    assert "UWP" in decision.message
    assert "DirectX" in decision.message


@pytest.mark.asyncio
async def test_win32desktop_manager_creates_and_closes_real_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win32desktop = _load_win32desktop_module()

    class _FakeUUID:
        hex = "0123456789abcdef0123456789abcdef"

    manager = win32desktop.Win32DesktopManager(width=1600, height=900)
    created_desktops: list[str] = []
    closed_handles: list[object] = []
    handle = object()

    monkeypatch.setattr(win32desktop.uuid, "uuid4", lambda: _FakeUUID())
    monkeypatch.setattr(
        manager,
        "_win32_create_desktop",
        lambda desktop_name: created_desktops.append(desktop_name) or handle,
    )
    monkeypatch.setattr(
        manager,
        "_win32_close_desktop",
        lambda desktop_handle: closed_handles.append(desktop_handle),
    )

    first = await manager.start()
    second = await manager.start()
    assert manager.desktop_name == "GUIClaw-Background-01234567"
    await manager.stop()
    await manager.stop()

    assert first == DisplayInfo(
        display_id="windows_isolated_desktop:GUIClaw-Background-01234567",
        width=1600,
        height=900,
        offset_x=0,
        offset_y=0,
        monitor_index=1,
    )
    assert second == first
    assert created_desktops == ["GUIClaw-Background-01234567"]
    assert closed_handles == [handle]


def test_windows_worker_main_services_observe_execute_list_apps_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows_worker = import_module("guiclaw.backends.windows_worker")
    backend_events: list[object] = []

    class _FakeLocalDesktopBackend:
        async def preflight(self) -> None:
            backend_events.append("preflight")

        async def observe(
            self,
            screenshot_path: pathlib.Path,
            timeout: float = 5.0,
        ) -> Observation:
            backend_events.append(("observe", screenshot_path, timeout))
            return Observation(
                screenshot_path=str(screenshot_path),
                screen_width=800,
                screen_height=600,
                foreground_app="Notepad",
                platform="windows",
                extra={"display_id": "windows_isolated_desktop:GUIClaw-Background-1"},
            )

        async def execute(self, action: Action, timeout: float = 5.0) -> str:
            backend_events.append(("execute", action, timeout))
            if action.x == 99.0:
                raise RuntimeError("execute boom")
            return f"executed:{action.action_type}"

        async def list_apps(self) -> list[str]:
            backend_events.append("list_apps")
            return ["Notepad", "Calculator"]

    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps(
                    {
                        "command": "observe",
                        "screenshot_path": "/tmp/windows-shot.png",
                        "timeout": 5.0,
                    }
                ),
                json.dumps(
                    {
                        "command": "execute",
                        "action": {"action_type": "tap", "x": 10.0, "y": 20.0},
                        "timeout": 5.0,
                    }
                ),
                json.dumps({"command": "list_apps"}),
                json.dumps(
                    {
                        "command": "execute",
                        "action": {"action_type": "tap", "x": 99.0, "y": 100.0},
                        "timeout": 5.0,
                    }
                ),
                json.dumps({"command": "shutdown"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    monkeypatch.setattr(windows_worker, "LocalDesktopBackend", _FakeLocalDesktopBackend)
    monkeypatch.setattr(windows_worker.sys, "stdin", stdin)
    monkeypatch.setattr(windows_worker.sys, "stdout", stdout)

    assert windows_worker.main(
        [
            "--desktop-name",
            "GUIClaw-Background-1",
            "--width",
            "1280",
            "--height",
            "720",
            "--control-path",
            "/tmp/windows-worker-control.json",
        ]
    ) == 0

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert responses == [
        {
            "ok": True,
            "observation": {
                "screenshot_path": "/tmp/windows-shot.png",
                "screen_width": 800,
                "screen_height": 600,
                "foreground_app": "Notepad",
                "platform": "windows",
                "extra": {"display_id": "windows_isolated_desktop:GUIClaw-Background-1"},
            },
        },
        {"ok": True, "result": "executed:tap"},
        {"ok": True, "apps": ["Notepad", "Calculator"]},
        {"ok": False, "error": "execute boom"},
        {"ok": True, "result": "shutdown"},
    ]
    assert backend_events == [
        "preflight",
        ("observe", pathlib.Path("/tmp/windows-shot.png"), 5.0),
        ("execute", Action(action_type="tap", x=10.0, y=20.0), 5.0),
        "list_apps",
        ("execute", Action(action_type="tap", x=99.0, y=100.0), 5.0),
    ]


@pytest.mark.asyncio
async def test_windows_isolated_backend_routes_observe_execute_and_list_apps_through_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows_isolated = _load_windows_isolated_module()
    display_info = _make_windows_display_info()
    manager = _make_windows_manager(display_info)
    inner = _make_windows_inner()
    backend = windows_isolated.WindowsIsolatedBackend(inner, manager)

    monkeypatch.setattr(
        backend,
        "_start_worker_session",
        AsyncMock(side_effect=lambda target_display: setattr(backend, "_worker_process", object())),
    )
    send_worker_command = AsyncMock(
        side_effect=[
            {
                "ok": True,
                "observation": {
                    "screenshot_path": "/tmp/windows-shot.png",
                    "screen_width": display_info.width,
                    "screen_height": display_info.height,
                    "foreground_app": "Notepad",
                    "platform": "windows",
                    "extra": {"display_id": display_info.display_id},
                },
            },
            {"ok": True, "result": "executed tap"},
            {"ok": True, "apps": ["Notepad", "Calculator"]},
            {"ok": True, "result": "shutdown"},
        ]
    )
    monkeypatch.setattr(backend, "_send_worker_command", send_worker_command)
    monkeypatch.setattr(backend, "_stop_worker_process", AsyncMock())
    monkeypatch.setattr(backend._display_manager, "stop", AsyncMock())

    try:
        await backend.preflight()
        observation = await backend.observe(pathlib.Path("/tmp/windows-shot.png"))
        result = await backend.execute(Action(action_type="tap", x=10.0, y=20.0))
        apps = await backend.list_apps()
    finally:
        await backend.shutdown()

    assert observation == Observation(
        screenshot_path="/tmp/windows-shot.png",
        screen_width=display_info.width,
        screen_height=display_info.height,
        foreground_app="Notepad",
        platform="windows",
        extra={"display_id": display_info.display_id},
    )
    assert result == "executed tap"
    assert apps == ["Notepad", "Calculator"]
    send_worker_command.assert_has_awaits(
        [
            call(
                {
                    "command": "observe",
                    "screenshot_path": "/tmp/windows-shot.png",
                    "timeout": 5.0,
                }
            ),
            call(
                {
                    "command": "execute",
                    "action": {
                        "action_type": "tap",
                        "x": 10.0,
                        "y": 20.0,
                        "x2": None,
                        "y2": None,
                        "text": None,
                        "key": None,
                        "pixels": None,
                        "duration_ms": None,
                        "relative": False,
                        "status": None,
                        "auto_enter": True,
                    },
                    "timeout": 5.0,
                }
            ),
            call({"command": "list_apps"}),
            call({"command": "shutdown"}),
        ]
    )
    inner.preflight.assert_not_awaited()
    inner.observe.assert_not_awaited()
    inner.execute.assert_not_awaited()
    inner.list_apps.assert_not_awaited()


@pytest.mark.asyncio
async def test_windows_isolated_backend_shutdown_closes_worker_before_desktop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    windows_isolated = _load_windows_isolated_module()

    class _FakePipe:
        def __init__(self, name: str, events: list[str]) -> None:
            self._name = name
            self._events = events

        def close(self) -> None:
            self._events.append(f"close:{self._name}")

    async def _run_shutdown(cleanup_reason: str) -> list[str]:
        display_info = _make_windows_display_info()
        manager = _make_windows_manager(display_info)
        inner = _make_windows_inner()
        backend = windows_isolated.WindowsIsolatedBackend(inner, manager)
        control_path = tmp_path / f"{cleanup_reason}.json"
        control_path.write_text("{}", encoding="utf-8")
        process_events: list[str] = []

        class _FakeLease:
            async def __aexit__(self, *_: object) -> None:
                process_events.append("lease_release")

        fake_process = SimpleNamespace(
            stdin=_FakePipe("stdin", process_events),
            stdout=_FakePipe("stdout", process_events),
            stderr=_FakePipe("stderr", process_events),
        )

        backend._display_info = display_info
        backend._worker_process = fake_process
        backend._worker_control_path = os.fspath(control_path)
        backend._lease_cm = _FakeLease()
        backend._stopped = False

        monkeypatch.setattr(
            backend,
            "_send_worker_command",
            AsyncMock(
                side_effect=lambda payload: process_events.append(f"send:{payload['command']}") or {
                    "ok": True,
                    "result": "shutdown",
                }
            ),
        )
        monkeypatch.setattr(
            backend,
            "_stop_worker_process",
            AsyncMock(side_effect=lambda timeout=1.0: process_events.append("stop_worker_process")),
        )
        monkeypatch.setattr(
            backend._display_manager,
            "stop",
            AsyncMock(side_effect=lambda: process_events.append("display_stop")),
        )

        await backend.shutdown(cleanup_reason)

        assert not control_path.exists()
        assert process_events.index("send:shutdown") < process_events.index("stop_worker_process")
        assert process_events.index("stop_worker_process") < process_events.index("close:stdin")
        assert process_events.index("close:stdin") < process_events.index("display_stop")
        assert process_events.index("close:stdout") < process_events.index("display_stop")
        assert process_events.index("close:stderr") < process_events.index("display_stop")
        assert process_events[-1] == "lease_release"
        return process_events

    caplog.set_level(logging.INFO)

    startup_failed_events = await _run_shutdown("startup_failed")
    assert "cleanup_reason=startup_failed" in caplog.text
    caplog.clear()

    cancelled_events = await _run_shutdown("cancelled")
    assert "cleanup_reason=cancelled" in caplog.text
    assert startup_failed_events.count("display_stop") == 1
    assert cancelled_events.count("display_stop") == 1


@pytest.mark.asyncio
async def test_windows_isolated_backend_exposes_handoff_target_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows_isolated = _load_windows_isolated_module()
    display_info = _make_windows_display_info()
    manager = _make_windows_manager(display_info)
    inner = _make_windows_inner()
    backend = windows_isolated.WindowsIsolatedBackend(inner, manager)

    monkeypatch.setattr(
        backend,
        "_start_worker_session",
        AsyncMock(side_effect=lambda target_display: setattr(backend, "_worker_process", object())),
    )
    monkeypatch.setattr(backend, "_shutdown_worker_session", AsyncMock())
    monkeypatch.setattr(backend._display_manager, "stop", AsyncMock())

    try:
        await backend.preflight()
        assert backend.get_intervention_target() == {
            "display_id": "windows_isolated_desktop:GUIClaw-Background-1",
            "desktop_name": "GUIClaw-Background-1",
            "width": 1280,
            "height": 720,
            "platform": "windows",
        }
    finally:
        await backend.shutdown()
