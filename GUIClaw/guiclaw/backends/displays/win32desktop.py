"""Windows isolated desktop support and lifecycle helpers.

This module stays import-safe on non-Windows hosts so contract tests can run in
CI while Phase 14 builds out the real worker-backed execution seam.
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import sys
import uuid
from typing import Any

from guiclaw.backends.virtual_display import DisplayInfo

logger = logging.getLogger(__name__)

_UNSUPPORTED_APP_CLASSES = {"uwp", "directx", "gpu-heavy", "electron-gpu"}
_DESKTOP_ACCESS_MASK = 0x000F01FF


@dataclasses.dataclass(frozen=True)
class WindowsIsolatedDesktopSupport:
    interactive_session: bool
    input_desktop_available: bool
    create_desktop_available: bool
    reason_code: str


def probe_windows_isolated_desktop_support(*, target_app_class: str | None = None) -> dict[str, object]:
    normalized_app_class = (target_app_class or "").strip().lower()
    if normalized_app_class in _UNSUPPORTED_APP_CLASSES:
        return {
            "supported": False,
            "reason_code": "windows_app_class_unsupported",
            "retryable": False,
        }

    support = _collect_windows_isolated_desktop_support()
    if not support.interactive_session:
        return {
            "supported": False,
            "reason_code": "windows_non_interactive_session",
            "retryable": False,
        }
    if not support.input_desktop_available:
        return {
            "supported": False,
            "reason_code": "windows_input_desktop_unavailable",
            "retryable": True,
        }
    if not support.create_desktop_available:
        return {
            "supported": False,
            "reason_code": "windows_create_desktop_failed",
            "retryable": True,
        }

    return {
        "supported": True,
        "reason_code": support.reason_code or "windows_isolated_desktop_available",
        "retryable": False,
    }


class Win32DesktopManager:
    """Manage a named isolated Windows desktop handle."""

    def __init__(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self._width = width
        self._height = height
        self._desktop_handle: Any | None = None
        self._display_info: DisplayInfo | None = None
        self._desktop_name: str | None = None

    @property
    def desktop_name(self) -> str:
        if self._desktop_name is None:
            raise RuntimeError("Windows desktop has not been started yet")
        return self._desktop_name

    async def start(self) -> DisplayInfo:
        if self._display_info is not None:
            return self._display_info

        if self._desktop_name is None:
            self._desktop_name = self._generate_desktop_name()

        handle = self._win32_create_desktop(self.desktop_name)
        info = DisplayInfo(
            display_id=f"windows_isolated_desktop:{self.desktop_name}",
            width=self._width,
            height=self._height,
            offset_x=0,
            offset_y=0,
            monitor_index=1,
        )
        self._desktop_handle = handle
        self._display_info = info
        return info

    async def stop(self) -> None:
        if self._desktop_handle is None:
            self._display_info = None
            return

        handle = self._desktop_handle
        try:
            self._win32_close_desktop(handle)
        except Exception:
            logger.exception("Error stopping Windows isolated desktop")
        finally:
            self._desktop_handle = None
            self._display_info = None
            self._desktop_name = None

    def _generate_desktop_name(self) -> str:
        return f"GUIClaw-Background-{uuid.uuid4().hex[:8]}"

    def _win32_create_desktop(self, desktop_name: str) -> Any:
        return _win32_create_desktop(desktop_name)

    def _win32_close_desktop(self, handle: Any) -> None:
        _win32_close_desktop(handle)


def _collect_windows_isolated_desktop_support() -> WindowsIsolatedDesktopSupport:
    if sys.platform != "win32":
        return WindowsIsolatedDesktopSupport(
            interactive_session=False,
            input_desktop_available=False,
            create_desktop_available=False,
            reason_code="windows_non_interactive_session",
        )

    interactive_session = _interactive_session_available()
    input_desktop_available = interactive_session and _input_desktop_available()
    create_desktop_available = interactive_session and _create_desktop_available()

    if not interactive_session:
        reason_code = "windows_non_interactive_session"
    elif not input_desktop_available:
        reason_code = "windows_input_desktop_unavailable"
    elif not create_desktop_available:
        reason_code = "windows_create_desktop_failed"
    else:
        reason_code = "windows_isolated_desktop_available"

    return WindowsIsolatedDesktopSupport(
        interactive_session=interactive_session,
        input_desktop_available=input_desktop_available,
        create_desktop_available=create_desktop_available,
        reason_code=reason_code,
    )


def _interactive_session_available() -> bool:
    session_name = os.environ.get("SESSIONNAME", "")
    if session_name.upper() == "SERVICES":
        return False
    try:
        current_session_id = _win32_process_session_id(os.getpid())
        return current_session_id == _win32_active_console_session_id()
    except Exception:
        logger.exception("Unable to determine Windows interactive session state")
        return False


def _input_desktop_available() -> bool:
    handle: Any | None = None
    try:
        handle = _win32_open_input_desktop()
        return True
    except Exception:
        logger.exception("Unable to open Windows input desktop")
        return False
    finally:
        if handle is not None:
            try:
                _win32_close_desktop(handle)
            except Exception:
                logger.exception("Unable to close Windows input desktop handle")


def _create_desktop_available() -> bool:
    handle: Any | None = None
    try:
        handle = _win32_create_desktop(f"GUIClaw-Probe-{uuid.uuid4().hex[:8]}")
        return True
    except Exception:
        logger.exception("Unable to create Windows probe desktop")
        return False
    finally:
        if handle is not None:
            try:
                _win32_close_desktop(handle)
            except Exception:
                logger.exception("Unable to close Windows probe desktop handle")


def _win32_create_desktop(desktop_name: str) -> Any:
    user32, wintypes = _load_user32()
    create_desktop = user32.CreateDesktopW
    create_desktop.argtypes = [
        wintypes.LPWSTR,
        wintypes.LPWSTR,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    create_desktop.restype = wintypes.HANDLE
    handle = create_desktop(desktop_name, None, None, 0, _DESKTOP_ACCESS_MASK, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _win32_close_desktop(handle: Any) -> None:
    user32, wintypes = _load_user32()
    close_desktop = user32.CloseDesktop
    close_desktop.argtypes = [wintypes.HANDLE]
    close_desktop.restype = wintypes.BOOL
    if not close_desktop(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _win32_open_input_desktop() -> Any:
    user32, wintypes = _load_user32()
    open_input_desktop = user32.OpenInputDesktop
    open_input_desktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_input_desktop.restype = wintypes.HANDLE
    handle = open_input_desktop(0, False, _DESKTOP_ACCESS_MASK)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _win32_process_session_id(process_id: int) -> int:
    kernel32, wintypes = _load_kernel32()
    process_id_to_session_id = kernel32.ProcessIdToSessionId
    process_id_to_session_id.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    process_id_to_session_id.restype = wintypes.BOOL
    session_id = wintypes.DWORD()
    if not process_id_to_session_id(process_id, ctypes.byref(session_id)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(session_id.value)


def _win32_active_console_session_id() -> int:
    kernel32, _ = _load_kernel32()
    active_console_session_id = kernel32.WTSGetActiveConsoleSessionId
    active_console_session_id.argtypes = []
    active_console_session_id.restype = ctypes.c_uint32
    return int(active_console_session_id())


def _load_user32() -> tuple[Any, Any]:
    return _load_win32_library("user32")


def _load_kernel32() -> tuple[Any, Any]:
    return _load_win32_library("kernel32")


def _load_win32_library(name: str) -> tuple[Any, Any]:
    if sys.platform != "win32":
        raise RuntimeError(f"{name} Windows APIs require Windows")
    windll = getattr(ctypes, "WinDLL", None)
    if windll is None:
        raise RuntimeError("ctypes.WinDLL is unavailable on this host")
    from ctypes import wintypes

    return windll(name, use_last_error=True), wintypes
