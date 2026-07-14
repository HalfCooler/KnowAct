"""
guiclaw.backends.desktop
========================
LocalDesktopBackend — macOS/Linux/Windows GUI automation via pyautogui + mss.

Design notes:
- mss is imported at module level (fast, pure Python, no display dependency).
- pyautogui and pyperclip are imported lazily inside methods because pyautogui
  triggers DPI-awareness calls at import time which can fail in headless CI.
- Text input uses pyperclip clipboard-paste, not pyautogui.typewrite, for
  reliable Unicode support.
- HiDPI (Retina) monitors are handled by comparing mss physical pixels against
  the monitor's logical width/height and downscaling via Pillow.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
from asyncio.subprocess import PIPE
from pathlib import Path
from types import SimpleNamespace

try:
    import mss
except ImportError:  # pragma: no cover
    mss = SimpleNamespace(mss=None)
from PIL import Image

from guiclaw.action import Action, describe_action, resolve_coordinate
from guiclaw.backends.virtual_display import DisplayInfo
from guiclaw.observation import Observation

# pyautogui and pyperclip are optional desktop dependencies.  Import them at
# module level so that patch("guiclaw.backends.desktop.pyautogui") works in
# tests.  Callers should install the `desktop` extra before using this backend.
try:
    import pyautogui
    import pyperclip
except ImportError:  # pragma: no cover
    pyautogui = None  # type: ignore[assignment]
    pyperclip = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform-aware modifier key normalisation
# ---------------------------------------------------------------------------

# Maps shorthand modifier names to the canonical pyautogui key names per
# sys.platform.  Keys not listed fall through unchanged (lowercase).
_MODIFIER_MAP: dict[str, dict[str, str]] = {
    "darwin": {
        "cmd": "command",
        "option": "alt",
        "super": "command",
        "win": "command",
    },
    "linux": {
        "cmd": "win",
        "super": "win",
        "option": "alt",
        "meta": "win",
    },
    "win32": {
        "cmd": "win",
        "option": "alt",
        "super": "win",
        "meta": "win",
    },
}


class LocalDesktopBackend:
    """Device backend that automates the local desktop using pyautogui + mss.

    Args:
        None — platform is detected automatically from :func:`platform.system`.
    """

    def __init__(self) -> None:
        if pyautogui is not None:
            pyautogui.PAUSE = 0.0  # disable built-in per-call delay

        system = platform.system()
        if system == "Darwin":
            self._platform = "macos"
        elif system == "Linux":
            self._platform = "linux"
        else:
            self._platform = "windows"

        self._screen_width: int = 0
        self._screen_height: int = 0
        self._monitor_left: int = 0
        self._monitor_top: int = 0
        self._target_display: DisplayInfo | None = None

    # ------------------------------------------------------------------
    # DeviceBackend protocol
    # ------------------------------------------------------------------

    @property
    def platform(self) -> str:
        """Return the canonical platform identifier."""
        return self._platform

    def configure_target_display(self, display_info: DisplayInfo | None) -> None:
        """Set the display metadata used by observe() for monitor selection."""
        self._target_display = display_info

    async def preflight(self) -> None:
        """Verify that pyautogui can access the display (accessibility check).

        Raises:
            RuntimeError: when the process lacks accessibility permissions,
                with instructions on how to grant them.
        """
        try:
            pyautogui.position()
        except Exception as exc:
            raise RuntimeError(
                "Enable Accessibility for Terminal/iTerm in "
                "System Settings > Privacy & Security > Accessibility. "
                f"Original error: {exc}"
            ) from exc

    async def observe(
        self,
        screenshot_path: Path,
        timeout: float = 5.0,
    ) -> Observation:
        """Capture a screenshot, downscale on HiDPI, and return an Observation.

        The screenshot is saved as PNG at *screenshot_path* (parent directories
        are created automatically).  On HiDPI / Retina displays mss returns
        physical pixels; the image is resized to the monitor's logical width and
        height before saving so the LLM receives coordinates it can address.

        Args:
            screenshot_path: Where to write the PNG file.
            timeout: Not used for local screen capture (kept for protocol compat).

        Returns:
            :class:`~guiclaw.observation.Observation` with logical screen dims.
        """
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        capture = getattr(mss, "mss", None)
        if capture is None:
            raise RuntimeError(
                "Install `mss` to use LocalDesktopBackend screen capture "
                "(optional dependency in the `desktop` extra)."
            )

        with capture() as sct:
            monitor_index = 1
            if self._target_display is not None:
                monitor_index = self._target_display.monitor_index
            available_monitor_count = max(0, len(sct.monitors) - 1)
            if available_monitor_count == 0:
                raise RuntimeError("No desktop monitors detected for screenshot capture")
            if monitor_index < 1 or monitor_index > available_monitor_count:
                logger.warning(
                    "Configured monitor index %s unavailable; falling back to primary monitor. "
                    "available_monitors=%s",
                    monitor_index,
                    available_monitor_count,
                )
                monitor_index = 1
            monitor = sct.monitors[monitor_index]
            logical_w: int = monitor["width"]
            logical_h: int = monitor["height"]
            sct_img = sct.grab(monitor)

        physical_w, physical_h = sct_img.size
        img = Image.frombytes("RGB", (physical_w, physical_h), sct_img.bgra, "raw", "BGRX")

        if physical_w != logical_w or physical_h != logical_h:
            img = img.resize((logical_w, logical_h), Image.LANCZOS)

        img.save(str(screenshot_path), "PNG")

        self._screen_width = logical_w
        self._screen_height = logical_h
        self._monitor_left = monitor.get("left", 0)
        self._monitor_top = monitor.get("top", 0)

        fg_app = await self._query_foreground_app()
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=logical_w,
            screen_height=logical_h,
            foreground_app=fg_app,
            platform=self._platform,
        )

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        """Dispatch a GUI action using pyautogui.

        Args:
            action: Validated :class:`~guiclaw.action.Action` to execute.
            timeout: Upper bound for subprocess operations (open_app, close_app).

        Returns:
            Human-readable description of the action via :func:`describe_action`.

        Raises:
            ValueError: for unrecognised action types.
        """
        t = action.action_type

        if t == "tap":
            x, y = self._resolve_point(action)
            pyautogui.click(x, y)

        elif t == "double_tap":
            x, y = self._resolve_point(action)
            pyautogui.doubleClick(x, y)

        elif t == "long_press":
            x, y = self._resolve_point(action)
            pyautogui.rightClick(x, y)

        elif t == "drag":
            x1, y1 = self._resolve_point(action)
            x2, y2 = self._resolve_second_point(action)
            duration = (action.duration_ms or 300) / 1000.0
            pyautogui.mouseDown(x1, y1, button="left")
            pyautogui.dragTo(x2, y2, duration=duration, button="left")

        elif t == "swipe":
            x1, y1 = self._resolve_point(action)
            x2, y2 = self._resolve_second_point(action)
            duration = (action.duration_ms or 300) / 1000.0
            pyautogui.mouseDown(x1, y1, button="left")
            pyautogui.moveTo(x2, y2, duration=duration)
            pyautogui.mouseUp(x2, y2, button="left")

        elif t == "scroll":
            if action.x is not None and action.y is not None:
                scroll_x = self._resolve_x(action.x, relative=action.relative)
                scroll_y = self._resolve_y(action.y, relative=action.relative)
                pyautogui.moveTo(scroll_x, scroll_y)
            pixels = abs(action.pixels or 120)
            clicks = max(1, pixels // 120)
            direction = (action.text or "down").lower()
            if direction == "down":
                pyautogui.scroll(-clicks)
            elif direction == "up":
                pyautogui.scroll(clicks)
            elif direction == "left":
                pyautogui.hscroll(-clicks)
            else:  # right
                pyautogui.hscroll(clicks)

        elif t == "input_text":
            paste_key = "command" if self._platform == "macos" else "ctrl"
            pyperclip.copy(action.text or "")
            pyautogui.hotkey(paste_key, "v")

        elif t == "hotkey":
            keys = self._normalize_keys(action.key or [])
            pyautogui.hotkey(*keys)

        elif t == "wait":
            await asyncio.sleep((action.duration_ms or 1000) / 1000.0)

        elif t == "done":
            pass  # terminal — no device interaction

        elif t == "back":
            if self._platform == "macos":
                pyautogui.hotkey("command", "[")
            else:
                pyautogui.hotkey("alt", "left")

        elif t == "home":
            if self._platform == "macos":
                pyautogui.hotkey("command", "shift", "h")
            elif self._platform == "linux":
                pyautogui.hotkey("super")
            else:  # windows
                pyautogui.hotkey("win", "d")

        elif t == "open_app":
            app_name = action.text or ""
            if self._platform == "macos":
                await self._run_cmd("open", "-a", app_name, timeout=timeout)
            elif self._platform == "linux":
                await self._run_cmd("xdg-open", app_name, timeout=timeout)
            else:
                await self._run_cmd("cmd", "/c", "start", "", app_name, timeout=timeout)

        elif t == "close_app":
            app_name = action.text or ""
            await self._close_app(app_name, timeout=timeout)

        else:
            raise ValueError(f"Unsupported action type for desktop backend: {t!r}")

        return describe_action(action)

    # ------------------------------------------------------------------
    # App discovery
    # ------------------------------------------------------------------

    async def list_apps(self) -> list[str]:
        """Return application names available on the local desktop.

        - macOS: scans ``/Applications`` and ``~/Applications`` for ``.app`` bundles.
        - Linux: parses ``.desktop`` files under ``/usr/share/applications``.
        - Windows: returns an empty list (not yet implemented).
        """
        if self._platform == "macos":
            return await self._list_apps_macos()
        if self._platform == "linux":
            return await self._list_apps_linux()
        return []

    async def _list_apps_macos(self) -> list[str]:
        app_dirs = [Path("/Applications"), Path.home() / "Applications"]
        names: list[str] = []
        for d in app_dirs:
            if not d.is_dir():
                continue
            for entry in sorted(d.iterdir()):
                if entry.suffix == ".app" and entry.is_dir():
                    names.append(entry.stem)
        return names

    async def _list_apps_linux(self) -> list[str]:
        desktop_dir = Path("/usr/share/applications")
        if not desktop_dir.is_dir():
            return []
        names: list[str] = []
        for entry in sorted(desktop_dir.iterdir()):
            if entry.suffix != ".desktop":
                continue
            try:
                for line in entry.read_text(errors="replace").splitlines():
                    if line.startswith("Name="):
                        names.append(line[len("Name="):].strip())
                        break
            except OSError:
                continue
        return names

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_x(self, value: float, *, relative: bool) -> int:
        return resolve_coordinate(value, self._screen_width, relative=relative) + self._monitor_left

    def _resolve_y(self, value: float, *, relative: bool) -> int:
        return resolve_coordinate(value, self._screen_height, relative=relative) + self._monitor_top

    def _resolve_point(self, action: Action) -> tuple[int, int]:
        if action.x is None or action.y is None:
            raise ValueError(f"Action {action.action_type!r} requires x and y coordinates.")
        return (
            self._resolve_x(action.x, relative=action.relative),
            self._resolve_y(action.y, relative=action.relative),
        )

    def _resolve_second_point(self, action: Action) -> tuple[int, int]:
        if action.x2 is None or action.y2 is None:
            raise ValueError(f"Action {action.action_type!r} requires x2 and y2 end-point coordinates.")
        return (
            self._resolve_x(action.x2, relative=action.relative),
            self._resolve_y(action.y2, relative=action.relative),
        )

    def _normalize_keys(self, keys: list[str]) -> list[str]:
        """Map shorthand modifier names to canonical pyautogui key names.

        Selects the per-platform map based on :data:`sys.platform`.
        Unrecognised keys are lowercased and passed through unchanged.
        """
        platform_key = sys.platform  # "darwin", "linux", "win32"
        modifier_map = _MODIFIER_MAP.get(platform_key, {})
        return [modifier_map.get(k.lower(), k.lower()) for k in keys]

    async def _run_cmd(self, *args: str, timeout: float = 5.0) -> str:
        """Run an external command asynchronously and return stdout.

        Returns an empty string on timeout or non-zero exit without raising,
        so that open_app / close_app failures degrade gracefully.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=PIPE,
                stderr=PIPE,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                return ""
            return stdout_bytes.decode(errors="replace").strip()
        except Exception:
            return ""

    async def _close_app(self, app_name: str, timeout: float = 5.0) -> None:
        """Close an application gracefully, falling back to force-kill."""
        if self._platform == "macos":
            # Graceful AppleScript quit
            await self._run_cmd(
                "osascript", "-e", f'quit app "{app_name}"',
                timeout=timeout,
            )
            # Force-kill any remaining processes with that name
            await self._run_cmd("pkill", "-f", app_name, timeout=timeout)
        elif self._platform == "linux":
            await self._run_cmd("pkill", "-f", app_name, timeout=timeout)
        else:  # windows
            await self._run_cmd(
                "taskkill", "/IM", f"{app_name}.exe",
                timeout=timeout,
            )

    async def _query_foreground_app(self) -> str:
        """Return the name of the currently active application.

        Uses platform-specific commands:
        - macOS: osascript (System Events)
        - Linux: xdotool + /proc/{pid}/comm
        - Windows: stub returning "unknown"
        """
        try:
            if self._platform == "macos":
                return await self._query_foreground_app_macos()
            elif self._platform == "linux":
                return await self._query_foreground_app_linux()
            else:
                return "unknown"
        except Exception:
            return "unknown"

    async def _query_foreground_app_macos(self) -> str:
        script = (
            "tell application \"System Events\" "
            "to get name of first process where it is frontmost"
        )
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), 5.0)
        return stdout_bytes.decode(errors="replace").strip()

    async def _query_foreground_app_linux(self) -> str:
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "getactivewindow", "getwindowpid",
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), 5.0)
        pid = stdout_bytes.decode().strip()
        if not pid.isdigit():
            return "unknown"
        comm_path = Path(f"/proc/{pid}/comm")
        if comm_path.exists():
            return comm_path.read_text().strip()
        return "unknown"
