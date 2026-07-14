"""
guiclaw.backends.hdc
====================
HarmonyOS Device Connector (HDC) backend for device automation.

Drives a connected HarmonyOS device via the ``hdc`` CLI tool distributed with
the HarmonyOS SDK.  Supports tap, swipe, text input, app launch/close, and
screenshot capture — the full DeviceBackend protocol.

All I/O is non-blocking via asyncio.create_subprocess_exec.
"""

from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path

from guiclaw.action import Action, describe_action, resolve_coordinate
from guiclaw.backends import read_png_size
from guiclaw.backends.keycodes import HDC_KEYCODE_MAP as _HDC_KEYCODE_MAP
from guiclaw.backends.keycodes import canonical_key_name
from guiclaw.observation import Observation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEVICE_SCREENSHOT_PATH = "/data/local/tmp/__guiclaw_cap.jpeg"

_RENDER_SERVICE_RE = re.compile(r"(\d{3,4})\s*[xX]\s*(\d{3,4})")
# aa dump -l output looks like: bundle_name #string[com.example.app]  state #FOREGROUND
_AA_BUNDLE_RE = re.compile(r"bundle_name\s+#string\[([^\]]+)\]")
_AA_FOREGROUND_BLOCK_RE = re.compile(
    r"(bundle_name\s+#string\[([^\]]+)\](?:(?!\nbundle_name).)*?state\s+#FOREGROUND)",
    re.DOTALL,
)

# Common HarmonyOS system bundles returned by list_apps() as a static fallback.
_COMMON_HARMONY_BUNDLES: list[str] = [
    "com.huawei.hmos.settings",
    "com.huawei.hmos.camera",
    "com.huawei.hmos.photos",
    "com.huawei.hmos.contacts",
    "com.huawei.hmos.mms",
    "com.huawei.hmos.calendar",
    "com.huawei.hmos.calculator",
    "com.huawei.hmos.clock",
    "com.huawei.hmos.filemanager",
    "com.huawei.hmos.browser",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class HdcError(Exception):
    """Raised when an hdc command exits with a non-zero return code."""

    def __init__(self, message: str, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


# ---------------------------------------------------------------------------
# PIL lazy import helper
# ---------------------------------------------------------------------------


def _import_pil_image() -> type:
    """Lazily import PIL.Image with a helpful installation hint on failure."""
    try:
        from PIL import Image  # type: ignore[import-untyped]  # noqa: PLC0415

        return Image
    except ImportError as exc:
        raise ImportError(
            "The 'Pillow' package is required for HDC screenshot conversion. "
            "Install it with: pip install Pillow"
        ) from exc


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class HdcBackend:
    """HarmonyOS device backend that drives the connected device via ``hdc``.

    Args:
        serial: Device serial identifier.  ``None`` uses the first available
                device reported by ``hdc list targets``.
        hdc_path: Path to the ``hdc`` binary.  Defaults to ``"hdc"`` (looks
                  up PATH).
    """

    def __init__(self, serial: str | None = None, hdc_path: str = "hdc") -> None:
        self._serial = serial
        self._hdc = hdc_path
        # Sensible HarmonyOS device defaults; refreshed on each observe() call.
        self._screen_width: int = 720
        self._screen_height: int = 1280

    @property
    def platform(self) -> str:
        return "harmonyos"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cmd(self, *args: str) -> list[str]:
        cmd: list[str] = [self._hdc]
        if self._serial:
            cmd += ["-t", self._serial]
        cmd += list(args)
        return cmd

    async def _run(self, *args: str, timeout: float = 10.0) -> str:
        """Run an hdc sub-command and return decoded stdout.

        Raises:
            HdcError: On non-zero exit code.
            TimeoutError: When the process does not complete within *timeout*.
        """
        cmd = self._build_cmd(*args)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise TimeoutError(f"hdc timed out after {timeout}s: {' '.join(cmd)}")

        stdout = stdout_bytes.decode(errors="replace").strip()
        stderr = stderr_bytes.decode(errors="replace").strip()
        if proc.returncode != 0:
            raise HdcError(
                f"hdc failed (exit {proc.returncode}): {' '.join(cmd)}"
                + (f"\nstderr: {stderr}" if stderr else ""),
                returncode=proc.returncode,
                stderr=stderr,
            )
        return stdout

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    async def preflight(self) -> None:
        """Verify that the target device is reachable via hdc.

        Raises:
            HdcError: When no device is found, or the specified serial is absent.
        """
        try:
            output = await self._run("list", "targets", timeout=15.0)
        except TimeoutError:
            raise HdcError("'hdc list targets' timed out during preflight")

        device_lines = [
            line.strip()
            for line in output.splitlines()
            if line.strip() and not line.strip().lower().startswith("[empty]")
        ]

        if self._serial:
            if self._serial not in device_lines:
                raise HdcError(
                    f"Device {self._serial!r} not found in 'hdc list targets'. "
                    f"Available: {device_lines or ['(none)']}"
                )
        else:
            if not device_lines:
                raise HdcError(
                    "No HarmonyOS device found. "
                    "Connect a device via USB or run 'hdc tconn <address>' for TCP."
                )

    # ------------------------------------------------------------------
    # App discovery
    # ------------------------------------------------------------------

    async def list_apps(self) -> list[str]:
        """Return bundle IDs of installed HarmonyOS apps.

        HarmonyOS does not expose a simple ``pm list packages`` equivalent via
        hdc, so a curated list of common system bundles is returned as a stable
        fallback.  This is sufficient for the LLM to resolve human-readable
        names such as "Settings" to ``com.huawei.hmos.settings``.
        """
        return list(_COMMON_HARMONY_BUNDLES)

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        """Capture screen state: screenshot, dimensions, and foreground app.

        The device screenshot is JPEG.  It is pulled to a temporary local path
        and then converted to PNG so consumers receive a consistent format.
        """
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Capture screenshot on device ---
        try:
            await self._run(
                "shell",
                "screenshot",
                _DEVICE_SCREENSHOT_PATH,
                timeout=timeout,
            )
        except (HdcError, TimeoutError):
            # Fallback to snapshot_display on older HarmonyOS builds.
            await self._run(
                "shell",
                "snapshot_display",
                "-f",
                _DEVICE_SCREENSHOT_PATH,
                timeout=timeout,
            )

        # --- Pull the JPEG to a local temp path ---
        jpeg_path = screenshot_path.with_suffix(".jpeg")
        await self._run(
            "file",
            "recv",
            _DEVICE_SCREENSHOT_PATH,
            str(jpeg_path),
            timeout=timeout,
        )

        # --- Convert JPEG → PNG (non-blocking, PIL in thread) ---
        image_module = _import_pil_image()

        def _convert() -> None:
            with image_module.open(str(jpeg_path)) as img:
                img.save(str(screenshot_path), format="PNG")
            jpeg_path.unlink(missing_ok=True)

        await asyncio.to_thread(_convert)

        screenshot_size = read_png_size(screenshot_path)
        if screenshot_size is None:
            # --- Screen size + foreground app in parallel ---
            (width, height), fg_app = await asyncio.gather(
                self._query_screen_size(timeout),
                self._query_foreground_app(timeout),
            )
        else:
            width, height = screenshot_size
            fg_app = await self._query_foreground_app(timeout)

        self._screen_width = width
        self._screen_height = height

        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=width,
            screen_height=height,
            foreground_app=fg_app,
            platform=self.platform,
        )

    async def _query_screen_size(self, timeout: float) -> tuple[int, int]:
        """Query physical screen resolution from RenderService."""
        try:
            output = await self._run(
                "shell",
                "hidumper",
                "-s",
                "RenderService",
                timeout=max(timeout, 5.0),
            )
            match = _RENDER_SERVICE_RE.search(output)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                return w, h
        except (HdcError, TimeoutError):
            pass
        return self._screen_width, self._screen_height

    async def _query_foreground_app(self, timeout: float) -> str:
        """Parse ``aa dump -l`` output for the FOREGROUND mission bundle name."""
        try:
            output = await self._run(
                "shell",
                "aa",
                "dump",
                "-l",
                timeout=max(timeout, 10.0),
            )
            # Find the mission block that contains "FOREGROUND" state
            fg_match = _AA_FOREGROUND_BLOCK_RE.search(output)
            if fg_match:
                # Extract bundle_name from inside that block
                bundle_match = _AA_BUNDLE_RE.search(fg_match.group(1))
                if bundle_match:
                    return bundle_match.group(1)
        except (HdcError, TimeoutError):
            pass
        return "unknown"

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        """Dispatch a single action to the HarmonyOS device via hdc uitest."""
        t = action.action_type

        if t == "tap":
            x, y = self._resolve_point(action)
            await self._run(
                "shell",
                "uitest",
                "uiInput",
                "click",
                str(x),
                str(y),
                timeout=timeout,
            )

        elif t == "double_tap":
            x, y = self._resolve_point(action)
            await self._run(
                "shell",
                "uitest",
                "uiInput",
                "doubleClick",
                str(x),
                str(y),
                timeout=timeout,
            )

        elif t == "long_press":
            x, y = self._resolve_point(action)
            await self._run(
                "shell",
                "uitest",
                "uiInput",
                "longClick",
                str(x),
                str(y),
                timeout=timeout,
            )

        elif t in ("drag", "swipe"):
            x1, y1 = self._resolve_point(action)
            x2, y2 = self._resolve_second_point(action)
            speed = _compute_swipe_speed(x1, y1, x2, y2, action.duration_ms)
            await self._run(
                "shell",
                "uitest",
                "uiInput",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(speed),
                timeout=timeout,
            )

        elif t == "input_text":
            text = action.text or ""
            if text:
                # HDC text input requires tap coordinates; use action coords or screen centre.
                if action.x is not None and action.y is not None:
                    tx = self._resolve_x(action.x, relative=action.relative)
                    ty = self._resolve_y(action.y, relative=action.relative)
                else:
                    tx = self._screen_width // 2
                    ty = self._screen_height // 2
                await self._run(
                    "shell",
                    "uitest",
                    "uiInput",
                    "inputText",
                    str(tx),
                    str(ty),
                    text,
                    timeout=timeout,
                )
            if action.auto_enter:
                await self._run(
                    "shell",
                    "uitest",
                    "uiInput",
                    "keyEvent",
                    "2054",
                    timeout=timeout,
                )

        elif t == "hotkey":
            keys = action.key or []
            for k in keys:
                keycode = _HDC_KEYCODE_MAP.get(canonical_key_name(k))
                if keycode is None:
                    raise ValueError(
                        f"Unknown HDC key {k!r}. Supported: {sorted(_HDC_KEYCODE_MAP.keys())}"
                    )
                await self._run(
                    "shell",
                    "uitest",
                    "uiInput",
                    "keyEvent",
                    keycode,
                    timeout=timeout,
                )

        elif t == "scroll":
            await self._do_scroll(action, timeout=timeout)

        elif t == "wait":
            await asyncio.sleep((action.duration_ms or 1000) / 1000.0)

        elif t == "back":
            await self._run(
                "shell",
                "uitest",
                "uiInput",
                "keyEvent",
                "Back",
                timeout=timeout,
            )

        elif t == "home":
            await self._run(
                "shell",
                "uitest",
                "uiInput",
                "keyEvent",
                "Home",
                timeout=timeout,
            )

        elif t == "done":
            pass  # terminal marker, no device command needed

        elif t == "open_app":
            spec = action.text or ""
            if spec:
                if "/" in spec:
                    bundle, ability = spec.split("/", 1)
                else:
                    bundle, ability = spec, "MainAbility"
                await self._run(
                    "shell",
                    "aa",
                    "start",
                    "-b",
                    bundle,
                    "-a",
                    ability,
                    timeout=timeout,
                )

        elif t == "close_app":
            bundle = action.text or ""
            if bundle:
                await self._run("shell", "aa", "force-stop", bundle, timeout=timeout)

        else:
            raise ValueError(f"Unsupported action type: {t!r}")

        return describe_action(action)

    # ------------------------------------------------------------------
    # Coordinate helpers  (mirror AdbBackend)
    # ------------------------------------------------------------------

    def _resolve_x(self, value: float, *, relative: bool) -> int:
        return resolve_coordinate(value, self._screen_width, relative=relative)

    def _resolve_y(self, value: float, *, relative: bool) -> int:
        return resolve_coordinate(value, self._screen_height, relative=relative)

    def _resolve_point(self, action: Action) -> tuple[int, int]:
        if action.x is None or action.y is None:
            raise ValueError(f"Action {action.action_type!r} requires coordinates.")
        return (
            self._resolve_x(action.x, relative=action.relative),
            self._resolve_y(action.y, relative=action.relative),
        )

    def _resolve_second_point(self, action: Action) -> tuple[int, int]:
        if action.x2 is None or action.y2 is None:
            raise ValueError(f"Action {action.action_type!r} requires end-point coordinates.")
        return (
            self._resolve_x(action.x2, relative=action.relative),
            self._resolve_y(action.y2, relative=action.relative),
        )

    async def _do_scroll(self, action: Action, *, timeout: float) -> None:
        """Simulate scroll via a short swipe gesture."""
        x = self._screen_width // 2
        y = self._screen_height // 2
        if action.x is not None and action.y is not None:
            x = self._resolve_x(action.x, relative=action.relative)
            y = self._resolve_y(action.y, relative=action.relative)

        pixels = abs(action.pixels or 200)
        direction = (action.text or "down").lower()

        # Touch scrolls are inverted relative to content direction:
        # "down" means reveal lower content, so the finger moves upward.
        if direction == "up":
            x2, y2 = x, y + pixels
        elif direction == "down":
            x2, y2 = x, y - pixels
        elif direction == "left":
            x2, y2 = x - pixels, y
        else:  # right
            x2, y2 = x + pixels, y

        speed = _compute_swipe_speed(x, y, x2, y2, action.duration_ms)
        await self._run(
            "shell",
            "uitest",
            "uiInput",
            "swipe",
            str(x),
            str(y),
            str(x2),
            str(y2),
            str(speed),
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Speed computation helper
# ---------------------------------------------------------------------------


def _compute_swipe_speed(x1: int, y1: int, x2: int, y2: int, duration_ms: int | None) -> int:
    """Convert a pixel distance + optional duration to a px/s speed value.

    HDC uitest uiInput swipe accepts speed in pixels per second.  When no
    explicit duration is requested, a default of 2000 px/s is returned.
    """
    if duration_ms is None or duration_ms <= 0:
        return 2000
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance == 0:
        return 2000
    speed = distance / (duration_ms / 1000.0)
    return max(1, int(speed))
