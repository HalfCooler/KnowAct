"""
guiclaw.backends.adb
====================
Android Debug Bridge backend for device and emulator automation.

Unicode text input prefers ADBKeyboard IME on the target device:
    adb shell ime set com.android.adbkeyboard/.AdbIME

If ADBKeyboard is unavailable, GUIClaw falls back to a device-side `yadb`
helper when it is installed at `/data/local/tmp/yadb`.

All I/O is non-blocking via asyncio.create_subprocess_exec.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.resources
import inspect
import json
import logging
import os
import re
import shlex
import tempfile
import threading
import time
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

from guiclaw.action import Action, describe_action, resolve_coordinate
from guiclaw.backends import read_png_size
from guiclaw.backends.adb_command import AdbCommandRunner
from guiclaw.backends.keycodes import ANDROID_KEYCODE_MAP as _KEYCODE_MAP
from guiclaw.backends.keycodes import canonical_key_name
from guiclaw.observation import Observation

logger = logging.getLogger(__name__)

_WM_SIZE_RE = re.compile(r"Physical size:\s*(\d+)x(\d+)", re.IGNORECASE)
_RESUMED_ACTIVITY_RE = re.compile(
    r"mResumedActivity.*?"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/",
    re.IGNORECASE,
)
_TOP_RESUMED_ACTIVITY_RE = re.compile(
    r"topResumedActivity.*?"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/",
    re.IGNORECASE,
)
_CURRENT_FOCUS_RE = re.compile(
    r"mCurrentFocus.*?"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/",
    re.IGNORECASE,
)
_FOCUSED_APP_RE = re.compile(
    r"mFocusedApp.*?"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/",
    re.IGNORECASE,
)

_DEVICE_SCREENSHOT_PATH = "/sdcard/__guiclaw_cap.png"
_DEVICE_UI_XML_PATH = "/sdcard/window.xml"
_UI_TREE_DUMP_TIMEOUT_SECONDS = 15.0
_ADB_KEYBOARD_IME = "com.android.adbkeyboard/.AdbIME"
_YADB_PATH = "/data/local/tmp/yadb"
_YADB_MAIN_CLASS = "com.ysbing.yadb.Main"

_KEYCOMBINATION_UNSUPPORTED_MARKERS = (
    "unknown command: keycombination",
    "invalid arguments for command: keycombination",
)
_ANDROID_PACKAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$")
_ANDROID_COMPONENT_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+/"
    r"(?:[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*|\.[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*)$"
)
_ANDROID_INTENT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*$")
_ANDROID_EXTRA_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_ANDROID_EXTRA_STREAM = "android.intent.extra.STREAM"
_APP_CONTEXT_LAUNCHER_COMPONENT_ENV = "GUICLAW_APP_CONTEXT_LAUNCHER_COMPONENT"
_AM_START_FAILURE_MARKERS = (
    "error:",
    "exception",
    "unable to resolve intent",
    "activity not started",
)


def _am_start_output_failed(output: str) -> bool:
    lowered = (output or "").lower()
    return any(marker in lowered for marker in _AM_START_FAILURE_MARKERS)


def _is_android_uri_extra(key: str, value: object) -> bool:
    if key != _ANDROID_EXTRA_STREAM:
        return False
    text = "" if value is None else str(value)
    return text.startswith(("content://", "file://", "http://", "https://"))


def _without_am_start_wait(remote_args: list[str]) -> list[str]:
    return [arg for arg in remote_args if arg != "-W"]


def _without_am_start_option(remote_args: list[str], option: str) -> list[str]:
    trimmed: list[str] = []
    skip_next = False
    for arg in remote_args:
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            skip_next = True
            continue
        trimmed.append(arg)
    return trimmed


def _am_start_data_scheme(remote_args: list[str]) -> str | None:
    try:
        data_uri = remote_args[remote_args.index("-d") + 1]
    except (ValueError, IndexError):
        return None
    return data_uri.split(":", 1)[0].lower() if ":" in data_uri else None


def _annotate_am_start_launch_variant(output: str, variant_name: str) -> str:
    marker = f"[guiclaw_launch_variant={variant_name}]"
    cleaned = str(output or "").rstrip()
    return f"{cleaned}\n{marker}" if cleaned else marker


def _am_start_launch_variant_marker(output: str) -> str | None:
    match = re.search(r"\[guiclaw_launch_variant=[A-Za-z0-9_:-]+\]", str(output or ""))
    return match.group(0) if match else None


@dataclass(frozen=True)
class ScrcpyFrameSnapshot:
    """Metadata for the latest scrcpy frame persisted to disk."""

    width: int
    height: int
    timestamp: float


class ScrcpyFrameSourceProtocol(Protocol):
    """Small protocol used by ``AdbBackend`` and tests."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def save_latest(
        self, path: Path, *, timeout_s: float, max_age_s: float
    ) -> ScrcpyFrameSnapshot: ...


def _pil_image_from_frame(frame: Any) -> Any:
    """Normalize py-scrcpy frame objects into a PIL Image."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised through preflight text
        raise RuntimeError(
            'ADB scrcpy capture requires Pillow. Install with `uv pip install -e ".[demo-live]"`.'
        ) from exc

    if isinstance(frame, Image.Image):
        return frame

    shape = getattr(frame, "shape", None)
    if shape is not None:
        if len(shape) >= 3 and shape[2] == 3:
            frame = frame[..., ::-1]
        elif len(shape) >= 3 and shape[2] == 4:
            frame = frame[..., [2, 1, 0, 3]]
        return Image.fromarray(frame)

    raise TypeError(f"Unsupported scrcpy frame object: {type(frame).__name__}")


def _frame_dimensions(frame: Any) -> tuple[int, int]:
    if hasattr(frame, "size") and not hasattr(frame, "shape"):
        width, height = frame.size
        return int(width), int(height)
    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    image = _pil_image_from_frame(frame)
    width, height = image.size
    return int(width), int(height)


def _jpeg_bytes_from_frame(frame: Any, *, quality: int) -> bytes:
    image = _pil_image_from_frame(frame)
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    return buf.getvalue()


class ScrcpyFrameSource:
    """Owns the py-scrcpy-sdk client and latest-frame buffer for ADB observe()."""

    def __init__(
        self,
        *,
        serial: str | None = None,
        adb_path: str = "adb",
        max_fps: int = 12,
        jpeg_quality: int = 80,
        frame_timeout_ms: int = 3000,
        on_jpeg_frame: Callable[[bytes, dict[str, Any]], None] | None = None,
    ) -> None:
        self._serial = serial
        self._adb_path = adb_path
        self._max_fps = max_fps
        self._jpeg_quality = jpeg_quality
        self._frame_timeout_ms = frame_timeout_ms
        self._on_jpeg_frame = on_jpeg_frame
        self._client: Any | None = None
        self._listener: Any | None = None
        self._latest_frame: Any | None = None
        self._latest_ts = 0.0
        self._reader_error: Exception | None = None
        self._started = False
        self._stopping = False
        self._condition = threading.Condition()

    def start(self) -> None:
        if self._started:
            return
        try:
            from py_scrcpy_sdk import ScrcpyClient, ScrcpyConfig
        except ImportError as exc:
            raise RuntimeError(
                "ADB scrcpy capture requires py-scrcpy-sdk==0.1.2. "
                'Install with `uv pip install -e ".[demo-live]"`.'
            ) from exc

        config_kwargs: dict[str, Any] = {
            "serial": self._serial,
            "adb_path": self._adb_path,
            "frame_timeout": self._frame_timeout_ms / 1000.0,
        }
        if self._max_fps > 0:
            config_kwargs["max_fps"] = self._max_fps
        config = self._construct_config(ScrcpyConfig, config_kwargs)
        client = ScrcpyClient(config)
        client.start()
        self._client = client
        self._stopping = False
        with self._condition:
            self._reader_error = None

        initial_frame = self._wait_until_ready(client)
        if initial_frame is not None:
            self._set_latest_frame(initial_frame)

        # py-scrcpy-sdk's helper listener lets thread exceptions escape to
        # stderr. Keep the listener under our control so reader failures are
        # surfaced through save_latest(), where AdbBackend can restart or
        # fall back to adb screencap.
        self._listener = threading.Thread(
            target=self._listen_forever,
            name="guiclaw-scrcpy-frame-listener",
            daemon=True,
        )
        self._listener.start()

        self._started = True

    @staticmethod
    def _construct_config(config_cls: Any, values: dict[str, Any]) -> Any:
        try:
            signature = inspect.signature(config_cls)
        except (TypeError, ValueError):
            return config_cls(**values)
        accepted = {
            name
            for name, param in signature.parameters.items()
            if param.kind in {param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY}
        }
        filtered = {key: value for key, value in values.items() if key in accepted}
        return config_cls(**filtered)

    @staticmethod
    def _wait_until_ready(client: Any) -> Any | None:
        ready = getattr(client, "wait_until_ready", None)
        if not callable(ready):
            get_frame = getattr(client, "get_frame", None)
            return get_frame() if callable(get_frame) else None
        try:
            return ready(timeout=5)
        except TypeError:
            return ready()

    def _listen_forever(self) -> None:
        frames = getattr(self._client, "frames", None)
        if not callable(frames):
            return
        try:
            for frame in frames():
                if self._on_frame(frame) is False:
                    break
        except Exception as exc:
            with self._condition:
                if not self._stopping:
                    self._reader_error = exc
                self._condition.notify_all()

    def _on_frame(self, frame: Any) -> bool:
        self._set_latest_frame(frame)
        return not self._stopping

    def _set_latest_frame(self, frame: Any) -> None:
        ts = time.time()
        with self._condition:
            self._latest_frame = frame
            self._latest_ts = ts
            self._reader_error = None
            self._condition.notify_all()
        if self._on_jpeg_frame is not None:
            try:
                width, height = _frame_dimensions(frame)
                self._on_jpeg_frame(
                    _jpeg_bytes_from_frame(frame, quality=self._jpeg_quality),
                    {"width": width, "height": height, "timestamp": ts, "source": "scrcpy"},
                )
            except Exception:
                pass

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        deadline = time.time() + timeout_s
        with self._condition:
            while True:
                frame = self._latest_frame
                ts = self._latest_ts
                reader_error = self._reader_error
                now = time.time()
                if reader_error is not None:
                    raise RuntimeError("scrcpy frame reader failed") from reader_error
                if frame is not None and (max_age_s <= 0 or now - ts <= max_age_s):
                    break
                remaining = deadline - now
                if remaining <= 0:
                    break
                self._condition.wait(timeout=max(0.01, remaining))

        if reader_error is not None and (
            frame is None or (max_age_s > 0 and time.time() - ts > max_age_s)
        ):
            raise RuntimeError("scrcpy frame reader failed") from reader_error
        if frame is None:
            raise TimeoutError(f"scrcpy frame not available within {timeout_s:.2f}s")
        if max_age_s > 0 and time.time() - ts > max_age_s:
            raise TimeoutError(
                f"latest scrcpy frame is stale ({time.time() - ts:.2f}s > {max_age_s:.2f}s)"
            )

        image = _pil_image_from_frame(frame)
        width, height = _frame_dimensions(frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return ScrcpyFrameSnapshot(width=width, height=height, timestamp=ts)

    def stop(self) -> None:
        self._stopping = True
        client = self._client
        if client is not None:
            stop = getattr(client, "stop", None)
            if callable(stop):
                stop()
        with self._condition:
            self._latest_frame = None
            self._latest_ts = 0.0
            self._reader_error = None
            self._condition.notify_all()
        self._listener = None
        self._client = None
        self._started = False


class AdbError(Exception):
    """Raised when an adb command fails."""

    def __init__(self, message: str, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


# -- root probe error classification -----------------------------------------

_ADB_TRANSPORT_ERROR_SUBSTRINGS: tuple[str, ...] = (
    "device offline",
    "device unauthorized",
    "no devices/emulators found",
    "device not found",
    "more than one device/emulator",
)


def _is_expected_root_probe_failure(exc: Exception) -> bool:
    """Return True when *exc* is an expected "root not available" response.

    Transport errors (device offline, etc.) return False so they propagate.
    Root-unavailable errors (permission denied, su not found) return True.
    Unknown AdbError variants also return False.
    """
    if isinstance(exc, AdbError):
        detail = f"{exc.stderr}\n{exc}".lower()
        # Re-raise transport errors so callers can detect connection problems
        if any(marker in detail for marker in _ADB_TRANSPORT_ERROR_SUBSTRINGS):
            return False
        # Expected "no root" signatures — device has no su, or su rejects us
        if any(
            marker in detail
            for marker in (
                "permission denied",
                "operation not permitted",
                "not permitted",
                "inaccessible or not found",
                "su: inaccessible",
                "su: not found",
                "not found",
                "invalid uid/gid",
                "invalid option",
                "usage: su",
            )
        ):
            return True
        # Unknown adb error — propagate for visibility
        return False
    # Timeout or unexpected exception type — propagate
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return False
    return False


def _escape_shell_text(text: str) -> str:
    """Escape text for ``adb shell input text``."""
    # Android's `input text` command treats `%s` as a space placeholder.
    # We are not invoking a shell here, so `\ ` would be passed literally and
    # can cause whitespace truncation on device-side parsing.
    special_chars = frozenset(r'\`$"!&|<>(){}[];#~*?^')
    escaped: list[str] = []
    for ch in text:
        if ch == " ":
            escaped.append("%s")
        elif ch in special_chars:
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)


def _is_ascii_safe(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _to_b64_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _is_emojiish_char(ch: str) -> bool:
    codepoint = ord(ch)
    if ch in ("\u200d", "\ufe0e", "\ufe0f", "\u20e3"):
        return True
    if 0x1F1E6 <= codepoint <= 0x1F1FF:  # Regional indicators
        return True
    if 0x1F3FB <= codepoint <= 0x1F3FF:  # Skin tone modifiers
        return True
    if 0x2600 <= codepoint <= 0x27BF:  # Misc symbols / dingbats often used as emoji
        return True
    if 0x1F000 <= codepoint <= 0x1FAFF:  # Main emoji-heavy planes
        return True
    return unicodedata.category(ch) == "So"


def _iter_text_input_segments(text: str) -> list[str]:
    """Split text into stable input chunks, isolating emoji-ish sequences.

    Some Android IME injection paths may truncate text that follows an emoji
    when the whole string is sent in one batch. By sending normal-text runs and
    emoji clusters separately, later text still lands even if the IME treats
    emoji boundaries specially.
    """
    if not text:
        return []

    segments: list[str] = []
    current = text[0]
    current_is_emoji = _is_emojiish_char(text[0])

    for ch in text[1:]:
        ch_is_emoji = _is_emojiish_char(ch)
        if ch_is_emoji == current_is_emoji:
            current += ch
            continue
        segments.append(current)
        current = ch
        current_is_emoji = ch_is_emoji

    segments.append(current)
    return segments


class AdbBackend:
    """ADB backend for Android device/emulator automation.

    Observations use scrcpy frames by default and ADB for all device actions.
    The legacy ``adb shell screencap`` capture path remains available for
    focused tests and explicit fallback wiring.

    Args:
        serial: Device serial (e.g. "emulator-5554"). None = default device.
        adb_path: Path to adb binary.
    """

    def __init__(
        self,
        serial: str | None = None,
        adb_path: str = "adb",
        *,
        scrcpy_max_fps: int = 12,
        scrcpy_jpeg_quality: int = 80,
        scrcpy_frame_timeout_ms: int = 3000,
        scrcpy_max_frame_age_ms: int = 1000,
        frame_source: ScrcpyFrameSourceProtocol | None = None,
        on_jpeg_frame: Callable[[bytes, dict[str, Any]], None] | None = None,
        use_scrcpy: bool = True,
        collect_ui_tree: bool = False,
        collect_ui_tree_nodes: bool = False,
    ) -> None:
        self._serial = serial
        self._adb = adb_path
        self._screen_width = 1080
        self._screen_height = 1920
        self._capture_width = 1080
        self._capture_height = 1920
        self._scrcpy_frame_timeout_ms = scrcpy_frame_timeout_ms
        self._scrcpy_max_frame_age_ms = scrcpy_max_frame_age_ms
        self._use_scrcpy = use_scrcpy
        self._collect_ui_tree = collect_ui_tree
        self._collect_ui_tree_nodes = collect_ui_tree_nodes
        self._adb_command_runner = AdbCommandRunner(self._run)
        self._scrcpy_started = False
        self._frame_source = (
            (
                frame_source
                if frame_source is not None
                else ScrcpyFrameSource(
                    serial=serial,
                    adb_path=adb_path,
                    max_fps=scrcpy_max_fps,
                    jpeg_quality=scrcpy_jpeg_quality,
                    frame_timeout_ms=scrcpy_frame_timeout_ms,
                    on_jpeg_frame=on_jpeg_frame,
                )
            )
            if use_scrcpy
            else None
        )

    @property
    def platform(self) -> str:
        return "android"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_cmd(self, *args: str) -> list[str]:
        cmd: list[str] = [self._adb]
        if self._serial:
            cmd += ["-s", self._serial]
        cmd += list(args)
        return cmd

    async def _run(self, *args: str, timeout: float = 10.0) -> str:
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
            raise TimeoutError(f"adb timed out after {timeout}s: {' '.join(cmd)}")

        stdout = stdout_bytes.decode(errors="replace").strip()
        stderr = stderr_bytes.decode(errors="replace").strip()
        if proc.returncode != 0:
            raise AdbError(
                f"adb failed (exit {proc.returncode}): {' '.join(cmd)}"
                + (f"\nstderr: {stderr}" if stderr else ""),
                returncode=proc.returncode,
                stderr=stderr,
            )
        return stdout

    def _app_context_launcher_component(self) -> str | None:
        component = os.environ.get(_APP_CONTEXT_LAUNCHER_COMPONENT_ENV, "").strip()
        if not component:
            return None
        if not _ANDROID_COMPONENT_RE.match(component):
            logger.warning(
                "Ignoring invalid %s value: %s",
                _APP_CONTEXT_LAUNCHER_COMPONENT_ENV,
                component,
            )
            return None
        return component

    def supports_app_context_launcher(self) -> bool:
        """Return whether an app-context launch bridge is configured.

        The bridge is optional and intentionally outside the Action schema.
        When configured, deeplink probing can use it as another launch
        transport while still relying on post-state contract verification.
        """

        return self._app_context_launcher_component() is not None

    async def root_available(self, *, timeout: float = 3.0) -> bool:
        cached = getattr(self, "_root_available_cache", None)
        if cached is True:
            return True
        for probe in ("su 0 id", "su -c id"):
            try:
                output = await self._run("shell", probe, timeout=timeout)
            except AdbError as exc:
                if not _is_expected_root_probe_failure(exc):
                    raise
                logger.debug(
                    "ADB root probe not available: probe=%r error_type=%s",
                    probe,
                    type(exc).__name__,
                )
                continue
            except (asyncio.TimeoutError, TimeoutError):
                raise
            if "uid=0" in output or "root" in output.lower():
                self._root_available_cache = True
                return True
            logger.debug("ADB root probe did not succeed: probe=%r", probe)
        return False

    async def run_as_root(self, command: str, *, timeout: float = 10.0) -> str:
        """Run a read-only adb shell command through root when available.

        Deeplink discovery uses this only for stronger capture evidence
        (e.g. dumpsys visibility), never for privileged activity replay.
        """

        command = str(command or "").strip()
        if not command:
            raise ValueError("root command must not be empty")
        if not await self.root_available(timeout=min(timeout, 3.0)):
            raise AdbError("adb root shell is not available")

        quoted = shlex.quote(command)
        failures: list[str] = []
        for root_command in (f"su 0 sh -c {quoted}", f"su -c {quoted}"):
            try:
                return await self._run("shell", root_command, timeout=timeout)
            except Exception as exc:
                failures.append(str(exc))
        raise AdbError("adb root shell failed: " + " | ".join(failures[-2:]))

    async def open_deeplink_app_context(self, action: Action, *, timeout: float = 10.0) -> str:
        if action.action_type != "open_deeplink":
            raise ValueError("open_deeplink_app_context requires action_type='open_deeplink'")
        return await self._run_app_context_launcher(action, timeout=timeout)

    async def open_intent_app_context(self, action: Action, *, timeout: float = 10.0) -> str:
        if action.action_type != "open_intent":
            raise ValueError("open_intent_app_context requires action_type='open_intent'")
        return await self._run_app_context_launcher(action, timeout=timeout)

    async def _run_app_context_launcher(self, action: Action, *, timeout: float) -> str:
        launcher_component = self._app_context_launcher_component()
        if not launcher_component:
            raise AdbError(
                f"app-context launcher is not configured; set {_APP_CONTEXT_LAUNCHER_COMPONENT_ENV}"
            )

        intent_action = (
            action.intent_action
            if action.action_type == "open_intent"
            else "android.intent.action.VIEW"
        )
        remote_args = [
            "am",
            "start",
            "-W",
            "-n",
            launcher_component,
            "--es",
            "guiclaw.intent_action",
            intent_action,
        ]
        if action.text:
            remote_args.extend(["--es", "guiclaw.data_uri", str(action.text)])
        if action.package:
            remote_args.extend(["--es", "guiclaw.package", str(action.package)])
        if action.component:
            remote_args.extend(["--es", "guiclaw.component", str(action.component)])
        if action.mime_type:
            remote_args.extend(["--es", "guiclaw.mime_type", str(action.mime_type)])
        if action.categories:
            remote_args.extend(["--esa", "guiclaw.categories", ",".join(action.categories)])
        if action.extras:
            extras_payload = json.dumps(
                {str(key): value for key, value in action.extras},
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            remote_args.extend(["--es", "guiclaw.extras_json", extras_payload])

        remote_cmd = " ".join(shlex.quote(str(arg)) for arg in remote_args)
        output = await self._run("shell", remote_cmd, timeout=timeout)
        if _am_start_output_failed(output):
            raise AdbError(f"app-context launcher failed: {output}")
        return _annotate_am_start_launch_variant(output, "app_context")

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    async def preflight(self) -> None:
        cmd = [self._adb, "devices"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        except asyncio.TimeoutError:
            raise AdbError("'adb devices' timed out during preflight")

        output = stdout_bytes.decode(errors="replace")
        device_lines = [
            line for line in output.splitlines() if "\t" in line and not line.startswith("List of")
        ]

        if self._serial:
            serial_ready = False
            for line in device_lines:
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[0].strip() == self._serial:
                    if parts[1].strip() != "device":
                        raise AdbError(
                            f"Device {self._serial!r} in state {parts[1].strip()!r}, "
                            "not 'device'. Check authorisation."
                        )
                    serial_ready = True
                    break
            if not serial_ready:
                raise AdbError(f"Device {self._serial!r} not found in 'adb devices'.")
        else:
            ready = [line for line in device_lines if line.split("\t", 1)[-1].strip() == "device"]
            if not ready:
                raise AdbError("No Android device found. Connect a device or start an emulator.")

        if self._use_scrcpy:
            try:
                await self._ensure_scrcpy_started()
            except Exception as exc:
                logger.warning("ADB scrcpy preflight failed; falling back to screencap: %s", exc)
                self._use_scrcpy = False

    # ------------------------------------------------------------------
    # App discovery
    # ------------------------------------------------------------------

    async def list_apps(self) -> list[str]:
        """Return package names of launchable apps on the device.

        Fetches third-party packages (``pm list packages -3``) and merges a
        small set of commonly-used system packages so the LLM can resolve
        human-readable names like "Settings" to ``com.android.settings``.
        """
        common_system_packages = [
            "com.android.settings",
            "com.android.contacts",
            "com.android.dialer",
            "com.android.mms",
            "com.android.camera2",
            "com.android.gallery3d",
            "com.android.calculator2",
            "com.android.calendar",
            "com.android.deskclock",
            "com.android.documentsui",
            "com.android.vending",
            "com.google.android.apps.messaging",
            "com.google.android.apps.photos",
            "com.google.android.gm",
            "com.google.android.googlequicksearchbox",
            "com.google.android.youtube",
            "com.google.android.apps.maps",
            "com.google.android.dialer",
            "com.google.android.contacts",
            "com.google.android.calendar",
        ]
        try:
            output = await self._run("shell", "pm", "list", "packages", "-3", timeout=10.0)
        except (AdbError, TimeoutError):
            return list(common_system_packages)

        packages: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("package:"):
                packages.append(line[len("package:") :])
        # Merge common system packages (deduplicated, order preserved)
        seen = set(packages)
        for pkg in common_system_packages:
            if pkg not in seen:
                packages.append(pkg)
                seen.add(pkg)
        return packages

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    async def observe(self, screenshot_path: Path, timeout: float = 30.0) -> Observation:
        if self._use_scrcpy:
            return await self._observe_via_scrcpy(screenshot_path, timeout=timeout)
        return await self._observe_via_screencap(screenshot_path, timeout=timeout)

    async def _observe_via_scrcpy(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        try:
            await self._ensure_scrcpy_started()
        except Exception as exc:
            logger.warning("ADB scrcpy capture unavailable; falling back to screencap: %s", exc)
            return await self._observe_via_screencap(screenshot_path, timeout=timeout)
        if self._frame_source is None:
            raise AdbError("ADB scrcpy capture is enabled but no frame source is configured.")

        snapshot = await self._capture_scrcpy_frame(screenshot_path, timeout=timeout)
        if snapshot is None:
            return await self._observe_via_screencap(screenshot_path, timeout=timeout)
        (input_width, input_height), fg_app, extra = await asyncio.gather(
            self._query_screen_size(timeout),
            self._query_foreground_app(timeout),
            self._collect_ui_tree_extra(timeout, screenshot_path=screenshot_path),
        )
        self._capture_width = snapshot.width
        self._capture_height = snapshot.height
        self._screen_width = input_width
        self._screen_height = input_height
        merged_extra: dict[str, Any] = {
            "capture_source": "scrcpy",
            "frame_timestamp": snapshot.timestamp,
        }
        merged_extra.update(extra)
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=snapshot.width,
            screen_height=snapshot.height,
            foreground_app=fg_app,
            platform=self.platform,
            extra=merged_extra,
        )

    async def _capture_scrcpy_frame(
        self,
        screenshot_path: Path,
        *,
        timeout: float,
        allow_restart: bool = True,
    ) -> ScrcpyFrameSnapshot | None:
        if self._frame_source is None:
            return None

        async def _capture_once() -> ScrcpyFrameSnapshot:
            return await asyncio.to_thread(
                self._frame_source.save_latest,
                screenshot_path,
                timeout_s=min(timeout, self._scrcpy_frame_timeout_ms / 1000.0),
                max_age_s=self._scrcpy_max_frame_age_ms / 1000.0,
            )

        try:
            return await _capture_once()
        except (TimeoutError, RuntimeError, OSError):
            if not allow_restart:
                return None

        try:
            await self._restart_scrcpy_frame_source()
        except Exception:
            return None
        try:
            return await _capture_once()
        except (TimeoutError, RuntimeError, OSError):
            return None

    async def _restart_scrcpy_frame_source(self) -> None:
        if self._frame_source is None:
            raise AdbError("ADB scrcpy capture is enabled but no frame source is configured.")
        await asyncio.to_thread(self._frame_source.stop)
        self._scrcpy_started = False
        await self._ensure_scrcpy_started()

    async def _observe_via_screencap(
        self, screenshot_path: Path, timeout: float = 5.0
    ) -> Observation:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        await self._run("shell", "screencap", "-p", _DEVICE_SCREENSHOT_PATH, timeout=timeout)
        await self._run("pull", _DEVICE_SCREENSHOT_PATH, str(screenshot_path), timeout=timeout)

        screenshot_size = read_png_size(screenshot_path)
        ui_tree_task = self._collect_ui_tree_extra(timeout, screenshot_path=screenshot_path)
        if screenshot_size is None:
            (width, height), fg_app, extra = await asyncio.gather(
                self._query_screen_size(timeout),
                self._query_foreground_app(timeout),
                ui_tree_task,
            )
        else:
            width, height = screenshot_size
            fg_app, extra = await asyncio.gather(
                self._query_foreground_app(timeout),
                ui_tree_task,
            )

        self._screen_width = width
        self._screen_height = height
        self._capture_width = width
        self._capture_height = height

        merged_extra: dict[str, Any] = {"capture_source": "screencap"}
        merged_extra.update(extra)
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=width,
            screen_height=height,
            foreground_app=fg_app,
            platform=self.platform,
            extra=merged_extra,
        )

    async def _collect_ui_tree_extra(
        self,
        timeout: float,
        *,
        screenshot_path: Path | None = None,
    ) -> dict[str, Any]:
        if not self._collect_ui_tree:
            return {}
        # uiautomator dump can be slow on accessibility-heavy app pages.
        ui_timeout = _UI_TREE_DUMP_TIMEOUT_SECONDS
        _clear_raw_ui_tree_artifacts(screenshot_path)
        attempts = [
            ("uiautomator", "dump", "--compressed", _DEVICE_UI_XML_PATH),
            ("uiautomator", "dump", _DEVICE_UI_XML_PATH),
        ]
        last_error: Exception | None = None
        for args in attempts:
            try:
                await self._run("shell", *args, timeout=ui_timeout)
                xml_text = await self._run(
                    "shell",
                    "cat",
                    _DEVICE_UI_XML_PATH,
                    timeout=ui_timeout,
                )
                _write_raw_ui_tree_xml(screenshot_path, xml_text)
                return _parse_ui_tree_xml(
                    xml_text,
                    include_nodes=self._collect_ui_tree_nodes,
                )
            except Exception as exc:
                last_error = exc
                continue
        message = str(last_error or "unknown error")
        logger.warning("ADB UI-tree capture unavailable after retries: %s", message)
        _write_raw_ui_tree_error(screenshot_path, message, ui_timeout)
        return {
            "ui_tree_error": message,
            "ui_tree_timeout_s": ui_timeout,
        }

    async def shutdown(self) -> None:
        if self._frame_source is not None:
            await asyncio.to_thread(self._frame_source.stop)
        self._scrcpy_started = False

    async def _ensure_scrcpy_started(self) -> None:
        if self._scrcpy_started:
            return
        if self._frame_source is None:
            raise AdbError("ADB scrcpy capture is enabled but no frame source is configured.")
        try:
            await asyncio.to_thread(self._frame_source.start)
        except Exception as exc:
            raise AdbError(f"ADB scrcpy frame source unavailable: {exc}") from exc
        self._scrcpy_started = True

    async def _query_screen_size(self, timeout: float) -> tuple[int, int]:
        try:
            output = await self._run("shell", "wm", "size", timeout=timeout)
            match = _WM_SIZE_RE.search(output)
            if match:
                return int(match.group(1)), int(match.group(2))
        except (AdbError, TimeoutError):
            pass
        return 1080, 1920

    async def _query_foreground_app(self, timeout: float) -> str:
        try:
            output = await self._run(
                "shell",
                "dumpsys",
                "activity",
                "activities",
                timeout=max(timeout, 10.0),
            )
            package = self._extract_foreground_app(output)
            if package != "unknown":
                return package
        except (AdbError, TimeoutError):
            pass

        for window_args in (
            ("shell", "dumpsys", "window", "windows"),
            ("shell", "dumpsys", "window"),
        ):
            try:
                output = await self._run(*window_args, timeout=max(timeout, 10.0))
                package = self._extract_foreground_app(output)
                if package != "unknown":
                    return package
            except (AdbError, TimeoutError):
                continue
        return "unknown"

    @staticmethod
    def _extract_foreground_app(output: str) -> str:
        for pattern in (
            _RESUMED_ACTIVITY_RE,
            _TOP_RESUMED_ACTIVITY_RE,
            _CURRENT_FOCUS_RE,
            _FOCUSED_APP_RE,
        ):
            match = pattern.search(output)
            if match:
                return match.group(1)
        return "unknown"

    async def _get_default_input_method(self, timeout: float) -> str | None:
        try:
            output = await self._run(
                "shell",
                "settings",
                "get",
                "secure",
                "default_input_method",
                timeout=timeout,
            )
        except (AdbError, TimeoutError):
            return None
        output = output.strip()
        return output or None

    async def _list_input_methods(self, timeout: float) -> set[str]:
        try:
            output = await self._run("shell", "ime", "list", "-s", timeout=timeout)
        except (AdbError, TimeoutError):
            return set()
        return {line.strip() for line in output.splitlines() if line.strip()}

    async def _needs_ime_enable_before_set(self, timeout: float) -> bool:
        del timeout
        return False

    def _get_packaged_yadb_path(self) -> Path:
        return Path(importlib.resources.files("guiclaw").joinpath("assets/android/yadb"))

    def _write_local_temp_text(self, text: str) -> Path:
        fd, path = tempfile.mkstemp(prefix="guiclaw-yadb-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return Path(path)

    def _make_yadb_device_text_path(self) -> str:
        return f"/data/local/tmp/guiclaw-yadb-{uuid.uuid4().hex}.txt"

    def _write_local_temp_yadb_script(self) -> Path:
        fd, path = tempfile.mkstemp(prefix="guiclaw-yadb-", suffix=".sh")
        script = (
            "#!/system/bin/sh\n"
            'text="$(cat "$1")"\n'
            f"app_process -Djava.class.path={_YADB_PATH} "
            f'/data/local/tmp {_YADB_MAIN_CLASS} -keyboard "$text"\n'
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(script)
        return Path(path)

    def _make_yadb_device_script_path(self) -> str:
        return f"/data/local/tmp/guiclaw-yadb-{uuid.uuid4().hex}.sh"

    async def _ensure_yadb_available(self, timeout: float) -> bool:
        try:
            await self._run("shell", "ls", _YADB_PATH, timeout=timeout)
            return True
        except (AdbError, TimeoutError):
            pass

        local_yadb = self._get_packaged_yadb_path()
        if not local_yadb.exists():
            return False

        try:
            await self._run("push", str(local_yadb), _YADB_PATH, timeout=timeout)
            await self._run("shell", "chmod", "755", _YADB_PATH, timeout=timeout)
        except (AdbError, TimeoutError):
            return False
        return True

    async def _ensure_adb_keyboard_ready(self, timeout: float) -> bool:
        current_ime = await self._get_default_input_method(timeout)
        if current_ime == _ADB_KEYBOARD_IME:
            return True

        available_imes = await self._list_input_methods(timeout)
        if _ADB_KEYBOARD_IME not in available_imes:
            return False

        if await self._needs_ime_enable_before_set(timeout=timeout):
            try:
                await self._run("shell", "ime", "enable", _ADB_KEYBOARD_IME, timeout=timeout)
            except (AdbError, TimeoutError):
                return False

        try:
            await self._run("shell", "ime", "set", _ADB_KEYBOARD_IME, timeout=timeout)
        except (AdbError, TimeoutError):
            return False
        return True

    async def _input_text_via_adb_keyboard(self, text: str, timeout: float) -> bool:
        if not await self._ensure_adb_keyboard_ready(timeout):
            return False
        for args in (
            (
                "shell",
                "am",
                "broadcast",
                "-a",
                "ADB_INPUT_B64",
                "--es",
                "msg",
                _to_b64_text(text),
            ),
            (
                "shell",
                "am",
                "broadcast",
                "-a",
                "ADB_INPUT_TEXT",
                "--es",
                "msg",
                text,
            ),
        ):
            try:
                await self._run(*args, timeout=timeout)
                return True
            except (AdbError, TimeoutError):
                continue
        return False

    async def _input_text_via_yadb(self, text: str, timeout: float) -> bool:
        if not await self._ensure_yadb_available(timeout):
            return False
        local_text_path = self._write_local_temp_text(text)
        device_text_path = self._make_yadb_device_text_path()
        local_script_path = self._write_local_temp_yadb_script()
        device_script_path = self._make_yadb_device_script_path()
        try:
            await self._run("push", str(local_text_path), device_text_path, timeout=timeout)
            await self._run("push", str(local_script_path), device_script_path, timeout=timeout)
            await self._run("shell", "chmod", "755", device_script_path, timeout=timeout)
            await self._run(
                "shell",
                "sh",
                device_script_path,
                device_text_path,
                timeout=timeout,
            )
            return True
        except (AdbError, TimeoutError):
            return False
        finally:
            try:
                local_text_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                local_script_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                await self._run("shell", "rm", "-f", device_text_path, timeout=timeout)
            except (AdbError, TimeoutError):
                pass
            try:
                await self._run("shell", "rm", "-f", device_script_path, timeout=timeout)
            except (AdbError, TimeoutError):
                pass

    async def _input_single_text(self, text: str, timeout: float) -> None:
        if await self._input_text_via_yadb(text, timeout):
            return
        if await self._input_text_via_adb_keyboard(text, timeout):
            return
        if _is_ascii_safe(text):
            await self._run(
                "shell",
                "input",
                "text",
                _escape_shell_text(text),
                timeout=timeout,
            )
            return
        raise AdbError(
            "Unicode text input failed. Install and activate ADBKeyboard "
            "(`adb shell ime set com.android.adbkeyboard/.AdbIME`) or "
            f"push yadb to {_YADB_PATH}."
        )

    async def _execute_input_text(self, text: str, *, auto_enter: bool, timeout: float) -> None:
        if not text:
            return
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized_text.split("\n")
        for index, line in enumerate(lines):
            for segment in _iter_text_input_segments(line):
                await self._input_single_text(segment, timeout)
            if index < len(lines) - 1:
                await self._run(
                    "shell",
                    "input",
                    "keyevent",
                    "KEYCODE_ENTER",
                    timeout=timeout,
                )
        if auto_enter:
            await self._run(
                "shell",
                "input",
                "keyevent",
                "KEYCODE_ENTER",
                timeout=timeout,
            )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        t = action.action_type
        action_result: str | None = None

        if t == "tap":
            x, y = self._resolve_point(action)
            await self._run(
                "shell",
                "input",
                "tap",
                str(x),
                str(y),
                timeout=timeout,
            )

        elif t == "click_multi":
            for px, py in self._resolve_points(action):
                await self._run(
                    "shell",
                    "input",
                    "tap",
                    str(px),
                    str(py),
                    timeout=timeout,
                )
                await asyncio.sleep(0.12)

        elif t == "click_then_type":
            points = self._resolve_points(action)
            if len(points) != 1:
                raise ValueError("Action 'click_then_type' requires exactly one point.")
            px, py = points[0]
            await self._run("shell", "input", "tap", str(px), str(py), timeout=timeout)
            await asyncio.sleep(0.20)
            await self._execute_input_text(action.text or "", auto_enter=action.auto_enter, timeout=timeout)

        elif t == "long_press":
            px, py = self._resolve_point(action)
            x, y = str(px), str(py)
            dur = str(action.duration_ms or 800)
            await self._run("shell", "input", "swipe", x, y, x, y, dur, timeout=timeout)

        elif t == "double_tap":
            px, py = self._resolve_point(action)
            x, y = str(px), str(py)
            await self._run("shell", "input", "tap", x, y, timeout=timeout)
            await asyncio.sleep(0.1)
            await self._run("shell", "input", "tap", x, y, timeout=timeout)

        elif t in ("drag", "swipe"):
            px1, py1 = self._resolve_point(action)
            px2, py2 = self._resolve_second_point(action)
            x1, y1 = str(px1), str(py1)
            x2, y2 = str(px2), str(py2)
            dur = str(action.duration_ms or 300)
            await self._run("shell", "input", "swipe", x1, y1, x2, y2, dur, timeout=timeout)

        elif t == "input_text":
            await self._execute_input_text(
                action.text or "",
                auto_enter=action.auto_enter,
                timeout=timeout,
            )

        elif t == "enter":
            await self._run("shell", "input", "keyevent", "KEYCODE_ENTER", timeout=timeout)

        elif t in ("app_switch", "recents"):
            await self._run("shell", "input", "keyevent", "KEYCODE_APP_SWITCH", timeout=timeout)

        elif t == "hotkey":
            keys = action.key or []
            keycodes: list[str] = []
            for k in keys:
                canonical = canonical_key_name(k)
                keycode = _KEYCODE_MAP.get(canonical)
                if keycode is None:
                    raise ValueError(f"Unknown key {k!r}. Supported: {sorted(_KEYCODE_MAP.keys())}")
                keycodes.append(keycode)

            if len(keycodes) >= 2:
                try:
                    await self._run("shell", "input", "keycombination", *keycodes, timeout=timeout)
                except AdbError as exc:
                    details = f"{exc}\n{exc.stderr}" if exc.stderr else str(exc)
                    lowered = details.lower()
                    if any(marker in lowered for marker in _KEYCOMBINATION_UNSUPPORTED_MARKERS):
                        raise AdbError(
                            "Device does not support simultaneous multi-key hotkeys via "
                            "`adb shell input keycombination`."
                        ) from exc
                    raise
            else:
                await self._run("shell", "input", "keyevent", keycodes[0], timeout=timeout)

        elif t == "scroll":
            await self._do_scroll(action, timeout=timeout)

        elif t == "wait":
            await asyncio.sleep((action.duration_ms or 1000) / 1000.0)

        elif t == "back":
            await self._run("shell", "input", "keyevent", "KEYCODE_BACK", timeout=timeout)

        elif t == "home":
            await self._run("shell", "input", "keyevent", "KEYCODE_HOME", timeout=timeout)

        elif t == "done":
            pass  # terminal action, no device command

        elif t == "open_app":
            pkg = action.text or ""
            if pkg:
                await self._run(
                    "shell",
                    "monkey",
                    "-p",
                    pkg,
                    "-c",
                    "android.intent.category.LAUNCHER",
                    "1",
                    timeout=timeout,
                )

        elif t == "open_deeplink":
            action_result = await self._open_deeplink(action, timeout=timeout)

        elif t == "open_intent":
            action_result = await self._open_intent(action, timeout=timeout)

        elif t == "adb_command":
            action_result = await self._adb_command_runner.execute(action, timeout=timeout)

        elif t == "close_app":
            pkg = action.text or ""
            if pkg:
                await self._run("shell", "am", "force-stop", pkg, timeout=timeout)

        else:
            raise ValueError(f"Unsupported action type: {t!r}")

        description = describe_action(action)
        if action.action_type == "adb_command":
            return action_result or description
        marker = _am_start_launch_variant_marker(action_result or "")
        return f"{description}\n{marker}" if marker else description

    async def _open_deeplink(self, action: Action, *, timeout: float) -> str:
        uri = action.text or ""
        if not uri:
            raise ValueError("open_deeplink requires a URI in action.text")
        if "\x00" in uri:
            raise ValueError("open_deeplink URI must not contain NUL bytes")
        remote_args = [
            "am",
            "start",
            "-W",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            uri,
        ]
        if action.component:
            if not _ANDROID_COMPONENT_RE.match(action.component):
                raise ValueError(
                    f"Invalid Android component for open_deeplink: {action.component!r}"
                )
            remote_args.extend(["-n", action.component])
        if action.package:
            if not _ANDROID_PACKAGE_RE.match(action.package):
                raise ValueError(f"Invalid Android package for open_deeplink: {action.package!r}")
            remote_args.extend(["-p", action.package])
        try:
            return await self._run_am_start_with_fallbacks(
                remote_args,
                timeout=timeout,
                label="open_deeplink",
            )
        except Exception as exc:
            if not self.supports_app_context_launcher():
                raise
            try:
                return await self._run_app_context_launcher(action, timeout=timeout)
            except Exception as app_context_exc:
                raise AdbError(
                    f"adb deeplink launch failed: {exc}; app-context launch failed: {app_context_exc}"
                ) from app_context_exc

    async def _open_intent(self, action: Action, *, timeout: float) -> str:
        intent_action = action.intent_action or ""
        if not intent_action:
            raise ValueError("open_intent requires action.intent_action")
        if not _ANDROID_INTENT_NAME_RE.match(intent_action):
            raise ValueError(f"Invalid Android intent action for open_intent: {intent_action!r}")
        remote_args = ["am", "start", "-W", "-a", intent_action]
        if action.text:
            if "\x00" in action.text:
                raise ValueError("open_intent data URI must not contain NUL bytes")
            remote_args.extend(["-d", action.text])
        if action.mime_type:
            if "\x00" in action.mime_type:
                raise ValueError("open_intent MIME type must not contain NUL bytes")
            remote_args.extend(["-t", action.mime_type])
        for category in action.categories:
            if not _ANDROID_INTENT_NAME_RE.match(category):
                raise ValueError(f"Invalid Android intent category for open_intent: {category!r}")
            remote_args.extend(["-c", category])
        if action.component:
            if not _ANDROID_COMPONENT_RE.match(action.component):
                raise ValueError(f"Invalid Android component for open_intent: {action.component!r}")
            remote_args.extend(["-n", action.component])
        if action.package:
            if not _ANDROID_PACKAGE_RE.match(action.package):
                raise ValueError(f"Invalid Android package for open_intent: {action.package!r}")
            remote_args.extend(["-p", action.package])
        if any(_is_android_uri_extra(key, value) for key, value in action.extras):
            remote_args.append("--grant-read-uri-permission")
        for key, value in action.extras:
            if not _ANDROID_EXTRA_KEY_RE.match(key):
                raise ValueError(f"Invalid Android extra key for open_intent: {key!r}")
            if isinstance(value, bool):
                remote_args.extend(["--ez", key, "true" if value else "false"])
            elif isinstance(value, int) and not isinstance(value, bool):
                remote_args.extend(
                    ["--ei" if -2147483648 <= value <= 2147483647 else "--el", key, str(value)]
                )
            elif isinstance(value, float):
                remote_args.extend(["--ef", key, str(value)])
            else:
                text_value = "" if value is None else str(value)
                if "\x00" in text_value:
                    raise ValueError(f"open_intent extra {key!r} must not contain NUL bytes")
                if _is_android_uri_extra(key, value):
                    remote_args.extend(["--eu", key, text_value])
                else:
                    remote_args.extend(["--es", key, text_value])
        try:
            return await self._run_am_start_with_fallbacks(
                remote_args,
                timeout=timeout,
                label="open_intent",
            )
        except Exception as exc:
            if not self.supports_app_context_launcher():
                raise
            try:
                return await self._run_app_context_launcher(action, timeout=timeout)
            except Exception as app_context_exc:
                raise AdbError(
                    f"adb intent launch failed: {exc}; app-context launch failed: {app_context_exc}"
                ) from app_context_exc

    async def _run_am_start_with_fallbacks(
        self,
        remote_args: list[str],
        *,
        timeout: float,
        label: str,
    ) -> str:
        """Run ``am start`` with conservative fallbacks for fragile launchers."""
        variants: list[tuple[str, list[str]]] = []

        def add_variant(name: str, args: list[str]) -> None:
            key = tuple(args)
            if any(tuple(existing) == key for _, existing in variants):
                return
            variants.append((name, args))

        add_variant("primary", list(remote_args))
        add_variant("no_wait", _without_am_start_wait(remote_args))
        if "-n" in remote_args:
            implicit_args = _without_am_start_option(remote_args, "-n")
            add_variant("implicit", implicit_args)
            add_variant("implicit_no_wait", _without_am_start_wait(implicit_args))
        if "-p" in remote_args and _am_start_data_scheme(remote_args) in {"http", "https"}:
            browser_redirect_args = _without_am_start_option(remote_args, "-p")
            add_variant("browser_redirect", browser_redirect_args)
            add_variant("browser_redirect_no_wait", _without_am_start_wait(browser_redirect_args))

        errors: list[str] = []
        for variant_name, args in variants:
            # `adb shell` still routes through the device shell. Quote each
            # fixed-position argument so URI query separators cannot split the
            # restricted am-start command or swallow later -n/-p arguments.
            remote_cmd = " ".join(shlex.quote(arg) for arg in args)
            try:
                output = await self._run("shell", remote_cmd, timeout=timeout)
            except (AdbError, TimeoutError, asyncio.TimeoutError) as exc:
                errors.append(f"{variant_name}: {type(exc).__name__}: {exc}")
                continue
            if _am_start_output_failed(output):
                errors.append(f"{variant_name}: {output}")
                continue
            if variant_name != "primary":
                logger.info("%s succeeded via %s fallback", label, variant_name)
            return _annotate_am_start_launch_variant(output, variant_name)

        detail = " | ".join(errors) if errors else "no launch variants attempted"
        raise AdbError(f"{label} failed: {detail}")

    def _resolve_x(self, value: float, *, relative: bool) -> int:
        if relative:
            return resolve_coordinate(value, self._screen_width, relative=True)
        return self._resolve_capture_coordinate(
            value,
            capture_extent=self._capture_width,
            input_extent=self._screen_width,
        )

    def _resolve_y(self, value: float, *, relative: bool) -> int:
        if relative:
            return resolve_coordinate(value, self._screen_height, relative=True)
        return self._resolve_capture_coordinate(
            value,
            capture_extent=self._capture_height,
            input_extent=self._screen_height,
        )

    @staticmethod
    def _resolve_capture_coordinate(value: float, *, capture_extent: int, input_extent: int) -> int:
        """Map screenshot-pixel coordinates into the ADB input coordinate space."""
        if capture_extent > 1 and capture_extent != input_extent:
            pixel = round(value / (capture_extent - 1) * (input_extent - 1))
        else:
            pixel = round(value)
        return max(0, min(pixel, input_extent - 1))

    def _resolve_point(self, action: Action) -> tuple[int, int]:
        if action.x is None or action.y is None:
            raise ValueError(f"Action {action.action_type!r} requires coordinates.")
        return (
            self._resolve_x(action.x, relative=action.relative),
            self._resolve_y(action.y, relative=action.relative),
        )

    def _resolve_points(self, action: Action) -> tuple[tuple[int, int], ...]:
        if not action.points:
            raise ValueError(f"Action {action.action_type!r} requires points.")
        return tuple(
            (
                self._resolve_x(x, relative=action.relative),
                self._resolve_y(y, relative=action.relative),
            )
            for x, y in action.points
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
        dur = str(action.duration_ms or 300)

        # For scroll, direction is stored in action.text
        direction = (action.text or "down").lower()

        # Touch scrolls are inverted relative to content direction:
        # "down" means reveal lower content, so the finger moves upward.
        if direction == "up":
            x2, y2 = x, y + pixels
        elif direction == "down":
            x2, y2 = x, y - pixels
        elif direction == "left":
            x2, y2 = x - pixels, y
        else:
            x2, y2 = x + pixels, y

        await self._run(
            "shell",
            "input",
            "swipe",
            str(x),
            str(y),
            str(x2),
            str(y2),
            dur,
            timeout=timeout,
        )


def _parse_ui_tree_xml(
    xml_text: str,
    *,
    max_nodes: int = 160,
    max_values: int = 80,
    include_nodes: bool = False,
) -> dict[str, Any]:
    """Parse Android uiautomator XML into compact observation metadata."""
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError:
        return {}

    visible_text: list[str] = []
    content_desc: list[str] = []
    resource_ids: list[str] = []
    clickable_text: list[str] = []
    focused_text: list[str] = []
    enabled_text: list[str] = []
    class_names: list[str] = []
    ui_tree: list[dict[str, Any]] | None = [] if include_nodes else None
    node_count = 0
    scrollable_present = False
    enabled_present = False

    for element in root.iter("node"):
        node_count += 1
        text = _clean_ui_attr(element.get("text"))
        desc = _clean_ui_attr(element.get("content-desc"))
        resource_id = _clean_ui_attr(element.get("resource-id"))
        class_name = _clean_ui_attr(element.get("class"))
        bounds = _clean_ui_attr(element.get("bounds"))
        clickable = element.get("clickable") == "true"
        focused = element.get("focused") == "true"
        enabled = element.get("enabled") == "true"
        scrollable = element.get("scrollable") == "true"
        if enabled:
            enabled_present = True
        if scrollable:
            scrollable_present = True

        if text:
            visible_text.append(text)
        if desc:
            content_desc.append(desc)
        if resource_id:
            resource_ids.append(resource_id)
        if class_name:
            class_names.append(class_name)

        label = text or desc
        if clickable:
            clickable_text.extend(_iter_ui_label_texts(element))
        if focused and label:
            focused_text.append(label)
        if enabled and label:
            enabled_text.append(label)

        if (
            ui_tree is not None
            and len(ui_tree) < max_nodes
            and (
                text
                or desc
                or resource_id
                or class_name
                or clickable
                or focused
                or enabled
                or scrollable
            )
        ):
            compact_node: dict[str, Any] = {}
            if text:
                compact_node["text"] = text
            if desc:
                compact_node["content_desc"] = desc
            if resource_id:
                compact_node["resource_id"] = resource_id
            if class_name:
                compact_node["class"] = class_name
            if clickable:
                compact_node["clickable"] = True
            if focused:
                compact_node["focused"] = True
            if enabled:
                compact_node["enabled"] = True
            if scrollable:
                compact_node["scrollable"] = True
            if bounds:
                compact_node["bounds"] = bounds
            ui_tree.append(compact_node)

    extra: dict[str, Any] = {"ui_tree_node_count": node_count}
    if visible_text:
        extra["visible_text"] = _dedupe_ui_values(visible_text)[:max_values]
    if content_desc:
        extra["content_desc"] = _dedupe_ui_values(content_desc)[:max_values]
    if resource_ids:
        extra["resource_ids"] = _dedupe_ui_values(resource_ids)[:max_values]
    if clickable_text:
        extra["clickable_text"] = _dedupe_ui_values(clickable_text)[:max_values]
    if focused_text:
        extra["focused_text"] = _dedupe_ui_values(focused_text)[:max_values]
    if enabled_text:
        extra["enabled_text"] = _dedupe_ui_values(enabled_text)[:max_values]
    if class_names:
        extra["class_names"] = _dedupe_ui_values(class_names)[:max_values]
    if enabled_present:
        extra["enabled_present"] = True
    if scrollable_present:
        extra["scrollable_present"] = True
    if ui_tree:
        extra["ui_tree"] = ui_tree
    return extra


def _raw_ui_tree_artifact_paths(screenshot_path: Path | None) -> tuple[Path, Path] | None:
    if screenshot_path is None:
        return None
    screenshot_path = Path(screenshot_path)
    parent = screenshot_path.parent
    run_dir = parent.parent if parent.name == "screenshots" else parent
    raw_xml = run_dir / "ui_tree" / f"{screenshot_path.stem}.xml"
    error_json = run_dir / "ui_tree" / f"{screenshot_path.stem}.error.json"
    return raw_xml, error_json


def _clear_raw_ui_tree_artifacts(screenshot_path: Path | None) -> None:
    paths = _raw_ui_tree_artifact_paths(screenshot_path)
    if paths is None:
        return
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not remove stale UI tree artifact %s: %s", path, exc)


def _write_raw_ui_tree_xml(screenshot_path: Path | None, xml_text: str) -> None:
    paths = _raw_ui_tree_artifact_paths(screenshot_path)
    if paths is None or not xml_text:
        return
    target, _error_path = paths
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(xml_text, encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write raw UI tree XML %s: %s", target, exc)


def _write_raw_ui_tree_error(screenshot_path: Path | None, message: str, timeout_s: float) -> None:
    paths = _raw_ui_tree_artifact_paths(screenshot_path)
    if paths is None:
        return
    _raw_xml, target = paths
    payload = {
        "ui_tree_error": message,
        "ui_tree_timeout_s": timeout_s,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write raw UI tree error %s: %s", target, exc)


def _clean_ui_attr(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _iter_ui_label_texts(element: ET.Element) -> list[str]:
    labels: list[str] = []
    for node in element.iter("node"):
        label = _clean_ui_attr(node.get("text")) or _clean_ui_attr(node.get("content-desc"))
        if label:
            labels.append(label)
    return labels


def _dedupe_ui_values(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_ui_attr(value)
        if not text:
            continue
        marker = text.casefold()
        if marker in seen:
            continue
        out.append(text)
        seen.add(marker)
    return out
