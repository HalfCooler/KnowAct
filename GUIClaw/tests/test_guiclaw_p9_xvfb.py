"""Phase 9 Plan 02 — XvfbDisplayManager unit tests with mocked subprocess.

All subprocess calls are mocked at the ``asyncio.create_subprocess_exec``
boundary, so no real Xvfb binary is required.  Tests cover VDISP-04:
subprocess launch, socket wait, error types, auto-increment, crash
detection, stop idempotency, and stderr capture.

Requirements covered:
  VDISP-04: XvfbDisplayManager (start/stop, error types, auto-increment)
"""
from __future__ import annotations

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from guiclaw.backends.displays.xvfb import (
    XvfbCrashedError,
    XvfbDisplayManager,
    XvfbNotFoundError,
)
from guiclaw.backends.virtual_display import DisplayInfo, VirtualDisplayManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAST_TIMEOUT = 0.5  # keep tests fast without waiting real seconds


def _make_process(
    pid: int = 12345,
    returncode: int | None = None,
    stderr_content: bytes = b"",
) -> MagicMock:
    """Build a mock ``asyncio.subprocess.Process``.

    ``returncode`` is an ordinary (non-async) attribute on the real class.
    ``stderr.read()`` is an async method on an ``asyncio.StreamReader``.
    ``terminate()`` and ``wait()`` are both synchronous / async respectively.
    """
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode

    # stderr: async read() that returns stderr_content
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr_content)

    # terminate() is synchronous (fire-and-forget signal)
    proc.terminate = MagicMock()

    # wait() is a coroutine — does NOT mutate returncode so that a shared mock
    # can be reused across multiple _try_start() calls without carrying over state.
    async def _wait() -> int:
        return proc.returncode if proc.returncode is not None else 0

    proc.wait = _wait

    return proc


def _patch_subprocess(proc: MagicMock) -> patch:
    """Return a context-manager patch that makes create_subprocess_exec return *proc*."""
    mock = AsyncMock(return_value=proc)
    return patch("asyncio.create_subprocess_exec", mock)


def _patch_path_exists(
    locked_displays: set[int] | None = None,
    socket_ready_for: int | None = None,
) -> patch:
    """Patch ``pathlib.Path.exists`` to simulate lock files and sockets.

    ``locked_displays``: set of display numbers whose lock files exist.
    ``socket_ready_for``: display number whose X11 socket file exists.
    """
    locked_displays = locked_displays or set()

    def _exists(self: object) -> bool:  # type: ignore[override]
        path_str = str(self)
        # Lock file: /tmp/.X{N}-lock
        for num in locked_displays:
            if path_str == f"/tmp/.X{num}-lock":
                return True
        # X11 socket: /tmp/.X11-unix/X{N}
        if socket_ready_for is not None:
            if path_str == f"/tmp/.X11-unix/X{socket_ready_for}":
                return True
        return False

    return patch("pathlib.Path.exists", _exists)


# ---------------------------------------------------------------------------
# VDISP-04: XvfbDisplayManager — isinstance, start, error paths
# ---------------------------------------------------------------------------


async def test_xvfb_isinstance_check() -> None:
    """XvfbDisplayManager satisfies the VirtualDisplayManager protocol."""
    mgr = XvfbDisplayManager()
    assert isinstance(mgr, VirtualDisplayManager)


async def test_xvfb_start_returns_display_info() -> None:
    """start() returns DisplayInfo with correct display_id, width, height."""
    proc = _make_process()
    with _patch_subprocess(proc), _patch_path_exists(socket_ready_for=99):
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=_FAST_TIMEOUT)
        info = await mgr.start()

    assert isinstance(info, DisplayInfo)
    assert info.display_id == ":99"
    assert info.width == 1920
    assert info.height == 1080


async def test_xvfb_start_custom_dimensions() -> None:
    """start() returns DisplayInfo that reflects custom width and height."""
    proc = _make_process()
    with _patch_subprocess(proc), _patch_path_exists(socket_ready_for=99):
        mgr = XvfbDisplayManager(
            display_num=99, width=800, height=600, startup_timeout=_FAST_TIMEOUT
        )
        info = await mgr.start()

    assert info.width == 800
    assert info.height == 600
    assert info.display_id == ":99"


async def test_xvfb_not_found_error() -> None:
    """start() raises XvfbNotFoundError with 'apt install xvfb' when binary is missing."""
    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("No such file or directory: 'Xvfb'"),
    ), _patch_path_exists():
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=_FAST_TIMEOUT)
        with pytest.raises(XvfbNotFoundError) as exc_info:
            await mgr.start()

    assert "apt install xvfb" in str(exc_info.value)


async def test_xvfb_start_timeout() -> None:
    """start() raises TimeoutError when socket never appears and process stays alive."""
    # Process stays alive (returncode=None) but socket never appears.
    proc = _make_process(returncode=None)
    with _patch_subprocess(proc), _patch_path_exists():
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=0.1)
        with pytest.raises(TimeoutError):
            await mgr.start()


async def test_xvfb_auto_increment() -> None:
    """start() skips :99 (lock exists) and launches Xvfb on :100."""
    proc = _make_process()
    mock_exec = AsyncMock(return_value=proc)

    with patch("asyncio.create_subprocess_exec", mock_exec), _patch_path_exists(
        locked_displays={99}, socket_ready_for=100
    ):
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=_FAST_TIMEOUT)
        info = await mgr.start()

    # subprocess must have been called with ":100"
    call_args = mock_exec.call_args
    assert call_args[0][1] == ":100", (
        f"Expected Xvfb to be launched on :100, got {call_args[0][1]}"
    )
    assert info.display_id == ":100"


async def test_xvfb_auto_increment_all_locked() -> None:
    """start() raises RuntimeError when all _MAX_RETRIES display numbers are locked."""
    # Lock displays :99 through :103 (5 slots).
    locked = {99, 100, 101, 102, 103}
    with _patch_path_exists(locked_displays=locked):
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=_FAST_TIMEOUT)
        with pytest.raises(RuntimeError, match="Could not acquire"):
            await mgr.start()


async def test_xvfb_crash_detection() -> None:
    """start() raises XvfbCrashedError containing stderr when process exits early."""
    stderr_msg = b"(EE) Server is already active for display 99"
    # Process is initially alive (returncode=None), but on first poll it will
    # have exited.  We simulate this by making returncode flip to 1 after
    # create_subprocess_exec returns.
    proc = _make_process(returncode=None, stderr_content=stderr_msg)

    call_count = 0
    original_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        proc.returncode = 1  # process dies on first poll iteration
        await original_sleep(0)

    with _patch_subprocess(proc), _patch_path_exists(), patch(
        "asyncio.sleep", _fast_sleep
    ):
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=_FAST_TIMEOUT)
        with pytest.raises(XvfbCrashedError) as exc_info:
            await mgr.start()

    assert "(EE) Server is already active" in str(exc_info.value)


async def test_xvfb_stop_never_started() -> None:
    """Calling stop() on a fresh XvfbDisplayManager does not raise."""
    mgr = XvfbDisplayManager()
    await mgr.stop()  # must not raise


async def test_xvfb_stop_idempotent() -> None:
    """Calling stop() twice after start() does not raise on the second call."""
    proc = _make_process()
    with _patch_subprocess(proc), _patch_path_exists(socket_ready_for=99):
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=_FAST_TIMEOUT)
        await mgr.start()

    await mgr.stop()  # first stop — must succeed
    await mgr.stop()  # second stop — must not raise


async def test_xvfb_stop_terminates_process() -> None:
    """stop() calls terminate() and wait() on the underlying process."""
    proc = _make_process()
    with _patch_subprocess(proc), _patch_path_exists(socket_ready_for=99):
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=_FAST_TIMEOUT)
        await mgr.start()
        await mgr.stop()

    proc.terminate.assert_called_once()
    # _process is cleared after stop
    assert mgr._process is None


async def test_xvfb_stderr_is_piped() -> None:
    """Xvfb subprocess is created with stderr=asyncio.subprocess.PIPE, never DEVNULL."""
    proc = _make_process()
    captured_kwargs: dict = {}

    async def _mock_exec(*args: object, **kwargs: object) -> MagicMock:
        captured_kwargs.update(kwargs)
        return proc

    with patch("asyncio.create_subprocess_exec", _mock_exec), _patch_path_exists(
        socket_ready_for=99
    ):
        mgr = XvfbDisplayManager(display_num=99, startup_timeout=_FAST_TIMEOUT)
        await mgr.start()

    assert captured_kwargs.get("stderr") == asyncio.subprocess.PIPE, (
        "stderr must be asyncio.subprocess.PIPE, not DEVNULL or None"
    )
