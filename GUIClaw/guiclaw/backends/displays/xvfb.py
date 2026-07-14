"""Xvfb-based virtual display manager for Linux.

Launches ``Xvfb`` as a subprocess and waits for the X11 socket to appear
before returning :class:`DisplayInfo`.  No Python dependencies beyond the
standard library — ``Xvfb`` must be installed on the host.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib

from guiclaw.backends.virtual_display import DisplayInfo

logger = logging.getLogger(__name__)

_SOCKET_POLL_INTERVAL = 0.2  # seconds between socket existence checks
_DEFAULT_STARTUP_TIMEOUT = 5.0  # seconds to wait for Xvfb socket
_MAX_RETRIES = 5  # maximum display number increments before giving up

__all__ = ["XvfbDisplayManager", "XvfbNotFoundError", "XvfbCrashedError"]


class XvfbNotFoundError(RuntimeError):
    """Raised when the Xvfb binary is not installed on the system."""


class XvfbCrashedError(RuntimeError):
    """Raised when the Xvfb process exits unexpectedly before the X11 socket appears."""


class XvfbDisplayManager:
    """Manage an Xvfb process for headless GUI execution on Linux.

    Auto-increments display number when a lock file is detected (up to
    ``_MAX_RETRIES`` attempts).  Captures stderr via PIPE for crash
    diagnostics — never silently discards subprocess output.

    Example::

        mgr = XvfbDisplayManager(display_num=99)
        info = await mgr.start()   # blocks until X11 socket is ready
        # ... use info.display_id as DISPLAY env var ...
        await mgr.stop()
    """

    def __init__(
        self,
        display_num: int = 99,
        width: int = 1920,
        height: int = 1080,
        depth: int = 24,
        startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT,
    ) -> None:
        self._display_num = display_num
        self._width = width
        self._height = height
        self._depth = depth
        self._startup_timeout = startup_timeout
        self._process: asyncio.subprocess.Process | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> DisplayInfo:
        """Launch Xvfb and wait for the X11 socket to become available.

        Tries ``_MAX_RETRIES`` display numbers starting from ``display_num``,
        skipping any slot whose lock file already exists.  If a free slot is
        found, delegates to ``_try_start()`` and propagates any errors directly
        (``XvfbNotFoundError``, ``XvfbCrashedError``, ``TimeoutError``).

        Raises:
            XvfbNotFoundError: If the ``Xvfb`` binary is not installed.
            XvfbCrashedError: If the Xvfb process exits before the socket appears.
            TimeoutError: If the socket does not appear within ``startup_timeout`` seconds.
            RuntimeError: If all ``_MAX_RETRIES`` display numbers have lock files.
        """
        for attempt in range(_MAX_RETRIES):
            num = self._display_num + attempt
            lock_path = pathlib.Path(f"/tmp/.X{num}-lock")

            if lock_path.exists():
                logger.debug(
                    "Skipping display :%d -- lock file exists (may be stale)", num
                )
                continue

            # Found a free slot — attempt launch; propagate all errors immediately.
            return await self._try_start(num)

        raise RuntimeError(
            f"Could not acquire a free display after {_MAX_RETRIES} attempts"
        )

    async def stop(self) -> None:
        """Terminate the Xvfb process.  Idempotent and safe on never-started managers."""
        if self._process is None:
            return
        try:
            self._process.terminate()
            await self._process.wait()
            logger.info("Stopped Xvfb (pid=%s)", self._process.pid)

            # Drain stderr so the pipe buffer is flushed; log any content for diagnostics.
            if self._process.stderr is not None:
                stderr_bytes = await self._process.stderr.read()
                if stderr_bytes:
                    logger.debug(
                        "Xvfb stderr on shutdown: %s",
                        stderr_bytes.decode(errors="replace").strip(),
                    )
        except ProcessLookupError:
            pass  # process already exited before terminate() reached it
        finally:
            self._process = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _try_start(self, display_num: int) -> DisplayInfo:
        """Attempt to start Xvfb on the given display number.

        Wraps subprocess launch, socket polling, and crash detection for a
        single display slot.  Callers (``start()``) handle retry logic.

        Raises:
            XvfbNotFoundError: If ``Xvfb`` binary is not found.
            XvfbCrashedError: If Xvfb exits before the socket appears.
            TimeoutError: If the socket does not appear within ``startup_timeout`` seconds.
        """
        display_id = f":{display_num}"
        screen_spec = f"{self._width}x{self._height}x{self._depth}"

        try:
            self._process = await asyncio.create_subprocess_exec(
                "Xvfb",
                display_id,
                "-screen",
                "0",
                screen_spec,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise XvfbNotFoundError(
                "Xvfb is not installed. Install it with: apt install xvfb"
            ) from exc

        logger.info("Started Xvfb on %s (pid=%s)", display_id, self._process.pid)

        socket_path = pathlib.Path(f"/tmp/.X11-unix/X{display_num}")

        try:
            await asyncio.wait_for(
                self._poll_socket(socket_path, display_num),
                timeout=self._startup_timeout,
            )
        except asyncio.TimeoutError:
            await self.stop()
            raise TimeoutError(
                f"Xvfb failed to create socket {socket_path} within "
                f"{self._startup_timeout}s"
            )

        return DisplayInfo(
            display_id=display_id,
            width=self._width,
            height=self._height,
        )

    async def _poll_socket(self, socket_path: pathlib.Path, display_num: int) -> None:
        """Poll until the X11 socket file appears or Xvfb crashes.

        Raises:
            XvfbCrashedError: If the process exits before the socket appears.
        """
        while True:
            if socket_path.exists():
                logger.debug("Xvfb socket ready: %s", socket_path)
                return

            if self._process is not None and self._process.returncode is not None:
                returncode = self._process.returncode
                # Process has exited — safe to drain stderr without blocking.
                stderr_bytes = await self._process.stderr.read()
                stderr_text = stderr_bytes.decode(errors="replace").strip()
                self._process = None
                raise XvfbCrashedError(
                    f"Xvfb exited with code {returncode} before socket appeared on "
                    f":{display_num}. Stderr: {stderr_text}"
                )

            await asyncio.sleep(_SOCKET_POLL_INTERVAL)
