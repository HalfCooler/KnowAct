"""Decorator backend that runs GUI operations on a virtual display.

Wraps any :class:`DeviceBackend` and delegates all calls after configuring
the process environment to target the virtual display.  For Xvfb on Linux
this means setting ``DISPLAY``; for macOS it applies coordinate offsets.

Usage::

    async with BackgroundDesktopBackend(inner, display_manager) as backend:
        await backend.execute(action)

Or with explicit lifecycle management::

    backend = BackgroundDesktopBackend(inner, display_manager)
    await backend.preflight()
    try:
        await backend.execute(action)
    finally:
        await backend.shutdown()
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Any

from guiclaw.backends import background_runtime
from guiclaw.backends.virtual_display import DisplayInfo, VirtualDisplayManager

if TYPE_CHECKING:
    from guiclaw.action import Action
    from guiclaw.interfaces import DeviceBackend
    from guiclaw.observation import Observation

logger = logging.getLogger(__name__)

# Sentinel distinguishes "not yet set" from None (DISPLAY unset on host).
_SENTINEL: object = object()
_NOT_STARTED_MSG = "call preflight() or use async with before observe/execute"


class BackgroundDesktopBackend:
    """Wrapper that runs a DeviceBackend on a virtual display.

    Satisfies the :class:`~guiclaw.interfaces.DeviceBackend` protocol via
    structural conformance (duck typing) — no explicit subclassing.

    Lifecycle::

        preflight()  → display_manager.start() → set DISPLAY → inner.preflight()
        observe()    → assert started → set DISPLAY → inner.observe()
        execute()    → assert started → set DISPLAY → adjust coords → inner.execute()
        shutdown()   → display_manager.stop() → restore DISPLAY → set _stopped flag
    """

    def __init__(
        self,
        inner: DeviceBackend,
        display_manager: VirtualDisplayManager,
        run_metadata: dict[str, str] | None = None,
    ) -> None:
        self._inner = inner
        self._display_manager = display_manager
        self._display_info: DisplayInfo | None = None
        self.runtime_coordinator = background_runtime.GLOBAL_BACKGROUND_RUNTIME_COORDINATOR
        self._run_metadata = {
            "owner": "background-backend",
            "task": "unknown",
            **(run_metadata or {}),
        }
        self._lease_cm: Any | None = None
        # Tracks the DISPLAY value before preflight() so shutdown() can restore it.
        # _SENTINEL means preflight() has not been called yet.
        self._original_display: str | None | object = _SENTINEL
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BackgroundDesktopBackend:
        await self.preflight()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.shutdown()

    # ------------------------------------------------------------------
    # DeviceBackend protocol
    # ------------------------------------------------------------------

    @property
    def platform(self) -> str:
        return self._inner.platform

    async def preflight(self) -> None:
        """Start the virtual display and prepare the inner backend."""
        if self._display_info is not None and not self._stopped:
            return

        self._stopped = False
        self._lease_cm = self.runtime_coordinator.lease(self._run_metadata, logger=logger)
        await self._lease_cm.__aenter__()
        try:
            self._display_info = await self._display_manager.start()
            self._original_display = os.environ.get("DISPLAY")
            self._apply_display_env()
            configure_target_display = self._resolve_configure_target_display()
            if callable(configure_target_display):
                configure_target_display(self._display_info)
            await self._inner.preflight()
        except Exception:
            await self._release_runtime_lease()
            raise

    async def observe(
        self,
        screenshot_path: pathlib.Path,
        timeout: float = 5.0,
    ) -> Observation:
        self._assert_started()
        self._apply_display_env()
        return await self._inner.observe(screenshot_path, timeout)

    async def execute(
        self,
        action: Action,
        timeout: float = 5.0,
    ) -> str:
        self._assert_started()
        self._apply_display_env()
        adjusted = self._apply_offset(action)
        return await self._inner.execute(adjusted, timeout)

    async def list_apps(self) -> list[str]:
        return await self._inner.list_apps()

    def get_intervention_target(self) -> dict[str, Any]:
        info = self._display_info
        if info is None:
            return {}
        return {
            "display_id": info.display_id,
            "monitor_index": info.monitor_index,
            "width": info.width,
            "height": info.height,
            "platform": self.platform,
        }

    async def shutdown(self) -> None:
        """Stop the virtual display and restore the DISPLAY env var.

        Idempotent — subsequent calls log a warning and return immediately.
        Exceptions from ``display_manager.stop()`` are suppressed so cleanup
        always completes even if the display process has already crashed.
        """
        if self._stopped:
            logger.warning("BackgroundDesktopBackend.shutdown() called more than once — ignoring")
            return
        try:
            await self._display_manager.stop()
        except Exception:
            logger.exception("Error stopping display manager during shutdown")
        finally:
            configure_target_display = self._resolve_configure_target_display()
            if callable(configure_target_display):
                configure_target_display(None)
            self._restore_display_env()
            self._stopped = True
            await self._release_runtime_lease()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_started(self) -> None:
        """Raise RuntimeError if preflight() has not been called yet."""
        if self._display_info is None:
            raise RuntimeError(_NOT_STARTED_MSG)

    def _apply_display_env(self) -> None:
        """Set ``DISPLAY`` env var for X11-based virtual displays."""
        if self._display_info and self._display_info.display_id.startswith(":"):
            os.environ["DISPLAY"] = self._display_info.display_id

    def _resolve_configure_target_display(self) -> Any | None:
        if "configure_target_display" in vars(self._inner):
            configure_target_display = vars(self._inner)["configure_target_display"]
            return configure_target_display if callable(configure_target_display) else None
        if inspect.isclass(type(self._inner)) and "configure_target_display" in vars(type(self._inner)):
            configure_target_display = getattr(self._inner, "configure_target_display", None)
            return configure_target_display if callable(configure_target_display) else None
        return None

    def _restore_display_env(self) -> None:
        """Restore ``DISPLAY`` to its pre-preflight value (or remove it)."""
        if self._original_display is _SENTINEL:
            # preflight() was never called — nothing to restore.
            return
        if isinstance(self._original_display, str):
            os.environ["DISPLAY"] = self._original_display
        elif "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]

    def _apply_offset(self, action: Action) -> Action:
        """Translate absolute coordinates by the virtual display offset.

        Xvfb uses offset (0, 0), while managers for positioned surfaces may
        provide a non-zero global offset. Relative actions pass through as-is.
        """
        info = self._display_info
        if not info or (info.offset_x == 0 and info.offset_y == 0):
            return action
        if action.x is None or action.relative:
            return action
        return dataclasses.replace(
            action,
            x=action.x + info.offset_x,
            y=action.y + info.offset_y,
            x2=(action.x2 + info.offset_x) if action.x2 is not None else None,
            y2=(action.y2 + info.offset_y) if action.y2 is not None else None,
        )

    async def _release_runtime_lease(self) -> None:
        if self._lease_cm is None:
            return
        lease_cm = self._lease_cm
        self._lease_cm = None
        await lease_cm.__aexit__(None, None, None)
