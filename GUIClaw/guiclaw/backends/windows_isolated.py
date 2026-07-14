"""Windows-specific isolated backend with explicit worker lifecycle ownership."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

from guiclaw.backends import background_runtime
from guiclaw.backends.displays.win32desktop import Win32DesktopManager
from guiclaw.backends.virtual_display import DisplayInfo
from guiclaw.backends.windows_worker import launch_windows_worker
from guiclaw.observation import Observation

if TYPE_CHECKING:
    from guiclaw.action import Action
    from guiclaw.interfaces import DeviceBackend

logger = logging.getLogger(__name__)

_BACKEND_NAME = "windows_isolated_desktop"
_NOT_STARTED_MSG = "call preflight() or use async with before observe/execute"


class WindowsIsolatedBackend:
    """Own the Windows alternate-desktop worker lifecycle for one run."""

    def __init__(
        self,
        inner: DeviceBackend,
        display_manager: Win32DesktopManager,
        run_metadata: dict[str, str] | None = None,
    ) -> None:
        self._inner = inner
        self._display_manager = display_manager
        self._display_info: DisplayInfo | None = None
        self._worker_process: Any | None = None
        self._worker_control_path: str | None = None
        self._target_display_configured = False
        self._stopped = False
        self._lease_cm: Any | None = None
        self.runtime_coordinator = background_runtime.GLOBAL_BACKGROUND_RUNTIME_COORDINATOR
        self._run_metadata = {
            "owner": "windows-isolated-backend",
            "task": "unknown",
            **(run_metadata or {}),
        }

    async def __aenter__(self) -> WindowsIsolatedBackend:
        await self.preflight()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.shutdown()

    @property
    def platform(self) -> str:
        return self._inner.platform

    async def preflight(self) -> None:
        if self._display_info is not None and not self._stopped:
            return

        self._stopped = False
        self._lease_cm = self.runtime_coordinator.lease(self._run_metadata, logger=logger)
        await self._lease_cm.__aenter__()
        try:
            self._display_info = await self._display_manager.start()
            configure_target_display = self._resolve_configure_target_display()
            if callable(configure_target_display):
                configure_target_display(self._display_info)
                self._target_display_configured = True
            await self._start_worker_session(self._display_info)
            logger.info(
                "windows isolated backend ready: owner=%s task=%s backend_name=%s display_id=%s desktop_name=%s",
                self._run_metadata.get("owner", "unknown"),
                self._run_metadata.get("task", "unknown"),
                _BACKEND_NAME,
                self._display_info.display_id,
                self._display_manager.desktop_name,
            )
        except asyncio.CancelledError:
            if self._display_info is not None:
                await self.shutdown("cancelled")
            else:
                await self._release_runtime_lease()
            raise
        except Exception:
            if self._display_info is not None:
                await self.shutdown("startup_failed")
            else:
                await self._release_runtime_lease()
            raise

    async def observe(
        self,
        screenshot_path: pathlib.Path,
        timeout: float = 5.0,
    ) -> Observation:
        self._assert_started()
        try:
            response = await self._send_worker_command(
                {
                    "command": "observe",
                    "screenshot_path": os.fspath(screenshot_path),
                    "timeout": timeout,
                }
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "worker observe failed")))
            return self._deserialize_observation(response.get("observation"))
        except asyncio.CancelledError:
            await self.shutdown("cancelled")
            raise

    async def execute(
        self,
        action: Action,
        timeout: float = 5.0,
    ) -> str:
        self._assert_started()
        try:
            response = await self._send_worker_command(
                {
                    "command": "execute",
                    "action": _serialize_action_for_windows_worker(action),
                    "timeout": timeout,
                }
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "worker execute failed")))
            return str(response.get("result", ""))
        except asyncio.CancelledError:
            await self.shutdown("cancelled")
            raise

    async def list_apps(self) -> list[str]:
        self._assert_started()
        try:
            response = await self._send_worker_command({"command": "list_apps"})
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "worker list-apps failed")))
            return self._deserialize_list_apps(response.get("apps"))
        except asyncio.CancelledError:
            await self.shutdown("cancelled")
            raise

    def get_intervention_target(self) -> dict[str, Any]:
        info = self._display_info
        if info is None:
            return {}
        return {
            "display_id": info.display_id,
            "desktop_name": self._display_manager.desktop_name,
            "width": info.width,
            "height": info.height,
            "platform": self.platform,
        }

    async def shutdown(self, cleanup_reason: str = "normal") -> None:
        if self._stopped:
            return

        self._stopped = True
        display_id = self._display_info.display_id if self._display_info is not None else "unknown"
        try:
            await self._shutdown_worker_session()
        except Exception:
            logger.exception(
                "windows isolated backend worker cleanup failed: cleanup_reason=%s backend_name=%s",
                cleanup_reason,
                _BACKEND_NAME,
            )
        try:
            await self._stop_display_manager()
        finally:
            self._clear_target_display()
            logger.info(
                "windows isolated backend cleanup: cleanup_reason=%s display_id=%s backend_name=%s",
                cleanup_reason,
                display_id,
                _BACKEND_NAME,
            )
            self._display_info = None
            await self._release_runtime_lease()

    def _assert_started(self) -> None:
        if self._display_info is None:
            raise RuntimeError(_NOT_STARTED_MSG)

    async def _start_worker_session(self, display_info: DisplayInfo) -> None:
        if self._worker_process is not None:
            return

        fd, control_path = tempfile.mkstemp(prefix="guiclaw-windows-worker-", suffix=".json")
        os.close(fd)
        self._worker_control_path = control_path
        process = launch_windows_worker(
            desktop_name=self._display_manager.desktop_name,
            width=display_info.width,
            height=display_info.height,
            control_path=control_path,
        )
        if getattr(process, "stdin", None) is None or getattr(process, "stdout", None) is None:
            self._cleanup_control_path()
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
            raise RuntimeError("Windows worker process did not expose stdin/stdout pipes")
        self._worker_process = process
        logger.info(
            "windows isolated backend worker launched: backend_name=%s display_id=%s desktop_name=%s "
            "lpDesktop=%s control_path=%s",
            _BACKEND_NAME,
            display_info.display_id,
            self._display_manager.desktop_name,
            f"WinSta0\\{self._display_manager.desktop_name}",
            control_path,
        )

    async def _shutdown_worker_session(self) -> None:
        process = self._worker_process
        if process is not None:
            try:
                await self._send_worker_command({"command": "shutdown"})
            except Exception:
                logger.exception("windows isolated backend worker shutdown command failed")
            await self._stop_worker_process(timeout=1.0)
            self._close_process_pipes(process)
        self._worker_process = None
        self._cleanup_control_path()

    def _resolve_configure_target_display(self) -> Any | None:
        if "configure_target_display" in vars(self._inner):
            configure_target_display = vars(self._inner)["configure_target_display"]
            return configure_target_display if callable(configure_target_display) else None
        if inspect.isclass(type(self._inner)) and "configure_target_display" in vars(type(self._inner)):
            configure_target_display = getattr(self._inner, "configure_target_display", None)
            return configure_target_display if callable(configure_target_display) else None
        return None

    def _clear_target_display(self) -> None:
        configure_target_display = self._resolve_configure_target_display()
        if callable(configure_target_display) and self._target_display_configured:
            configure_target_display(None)
        self._target_display_configured = False

    async def _stop_display_manager(self) -> None:
        try:
            await self._display_manager.stop()
        except Exception:
            logger.exception("windows isolated backend display cleanup failed")

    async def _release_runtime_lease(self) -> None:
        if self._lease_cm is None:
            return
        lease_cm = self._lease_cm
        self._lease_cm = None
        await lease_cm.__aexit__(None, None, None)

    async def _send_worker_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._require_worker_process()
        stdin = getattr(process, "stdin", None)
        if stdin is None:
            raise RuntimeError("Windows worker stdin is unavailable")
        message = json.dumps(payload) + "\n"
        await asyncio.to_thread(self._write_worker_command, stdin, message)
        return await self._read_worker_response()

    async def _read_worker_response(self) -> dict[str, Any]:
        process = self._require_worker_process()
        stdout = getattr(process, "stdout", None)
        if stdout is None:
            raise RuntimeError("Windows worker stdout is unavailable")
        line = await asyncio.to_thread(stdout.readline)
        if line == "":
            raise RuntimeError("Windows worker closed its response channel")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid Windows worker response: {exc}") from exc
        if not isinstance(response, dict):
            raise RuntimeError("Windows worker response must be a JSON object")
        return response

    def _deserialize_observation(self, payload: Any) -> Observation:
        if not isinstance(payload, dict):
            raise RuntimeError("Windows worker observation payload must be an object")
        extra = payload.get("extra") or {}
        if not isinstance(extra, dict):
            raise RuntimeError("Windows worker observation extra payload must be an object")
        return Observation(
            screenshot_path=payload.get("screenshot_path"),
            screen_width=int(payload["screen_width"]),
            screen_height=int(payload["screen_height"]),
            foreground_app=payload.get("foreground_app"),
            platform=str(payload.get("platform", "unknown")),
            extra=extra,
        )

    def _deserialize_list_apps(self, payload: Any) -> list[str]:
        if not isinstance(payload, list):
            raise RuntimeError("Windows worker app list payload must be an array")
        return [str(item) for item in payload]

    async def _stop_worker_process(self, timeout: float) -> None:
        process = self._worker_process
        if process is None:
            return
        try:
            await asyncio.wait_for(self._wait_for_process(process), timeout=timeout)
            return
        except asyncio.TimeoutError:
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
        try:
            await asyncio.wait_for(self._wait_for_process(process), timeout=timeout)
            return
        except asyncio.TimeoutError:
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
            await self._wait_for_process(process)

    async def _wait_for_process(self, process: Any) -> Any:
        wait = getattr(process, "wait", None)
        if not callable(wait):
            return None
        try:
            return await asyncio.to_thread(wait)
        except TypeError:
            result = wait()
            if inspect.isawaitable(result):
                return await result
            return result

    def _close_process_pipes(self, process: Any) -> None:
        for name in ("stdin", "stdout", "stderr"):
            pipe = getattr(process, name, None)
            close = getattr(pipe, "close", None)
            if callable(close):
                close()

    def _cleanup_control_path(self) -> None:
        control_path = self._worker_control_path
        self._worker_control_path = None
        if control_path and os.path.exists(control_path):
            os.unlink(control_path)

    def _require_worker_process(self) -> Any:
        if self._worker_process is None:
            raise RuntimeError("Windows worker process has not been started")
        return self._worker_process

    @staticmethod
    def _write_worker_command(stdin: Any, message: str) -> None:
        stdin.write(message)
        flush = getattr(stdin, "flush", None)
        if callable(flush):
            flush()


def _serialize_action_for_windows_worker(action: Action) -> dict[str, Any]:
    payload = dataclasses.asdict(action)
    normalized: dict[str, Any] = {
        key: payload[key]
        for key in (
            "action_type",
            "x",
            "y",
            "x2",
            "y2",
            "text",
            "key",
            "pixels",
            "duration_ms",
            "relative",
            "status",
            "auto_enter",
            "component",
            "package",
            "intent_action",
            "mime_type",
            "categories",
            "extras",
        )
        if key in payload
    }
    if normalized.get("component") is None:
        normalized.pop("component", None)
    if normalized.get("package") is None:
        normalized.pop("package", None)
    if normalized.get("intent_action") is None:
        normalized.pop("intent_action", None)
    if normalized.get("mime_type") is None:
        normalized.pop("mime_type", None)
    if not normalized.get("categories"):
        normalized.pop("categories", None)
    if not normalized.get("extras"):
        normalized.pop("extras", None)
    return normalized
