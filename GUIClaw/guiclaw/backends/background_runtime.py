"""Shared runtime contracts for background GUI execution."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import shutil
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

logger = logging.getLogger(__name__)

HostPlatform = Literal["linux", "macos", "windows", "unknown"]
RunMode = Literal["isolated", "fallback", "blocked"]

_DEFAULT_REMEDIATION = (
    "Review background runtime availability on this host and retry when isolation support is available."
)
_REMEDIATIONS = {
    "xvfb_missing": "Install Xvfb to enable isolated background execution.",
    "platform_unsupported": "Run without background isolation on this host until a supported isolated backend exists.",
    "windows_non_interactive_session": (
        "Sign in to the interactive Windows session and rerun; Session 0 and service contexts "
        "cannot host isolated desktop automation."
    ),
    "windows_input_desktop_unavailable": (
        "Unlock the interactive desktop or switch back to the default input desktop before "
        "retrying isolated automation."
    ),
    "windows_create_desktop_failed": (
        "Check desktop-creation permissions and close stale GUIClaw desktops before retrying."
    ),
    "windows_app_class_unsupported": (
        "Use classic Win32/GDI apps on the isolated desktop; UWP, DirectX, and GPU-heavy "
        "surfaces must run in the foreground session."
    ),
}


@dataclasses.dataclass(frozen=True)
class IsolationProbeResult:
    supported: bool
    reason_code: str
    retryable: bool
    host_platform: HostPlatform
    backend_name: str | None
    sys_platform: str


@dataclasses.dataclass(frozen=True)
class ResolvedRunMode:
    mode: RunMode
    reason_code: str
    message: str
    requires_acknowledgement: bool = False


def normalize_host_platform(sys_platform: str | None = None) -> HostPlatform:
    value = (sys_platform or sys.platform).lower()
    if value in {"linux", "linux2"}:
        return "linux"
    if value == "darwin":
        return "macos"
    if value in {"win32", "cygwin"}:
        return "windows"
    return "unknown"


def probe_isolated_background_support(
    *,
    sys_platform: str | None = None,
    xvfb_binary: str = "Xvfb",
    target_app_class: str | None = None,
) -> IsolationProbeResult:
    host_platform = normalize_host_platform(sys_platform)
    raw_platform = sys_platform or sys.platform
    if host_platform == "linux":
        binary_path = shutil.which(xvfb_binary)
        if binary_path:
            return IsolationProbeResult(
                supported=True,
                reason_code="xvfb_available",
                retryable=False,
                host_platform=host_platform,
                backend_name="xvfb",
                sys_platform=raw_platform,
            )
        return IsolationProbeResult(
            supported=False,
            reason_code="xvfb_missing",
            retryable=True,
            host_platform=host_platform,
            backend_name="xvfb",
            sys_platform=raw_platform,
        )
    if host_platform == "windows":
        return _probe_windows_isolated_support(
            raw_platform,
            target_app_class=target_app_class,
        )

    return IsolationProbeResult(
        supported=False,
        reason_code="platform_unsupported",
        retryable=False,
        host_platform=host_platform,
        backend_name=None,
        sys_platform=raw_platform,
    )


def resolve_run_mode(
    probe: IsolationProbeResult,
    *,
    require_isolation: bool,
    require_acknowledgement_for_fallback: bool,
) -> ResolvedRunMode:
    if probe.supported:
        return ResolvedRunMode(
            mode="isolated",
            reason_code=probe.reason_code,
            message=(
                f"Background runtime resolved to isolated (reason: {probe.reason_code}). "
                "Isolation is available."
            ),
        )

    mode: RunMode = "blocked" if require_isolation else "fallback"
    remediation = _REMEDIATIONS.get(probe.reason_code, _DEFAULT_REMEDIATION)
    return ResolvedRunMode(
        mode=mode,
        reason_code=probe.reason_code,
        message=f"Background runtime resolved to {mode} (reason: {probe.reason_code}). {remediation}",
        requires_acknowledgement=(mode == "fallback" and require_acknowledgement_for_fallback),
    )


def log_mode_resolution(
    logger: logging.Logger,
    decision: ResolvedRunMode,
    *,
    owner: str,
    task: str,
) -> None:
    log_fn = logger.info if decision.mode == "isolated" else logger.warning
    log_fn(
        "background runtime resolved: owner=%s mode=%s reason=%s requires_ack=%s task=%s message=%s",
        owner,
        decision.mode,
        decision.reason_code,
        decision.requires_acknowledgement,
        task,
        decision.message,
    )


class BackgroundRuntimeCoordinator:
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active_run: dict[str, str] | None = None

    @asynccontextmanager
    async def lease(
        self,
        run_metadata: dict[str, str],
        logger: logging.Logger | None = None,
    ) -> AsyncIterator[None]:
        active_logger = logger or globals()["logger"]
        waiting_owner = run_metadata.get("owner", "unknown")
        waiting_task = run_metadata.get("task", "unknown")
        async with self._condition:
            while self._active_run is not None:
                active_owner = self._active_run.get("owner", "unknown")
                active_task = self._active_run.get("task", "unknown")
                active_logger.warning(
                    "background runtime busy: waiting_owner=%s waiting_task=%s active_owner=%s active_task=%s",
                    waiting_owner,
                    waiting_task,
                    active_owner,
                    active_task,
                )
                await self._condition.wait()
            self._active_run = dict(run_metadata)

        try:
            yield
        finally:
            async with self._condition:
                self._active_run = None
                self._condition.notify_all()


GLOBAL_BACKGROUND_RUNTIME_COORDINATOR = BackgroundRuntimeCoordinator()


def _probe_windows_isolated_support(
    raw_platform: str,
    *,
    target_app_class: str | None = None,
) -> IsolationProbeResult:
    from guiclaw.backends.displays.win32desktop import probe_windows_isolated_desktop_support

    support = probe_windows_isolated_desktop_support(target_app_class=target_app_class)
    return IsolationProbeResult(
        supported=bool(support["supported"]),
        reason_code=str(support["reason_code"]),
        retryable=bool(support["retryable"]),
        host_platform="windows",
        backend_name="windows_isolated_desktop",
        sys_platform=raw_platform,
    )
