"""Phase 12 runtime contract coverage."""

from __future__ import annotations

import asyncio
import logging

import pytest

import guiclaw.backends.background_runtime as runtime


def test_probe_result_shape_and_platform_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    assert runtime.normalize_host_platform("linux") == "linux"
    assert runtime.normalize_host_platform("darwin") == "macos"
    assert runtime.normalize_host_platform("win32") == "windows"
    assert runtime.normalize_host_platform("plan9") == "unknown"

    monkeypatch.setattr(runtime.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    supported = runtime.probe_isolated_background_support(sys_platform="linux", xvfb_binary="Xvfb")
    assert supported == runtime.IsolationProbeResult(
        supported=True,
        reason_code="xvfb_available",
        retryable=False,
        host_platform="linux",
        backend_name="xvfb",
        sys_platform="linux",
    )

    monkeypatch.setattr(runtime.shutil, "which", lambda binary: None)
    missing = runtime.probe_isolated_background_support(sys_platform="linux", xvfb_binary="Xvfb")
    assert missing.supported is False
    assert missing.retryable is True
    assert missing.reason_code == "xvfb_missing"
    assert missing.backend_name == "xvfb"

    unsupported = runtime.probe_isolated_background_support(sys_platform="darwin")
    assert unsupported == runtime.IsolationProbeResult(
        supported=False,
        reason_code="platform_unsupported",
        retryable=False,
        host_platform="macos",
        backend_name=None,
        sys_platform="darwin",
    )


def test_resolve_run_mode_variants() -> None:
    fallback_probe = runtime.IsolationProbeResult(
        supported=False,
        reason_code="platform_unsupported",
        retryable=False,
        host_platform="macos",
        backend_name=None,
        sys_platform="darwin",
    )
    fallback = runtime.resolve_run_mode(
        fallback_probe,
        require_isolation=False,
        require_acknowledgement_for_fallback=False,
    )
    assert fallback.mode == "fallback"
    assert fallback.requires_acknowledgement is False
    assert "fallback" in fallback.message
    assert "platform_unsupported" in fallback.message
    assert "supported isolated backend exists" in fallback.message

    blocked = runtime.resolve_run_mode(
        fallback_probe,
        require_isolation=True,
        require_acknowledgement_for_fallback=True,
    )
    assert blocked.mode == "blocked"
    assert blocked.reason_code == fallback_probe.reason_code
    assert blocked.requires_acknowledgement is False
    assert "blocked" in blocked.message
    assert "platform_unsupported" in blocked.message

    isolated_probe = runtime.IsolationProbeResult(
        supported=True,
        reason_code="xvfb_available",
        retryable=False,
        host_platform="linux",
        backend_name="xvfb",
        sys_platform="linux",
    )
    isolated = runtime.resolve_run_mode(
        isolated_probe,
        require_isolation=False,
        require_acknowledgement_for_fallback=True,
    )
    assert isolated.mode == "isolated"
    assert isolated.requires_acknowledgement is False
    assert "Isolation is available." in isolated.message


@pytest.mark.asyncio
async def test_runtime_coordinator_serializes_waiters_and_releases_on_exit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = runtime.BackgroundRuntimeCoordinator()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    events: list[str] = []

    async def first_run() -> None:
        async with coordinator.lease({"owner": "cli", "task": "task-a"}):
            events.append("first-enter")
            first_entered.set()
            await release_first.wait()
        events.append("first-exit")

    async def second_run() -> None:
        await first_entered.wait()
        async with coordinator.lease(
            {"owner": "cli", "task": "task-b"},
            logger=logging.getLogger("guiclaw.backends.background_runtime"),
        ):
            events.append("second-enter")

    with caplog.at_level(logging.WARNING, logger="guiclaw.backends.background_runtime"):
        first_task = asyncio.create_task(first_run())
        second_task = asyncio.create_task(second_run())
        await first_entered.wait()
        await asyncio.sleep(0)
        assert events == ["first-enter"]
        release_first.set()
        await asyncio.gather(first_task, second_task)

    assert events == ["first-enter", "first-exit", "second-enter"]
    busy_lines = [record.message for record in caplog.records if record.message.startswith("background runtime busy:")]
    assert busy_lines
    assert "waiting_owner=cli" in busy_lines[0]
    assert "waiting_task=task-b" in busy_lines[0]
    assert "active_task=task-a" in busy_lines[0]

    async with coordinator.lease({"owner": "cli", "task": "task-c"}):
        assert True
