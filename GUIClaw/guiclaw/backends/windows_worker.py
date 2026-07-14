"""Windows worker launch helper for alternate-desktop execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any, TextIO

from guiclaw.action import parse_action
from guiclaw.observation import Observation

if TYPE_CHECKING:
    from guiclaw.backends.desktop import LocalDesktopBackend


LocalDesktopBackend = None


def launch_windows_worker(
    *,
    desktop_name: str,
    width: int,
    height: int,
    control_path: str,
) -> subprocess.Popen[Any] | Any:
    """Launch the worker process on the named desktop via ``lpDesktop``."""
    startupinfo = _build_startupinfo(desktop_name)
    command = [
        sys.executable,
        "-m",
        "guiclaw.backends.windows_worker",
        "--desktop-name",
        desktop_name,
        "--width",
        str(width),
        "--height",
        str(height),
        "--control-path",
        control_path,
    ]
    if sys.platform != "win32":
        raise RuntimeError(
            f"launch_windows_worker requires Windows (lpDesktop={startupinfo.lpDesktop})"
        )
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        startupinfo=startupinfo,
    )


def _build_startupinfo(desktop_name: str) -> Any:
    startupinfo_cls = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_cls is None:

        class _StartupInfo:
            lpDesktop: str | None = None  # noqa: N815 - subprocess API field name

        startupinfo = _StartupInfo()
    else:
        startupinfo = startupinfo_cls()
    startupinfo.lpDesktop = f"WinSta0\\{desktop_name}"
    return startupinfo


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUIClaw Windows alternate-desktop worker")
    parser.add_argument("--desktop-name", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--control-path", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run_worker(args, stdin=sys.stdin, stdout=sys.stdout))


async def _run_worker(
    args: argparse.Namespace,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> int:
    _ = args.control_path
    backend_cls = LocalDesktopBackend
    if backend_cls is None:
        from guiclaw.backends.desktop import LocalDesktopBackend as ImportedLocalDesktopBackend

        backend_cls = ImportedLocalDesktopBackend
    backend = backend_cls()
    await backend.preflight()

    while True:
        line = await asyncio.to_thread(stdin.readline)
        if line == "":
            break
        if not line.strip():
            continue

        response = await _handle_command(backend, line)
        await asyncio.to_thread(_write_response, stdout, response)
        if response.get("ok") and response.get("result") == "shutdown":
            break

    return 0


async def _handle_command(backend: LocalDesktopBackend, line: str) -> dict[str, Any]:
    try:
        payload = json.loads(line)
        command = payload.get("command")

        if command == "observe":
            observation = await backend.observe(
                pathlib.Path(str(payload["screenshot_path"])),
                float(payload.get("timeout", 5.0)),
            )
            return {"ok": True, "observation": _serialize_observation(observation)}

        if command == "execute":
            action = parse_action(payload.get("action", {}))
            result = await backend.execute(action, float(payload.get("timeout", 5.0)))
            return {"ok": True, "result": result}

        if command == "list_apps":
            return {"ok": True, "apps": await backend.list_apps()}

        if command == "shutdown":
            return {"ok": True, "result": "shutdown"}

        raise RuntimeError(f"Unknown worker command: {command}")
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _serialize_observation(observation: Observation) -> dict[str, Any]:
    return {
        "screenshot_path": observation.screenshot_path,
        "screen_width": observation.screen_width,
        "screen_height": observation.screen_height,
        "foreground_app": observation.foreground_app,
        "platform": observation.platform,
        "extra": observation.extra,
    }


def _write_response(stdout: TextIO, response: dict[str, Any]) -> None:
    stdout.write(json.dumps(response) + "\n")
    stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
