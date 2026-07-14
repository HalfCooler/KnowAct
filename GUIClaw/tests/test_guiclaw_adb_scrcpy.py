from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import guiclaw.cli as cli
from nanobot.config.schema import GuiConfig
from guiclaw.action import Action
from guiclaw.backends import adb as adb_backend_module
from guiclaw.backends.adb import (
    AdbBackend,
    AdbError,
    ScrcpyFrameSnapshot,
    ScrcpyFrameSource,
    _parse_ui_tree_xml,
    _pil_image_from_frame,
)
from guiclaw.observation import Observation
from guiclaw.skills.state_contract import evaluate_state_contract


class FakeFrameSource:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        Image.new("RGB", (320, 640), color=(10, 20, 30)).save(path)
        return ScrcpyFrameSnapshot(width=320, height=640, timestamp=123.0)


class StaleThenFreshFrameSource:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.save_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        self.save_calls += 1
        if self.save_calls == 1:
            raise TimeoutError("latest scrcpy frame is stale (8.00s > 5.00s)")
        Image.new("RGB", (320, 640), color=(40, 50, 60)).save(path)
        return ScrcpyFrameSnapshot(width=320, height=640, timestamp=time.time())


class AlwaysStaleFrameSource:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.save_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        self.save_calls += 1
        raise TimeoutError("latest scrcpy frame is stale (8.00s > 5.00s)")


class ErrorThenFreshFrameSource:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.start_calls = 0
        self.stop_calls = 0
        self.save_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        self.save_calls += 1
        if self.save_calls == 1:
            raise self.exc
        Image.new("RGB", (320, 640), color=(70, 80, 90)).save(path)
        return ScrcpyFrameSnapshot(width=320, height=640, timestamp=time.time())


class AlwaysErrorFrameSource:
    def __init__(self, exc_factory: type[Exception]) -> None:
        self.exc_factory = exc_factory
        self.start_calls = 0
        self.stop_calls = 0
        self.save_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        self.save_calls += 1
        raise self.exc_factory("scrcpy reader failed")


@pytest.mark.asyncio
async def test_adb_observe_uses_scrcpy_frame_and_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_source = FakeFrameSource()
    backend = AdbBackend(frame_source=frame_source)
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))

    screenshot_path = tmp_path / "screen.png"
    observation = await backend.observe(screenshot_path)

    assert frame_source.started is True
    assert screenshot_path.exists()
    assert observation.screen_width == 320
    assert observation.screen_height == 640
    assert observation.foreground_app == "com.example.app"
    assert observation.extra["capture_source"] == "scrcpy"
    assert backend._capture_width == 320
    assert backend._capture_height == 640
    assert backend._screen_width == 1080
    assert backend._screen_height == 2376


def test_scrcpy_save_latest_waits_for_fresh_frame_after_stale_cache(tmp_path: Path) -> None:
    source = ScrcpyFrameSource()
    stale = Image.new("RGB", (16, 16), color=(255, 0, 0))
    fresh = Image.new("RGB", (16, 16), color=(0, 255, 0))

    with source._condition:
        source._latest_frame = stale
        source._latest_ts = time.time() - 10.0

    def publish_fresh_frame() -> None:
        time.sleep(0.05)
        source._set_latest_frame(fresh)

    thread = threading.Thread(target=publish_fresh_frame)
    thread.start()
    screenshot_path = tmp_path / "screen.png"
    snapshot = source.save_latest(screenshot_path, timeout_s=1.0, max_age_s=0.5)
    thread.join(timeout=1.0)

    assert snapshot.width == 16
    assert snapshot.height == 16
    with Image.open(screenshot_path) as image:
        assert image.getpixel((0, 0)) == (0, 255, 0)


@pytest.mark.asyncio
async def test_adb_observe_restarts_scrcpy_after_stale_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_source = StaleThenFreshFrameSource()
    backend = AdbBackend(frame_source=frame_source)
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation.extra["capture_source"] == "scrcpy"
    assert frame_source.start_calls == 2
    assert frame_source.stop_calls == 1
    assert frame_source.save_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [RuntimeError("scrcpy reader failed"), OSError("Bad file descriptor")])
async def test_adb_observe_restarts_scrcpy_after_scrcpy_reader_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exc: Exception,
) -> None:
    frame_source = ErrorThenFreshFrameSource(exc)
    backend = AdbBackend(frame_source=frame_source)
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation.extra["capture_source"] == "scrcpy"
    assert frame_source.start_calls == 2
    assert frame_source.stop_calls == 1
    assert frame_source.save_calls == 2


@pytest.mark.asyncio
async def test_adb_observe_falls_back_to_screencap_after_scrcpy_restart_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_source = AlwaysStaleFrameSource()
    backend = AdbBackend(frame_source=frame_source)
    fallback_observation = Observation(
        screenshot_path=str(tmp_path / "fallback.png"),
        screen_width=1080,
        screen_height=2376,
        foreground_app="com.example.app",
        platform="android",
    )
    fallback = AsyncMock(return_value=fallback_observation)
    monkeypatch.setattr(backend, "_observe_via_screencap", fallback)

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation is fallback_observation
    assert frame_source.start_calls == 2
    assert frame_source.stop_calls == 1
    assert frame_source.save_calls == 2
    fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_adb_observe_falls_back_to_screencap_when_scrcpy_start_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend(frame_source=FakeFrameSource())
    fallback_observation = Observation(
        screenshot_path=str(tmp_path / "fallback.png"),
        screen_width=1080,
        screen_height=2376,
        foreground_app="com.example.app",
        platform="android",
    )
    fallback = AsyncMock(return_value=fallback_observation)
    monkeypatch.setattr(backend, "_ensure_scrcpy_started", AsyncMock(side_effect=AdbError("broken scrcpy")))
    monkeypatch.setattr(backend, "_observe_via_screencap", fallback)

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation is fallback_observation
    fallback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_factory", [RuntimeError, OSError])
async def test_adb_observe_falls_back_to_screencap_after_scrcpy_reader_error_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exc_factory: type[Exception],
) -> None:
    frame_source = AlwaysErrorFrameSource(exc_factory)
    backend = AdbBackend(frame_source=frame_source)
    fallback_observation = Observation(
        screenshot_path=str(tmp_path / "fallback.png"),
        screen_width=1080,
        screen_height=2376,
        foreground_app="com.example.app",
        platform="android",
    )
    fallback = AsyncMock(return_value=fallback_observation)
    monkeypatch.setattr(backend, "_observe_via_screencap", fallback)

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation is fallback_observation
    assert frame_source.start_calls == 2
    assert frame_source.stop_calls == 1
    assert frame_source.save_calls == 2
    fallback.assert_awaited_once()


def test_scrcpy_frame_source_records_listener_error_and_save_latest_fails_fast(tmp_path: Path) -> None:
    class BrokenClient:
        def frames(self):
            raise OSError("Bad file descriptor")

    source = ScrcpyFrameSource()
    source._client = BrokenClient()

    source._listen_forever()

    with pytest.raises(RuntimeError, match="scrcpy frame reader failed") as exc_info:
        source.save_latest(tmp_path / "screen.png", timeout_s=1.0, max_age_s=0.5)
    assert isinstance(exc_info.value.__cause__, OSError)


def test_scrcpy_frame_source_uses_owned_listener_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeClient:
        instances: list["FakeClient"] = []

        def __init__(self, config: FakeConfig) -> None:
            self.config = config
            self.start_frame_listener_called = False
            self.stopped = False
            FakeClient.instances.append(self)

        def start(self) -> None:
            pass

        def wait_until_ready(self, timeout: float = 5) -> Image.Image:
            del timeout
            return Image.new("RGB", (16, 16), color=(1, 2, 3))

        def start_frame_listener(self, callback: object) -> threading.Thread:
            del callback
            self.start_frame_listener_called = True
            worker = threading.Thread(target=lambda: None)
            worker.start()
            return worker

        def frames(self):
            raise OSError("Bad file descriptor")
            yield  # pragma: no cover

        def stop(self) -> None:
            self.stopped = True

    fake_module = types.SimpleNamespace(
        ScrcpyClient=FakeClient,
        ScrcpyConfig=FakeConfig,
    )
    monkeypatch.setitem(sys.modules, "py_scrcpy_sdk", fake_module)

    source = ScrcpyFrameSource(frame_timeout_ms=100)
    source.start()
    client = FakeClient.instances[0]

    deadline = time.time() + 1.0
    while time.time() < deadline:
        with source._condition:
            if source._reader_error is not None:
                break
        time.sleep(0.01)

    assert client.start_frame_listener_called is False
    with pytest.raises(RuntimeError, match="scrcpy frame reader failed") as exc_info:
        source.save_latest(tmp_path / "screen.png", timeout_s=1.0, max_age_s=0.5)
    assert isinstance(exc_info.value.__cause__, OSError)
    source.stop()


@pytest.mark.asyncio
async def test_adb_observe_can_attach_compact_ui_tree_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
    <hierarchy rotation="0">
      <node index="0" text="青少年模式" resource-id="tv.danmaku.bili:id/minor_mode"
            class="android.widget.TextView" package="tv.danmaku.bili"
            content-desc="" clickable="true" focused="false" scrollable="false"
            bounds="[0,0][100,50]" />
      <node index="1" text="" resource-id="tv.danmaku.bili:id/search"
            class="android.widget.EditText" package="tv.danmaku.bili"
            content-desc="搜索" clickable="true" focused="true" scrollable="true"
            bounds="[0,60][100,110]" />
    </hierarchy>"""
    backend = AdbBackend(use_scrcpy=False, collect_ui_tree=True)
    run_mock = AsyncMock(side_effect=["", "", "UI hierarchy dumped", xml])
    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="tv.danmaku.bili"))
    monkeypatch.setattr(adb_backend_module, "read_png_size", lambda _path: (320, 640))

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation.extra["capture_source"] == "screencap"
    assert observation.extra["visible_text"] == ["青少年模式"]
    assert observation.extra["content_desc"] == ["搜索"]
    assert observation.extra["clickable_text"] == ["青少年模式", "搜索"]
    assert observation.extra["focused_text"] == ["搜索"]
    assert observation.extra["scrollable_present"] is True
    assert observation.extra["resource_ids"] == [
        "tv.danmaku.bili:id/minor_mode",
        "tv.danmaku.bili:id/search",
    ]
    assert observation.extra["ui_tree_node_count"] == 2
    assert "ui_tree" not in observation.extra


@pytest.mark.asyncio
async def test_adb_observe_writes_raw_ui_tree_xml_sibling_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
    <hierarchy rotation="0">
      <node index="0" text="搜索" resource-id="tv.danmaku.bili:id/search_src_text"
            class="android.widget.EditText" package="tv.danmaku.bili"
            content-desc="搜索查询" clickable="true" focused="true" scrollable="false"
            bounds="[0,0][100,50]" />
    </hierarchy>"""
    backend = AdbBackend(use_scrcpy=False, collect_ui_tree=True, collect_ui_tree_nodes=True)
    run_mock = AsyncMock(side_effect=["", "", "UI hierarchy dumped", xml])
    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="tv.danmaku.bili"))
    monkeypatch.setattr(adb_backend_module, "read_png_size", lambda _path: (320, 640))

    screenshot = tmp_path / "run" / "screenshots" / "step_000.png"
    observation = await backend.observe(screenshot)

    raw_xml_path = tmp_path / "run" / "ui_tree" / "step_000.xml"
    assert raw_xml_path.read_text(encoding="utf-8") == xml
    assert not list((tmp_path / "run" / "screenshots").glob("*.xml"))
    assert observation.extra["ui_tree"][0]["resource_id"] == "tv.danmaku.bili:id/search_src_text"
    assert "hierarchy" not in json.dumps(observation.extra, ensure_ascii=False)


@pytest.mark.asyncio
async def test_adb_collect_ui_tree_clears_stale_artifacts_before_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
    <hierarchy rotation="0">
      <node index="0" text="搜索" resource-id="tv.danmaku.bili:id/search"
            class="android.widget.TextView" package="tv.danmaku.bili"
            content-desc="" clickable="true" focused="false" scrollable="false"
            bounds="[0,0][100,50]" />
    </hierarchy>"""
    backend = AdbBackend(use_scrcpy=False, collect_ui_tree=True)
    run_mock = AsyncMock(side_effect=["UI hierarchy dumped", xml])
    monkeypatch.setattr(backend, "_run", run_mock)

    screenshot = tmp_path / "run" / "screenshots" / "step_001.png"
    raw_xml_path = tmp_path / "run" / "ui_tree" / "step_001.xml"
    error_path = tmp_path / "run" / "ui_tree" / "step_001.error.json"
    raw_xml_path.parent.mkdir(parents=True)
    raw_xml_path.write_text("<stale />", encoding="utf-8")
    error_path.write_text("{}", encoding="utf-8")

    extra = await backend._collect_ui_tree_extra(timeout=5.0, screenshot_path=screenshot)

    assert raw_xml_path.read_text(encoding="utf-8") == xml
    assert not error_path.exists()
    assert extra["visible_text"] == ["搜索"]


@pytest.mark.asyncio
async def test_adb_collect_ui_tree_failure_removes_stale_xml_and_writes_error_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend(use_scrcpy=False, collect_ui_tree=True)
    run_mock = AsyncMock(side_effect=RuntimeError("dump failed"))
    monkeypatch.setattr(backend, "_run", run_mock)

    screenshot = tmp_path / "run" / "screenshots" / "step_001.png"
    raw_xml_path = tmp_path / "run" / "ui_tree" / "step_001.xml"
    error_path = tmp_path / "run" / "ui_tree" / "step_001.error.json"
    raw_xml_path.parent.mkdir(parents=True)
    raw_xml_path.write_text("<stale />", encoding="utf-8")

    extra = await backend._collect_ui_tree_extra(timeout=1.0, screenshot_path=screenshot)

    assert extra == {
        "ui_tree_error": "dump failed",
        "ui_tree_timeout_s": 15.0,
    }
    assert not raw_xml_path.exists()
    assert json.loads(error_path.read_text(encoding="utf-8")) == extra


def test_adb_ui_tree_marks_child_text_under_clickable_container_as_clickable() -> None:
    xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
    <hierarchy rotation="0">
      <node index="0" text="" resource-id="com.max.xiaoheihe:id/mall_entry"
            class="android.widget.LinearLayout" package="com.max.xiaoheihe"
            content-desc="" clickable="true" focused="false" scrollable="false"
            bounds="[700,300][980,370]">
        <node index="0" text="黑盒商城" resource-id="com.max.xiaoheihe:id/title"
              class="android.widget.TextView" package="com.max.xiaoheihe"
              content-desc="" clickable="false" focused="false" scrollable="false"
              bounds="[780,315][900,350]" />
      </node>
    </hierarchy>"""

    extra = _parse_ui_tree_xml(xml)

    assert extra["visible_text"] == ["黑盒商城"]
    assert extra["clickable_text"] == ["黑盒商城"]
    assert "ui_tree" not in extra
    assert evaluate_state_contract(
        {"must_exist": [{"text": "黑盒商城", "clickable": True}]},
        observation_extra=extra,
    ) is True


@pytest.mark.asyncio
async def test_adb_collect_ui_tree_uses_longer_timeout_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
    <hierarchy rotation="0">
      <node index="0" text="设置" resource-id="android:id/title"
            class="android.widget.TextView" package="com.android.settings"
            content-desc="" clickable="false" focused="false" scrollable="false"
            bounds="[0,0][100,50]" />
    </hierarchy>"""
    backend = AdbBackend(use_scrcpy=False, collect_ui_tree=True)
    run_mock = AsyncMock(side_effect=["UI hierarchy dumped", xml])
    monkeypatch.setattr(backend, "_run", run_mock)

    extra = await backend._collect_ui_tree_extra(timeout=30.0)

    assert run_mock.await_args_list[0].kwargs["timeout"] == 15.0
    assert run_mock.await_args_list[1].kwargs["timeout"] == 15.0
    assert extra["ui_tree_node_count"] == 1
    assert extra["visible_text"] == ["设置"]


@pytest.mark.asyncio
async def test_adb_collect_ui_tree_can_include_full_nodes_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
    <hierarchy rotation="0">
      <node index="0" text="设置" resource-id="android:id/title"
            class="android.widget.TextView" package="com.android.settings"
            content-desc="" clickable="false" focused="false" scrollable="false"
            bounds="[0,0][100,50]" />
    </hierarchy>"""
    backend = AdbBackend(use_scrcpy=False, collect_ui_tree=True, collect_ui_tree_nodes=True)
    run_mock = AsyncMock(side_effect=["UI hierarchy dumped", xml])
    monkeypatch.setattr(backend, "_run", run_mock)

    extra = await backend._collect_ui_tree_extra(timeout=5.0)

    assert extra["ui_tree_node_count"] == 1
    assert extra["ui_tree"][0]["text"] == "设置"


@pytest.mark.asyncio
async def test_adb_observe_queries_ui_context_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started: list[str] = []
    gate = asyncio.Event()

    async def _wait(name: str, result: Any) -> Any:
        started.append(name)
        await gate.wait()
        return result

    backend = AdbBackend(frame_source=FakeFrameSource(), collect_ui_tree=True)
    monkeypatch.setattr(backend, "_ensure_scrcpy_started", AsyncMock(return_value=None))
    monkeypatch.setattr(
        backend,
        "_capture_scrcpy_frame",
        AsyncMock(return_value=ScrcpyFrameSnapshot(width=320, height=640, timestamp=123.0)),
    )
    monkeypatch.setattr(backend, "_query_screen_size", lambda timeout: _wait("screen_size", (1080, 2376)))
    monkeypatch.setattr(backend, "_query_foreground_app", lambda timeout: _wait("foreground_app", "com.example.app"))
    monkeypatch.setattr(backend, "_collect_ui_tree_extra", lambda timeout, **_kwargs: _wait(
        "ui_tree",
        {
            "ui_tree_node_count": 1,
            "visible_text": ["设置"],
            "scrollable_present": True,
        },
    ))

    task = asyncio.create_task(backend.observe(tmp_path / "screen.png"))
    for _ in range(20):
        if len(started) == 3:
            break
        await asyncio.sleep(0)

    assert set(started) == {"screen_size", "foreground_app", "ui_tree"}
    gate.set()
    observation = await asyncio.wait_for(task, timeout=1.0)

    assert observation.foreground_app == "com.example.app"
    assert observation.extra["scrollable_present"] is True


@pytest.mark.asyncio
async def test_adb_scrcpy_relative_tap_uses_physical_input_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend(frame_source=FakeFrameSource())
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))
    await backend.observe(tmp_path / "screen.png")

    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = Action(action_type="tap", x=900, y=950, relative=True)
    await backend.execute(action)

    run_mock.assert_awaited_once_with("shell", "input", "tap", "972", "2259", timeout=5.0)


@pytest.mark.asyncio
async def test_adb_scrcpy_absolute_tap_maps_capture_pixels_to_input_pixels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend(frame_source=FakeFrameSource())
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))
    await backend.observe(tmp_path / "screen.png")

    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = Action(action_type="tap", x=288, y=576)
    await backend.execute(action)

    run_mock.assert_awaited_once_with("shell", "input", "tap", "974", "2141", timeout=5.0)


@pytest.mark.asyncio
async def test_adb_scrcpy_capture_keeps_actions_on_adb(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = AdbBackend(frame_source=FakeFrameSource())
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = Action(action_type="tap", x=1, y=2)
    assert await backend.execute(action) == "tap at (1, 2)"
    run_mock.assert_awaited_once_with("shell", "input", "tap", "1", "2", timeout=5.0)


def test_gui_config_accepts_adb_backend_with_scrcpy_camel_case_fields() -> None:
    config = GuiConfig.model_validate({
        "backend": "adb",
        "scrcpy": {
            "maxFps": 8,
            "jpegQuality": 70,
            "frameTimeoutMs": 1500,
            "maxFrameAgeMs": 500,
        },
    })

    assert config.backend == "adb"
    assert config.scrcpy.max_fps == 8
    assert config.scrcpy.jpeg_quality == 70
    assert config.scrcpy.frame_timeout_ms == 1500
    assert config.scrcpy.max_frame_age_ms == 500


def test_gui_config_accepts_ios_wda_url() -> None:
    config = GuiConfig.model_validate({
        "backend": "ios",
        "ios": {
            "wdaUrl": "http://localhost:8100",
        },
    })

    assert config.ios.wda_url == "http://localhost:8100"


def test_cli_builds_adb_backend_with_scrcpy_capture_config() -> None:
    config = cli.CliConfig(
        provider=cli.ProviderConfig(base_url="http://localhost:1/v1", model="m"),
        adb=cli.AdbConfig(serial="emulator-5554", adb_path="/tmp/adb"),
        scrcpy=cli.ScrcpyConfig(
            max_fps=7,
            jpeg_quality=65,
            frame_timeout_ms=1200,
            max_frame_age_ms=400,
        ),
    )

    backend = cli.build_backend("adb", config)

    assert isinstance(backend, AdbBackend)
    assert backend._scrcpy_frame_timeout_ms == 1200
    assert backend._scrcpy_max_frame_age_ms == 400
    assert backend._frame_source._adb_path == "/tmp/adb"  # type: ignore[attr-defined]
    assert backend._frame_source._frame_timeout_ms == 1200  # type: ignore[attr-defined]


def test_scrcpy_numpy_frames_are_converted_from_bgr_to_rgb() -> None:
    np = pytest.importorskip("numpy")
    frame = np.array([[[0, 0, 255]]], dtype=np.uint8)

    image = _pil_image_from_frame(frame)

    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 0, 0)
