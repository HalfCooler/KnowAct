from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from guiclaw import shortcut_validation as validator
from guiclaw.skills.deeplink import AppShortcutProfile, DeepIntent, DeepLink


def test_candidate_records_prioritize_search_and_skip_risky() -> None:
    profile = AppShortcutProfile(
        package="tv.danmaku.bili",
        deep_links=(
            DeepLink(
                uri_template="bilibili://bilipay/bcoin/recharge",
                scheme="bilibili",
                host="bilipay",
                path="/bcoin/recharge",
                component="tv.danmaku.bili/.Router",
                description="pay route",
            ),
            DeepLink(
                uri_template="bilibili://search",
                scheme="bilibili",
                host="search",
                path=None,
                component="tv.danmaku.bili/.Router",
                description="search route",
            ),
        ),
        deep_intents=(
            DeepIntent(
                action="android.intent.action.SEARCH",
                component="tv.danmaku.bili/.SearchActivity",
                description="search intent",
            ),
        ),
    )

    records = validator.candidate_records(profile, include_risky=False)

    assert [record.kind for record in records] == ["deeplink", "intent"]
    assert records[0].uri_template == "bilibili://search"


def test_candidate_records_do_not_treat_livehome_as_home_route() -> None:
    profile = AppShortcutProfile(
        package="tv.danmaku.bili",
        deep_links=(
            DeepLink(
                uri_template="bilibili://search",
                scheme="bilibili",
                host="search",
                path=None,
                component="tv.danmaku.bili/.Router",
                description="search route",
            ),
        ),
        deep_intents=(
            DeepIntent(
                action="android.intent.action.SEARCH",
                component="tv.danmaku.bili/com.bilibili.bililive.videoliveplayer.ui.live.home.LiveHomeActivity",
                description="live search",
            ),
        ),
    )

    records = validator.candidate_records(profile, include_risky=False)

    assert records[0].kind == "deeplink"
    assert records[0].uri_template == "bilibili://search"


def test_build_probe_plans_generates_search_plan_for_youtube() -> None:
    profile = AppShortcutProfile(
        package="com.google.android.youtube",
        deep_links=(),
        deep_intents=(
            DeepIntent(
                action="android.intent.action.SEARCH",
                component="com.google.android.youtube/.ResultsActivity",
                description="YouTube search",
            ),
        ),
    )
    args = SimpleNamespace(task="", query="", include_risky=False, max_candidates=8, max_try=6)

    plans = validator.build_probe_plans(profile, args)

    assert plans[0].capability == "search"
    assert plans[0].query == "Never Gonna Give You Up"
    assert "YouTube 搜索" in plans[0].task
    assert plans[0].candidate_limit == 8
    assert plans[0].variant_limit == 6


def test_build_probe_plans_uses_bilibili_default_query() -> None:
    profile = AppShortcutProfile(
        package="tv.danmaku.bili",
        deep_links=(
            DeepLink(
                uri_template="bilibili://search",
                scheme="bilibili",
                host="search",
                path=None,
                component="tv.danmaku.bili/.Router",
                description="search route",
            ),
        ),
    )
    args = SimpleNamespace(task="", query="", include_risky=False, max_candidates=5, max_try=4)

    plans = validator.build_probe_plans(profile, args)

    assert plans[0].capability == "search"
    assert plans[0].query == "敢杀我的马"


def test_build_probe_plans_covers_bilibili_route_domains() -> None:
    profile = AppShortcutProfile(
        package="tv.danmaku.bili",
        deep_links=(
            DeepLink(
                uri_template="bilibili://search",
                scheme="bilibili",
                host="search",
                path=None,
                component="tv.danmaku.bili/.Router",
                description="search route",
            ),
            DeepLink(
                uri_template="bilibili://live",
                scheme="bilibili",
                host="live",
                path=None,
                component="tv.danmaku.bili/.Router",
                description="live route",
            ),
            DeepLink(
                uri_template="bilibili://following",
                scheme="bilibili",
                host="following",
                path=None,
                component="tv.danmaku.bili/.Router",
                description="following route",
            ),
            DeepLink(
                uri_template="bilibili://history",
                scheme="bilibili",
                host="history",
                path=None,
                component="tv.danmaku.bili/.Router",
                description="history route",
            ),
            DeepLink(
                uri_template="bilibili://space",
                scheme="bilibili",
                host="space",
                path=None,
                component="tv.danmaku.bili/.Router",
                description="space route",
            ),
        ),
    )
    args = SimpleNamespace(
        task="", query="", include_risky=False, max_candidates=5, max_try=4, max_probe_plans=8
    )

    plans = validator.build_probe_plans(profile, args)
    capabilities = {plan.capability for plan in plans}

    assert {"search", "live", "social", "collection", "profile"} <= capabilities


def test_build_probe_plans_covers_intent_heavy_settings_cache() -> None:
    profile = AppShortcutProfile(
        package="com.android.settings",
        deep_intents=(
            DeepIntent(
                action="android.settings.WIFI_SETTINGS",
                component="com.android.settings/.Settings$WifiSettingsActivity",
                description="Wi-Fi settings",
            ),
            DeepIntent(
                action="android.settings.BLUETOOTH_SETTINGS",
                component="com.android.settings/.Settings$BluetoothDashboardActivity",
                description="Bluetooth settings",
            ),
        ),
    )
    args = SimpleNamespace(
        task="", query="", include_risky=False, max_candidates=5, max_try=4, max_probe_plans=8
    )

    plans = validator.build_probe_plans(profile, args)

    assert plans[0].capability == "settings_system"
    assert "系统设置页面" in plans[0].task


def test_build_probe_plans_keeps_specific_capabilities_with_small_counts() -> None:
    profile = AppShortcutProfile(
        package="com.xingin.xhs",
        deep_links=tuple(
            DeepLink(
                uri_template=f"xhsdiscover://note/{index}",
                scheme="xhsdiscover",
                host="note",
                path=f"/{index}",
                component="com.xingin.xhs/.Router",
                description="note content",
            )
            for index in range(6)
        )
        + (
            DeepLink(
                uri_template="xhsdiscover://poi/location",
                scheme="xhsdiscover",
                host="poi",
                path="/location",
                component="com.xingin.xhs/.Router",
                description="poi route",
            ),
        ),
        deep_intents=(
            DeepIntent(
                action="android.appwidget.action.APPWIDGET_CONFIGURE",
                component="com.xingin.xhs/.WidgetConfigureActivity",
                description="widget quick action",
            ),
        ),
    )
    args = SimpleNamespace(
        task="", query="", include_risky=False, max_candidates=5, max_try=4, max_probe_plans=3
    )

    plans = validator.build_probe_plans(profile, args)
    capabilities = [plan.capability for plan in plans]

    assert "poi_location" in capabilities
    assert "widget_quick_action" in capabilities


def test_build_probe_plans_manual_override_keeps_user_payload() -> None:
    profile = AppShortcutProfile(
        package="com.google.android.youtube",
        deep_intents=(
            DeepIntent(
                action="android.intent.action.SEARCH",
                component="com.google.android.youtube/.ResultsActivity",
                description="search",
            ),
        ),
    )
    args = SimpleNamespace(
        task="验证 YouTube 自定义搜索",
        query="lofi hip hop",
        include_risky=False,
        max_candidates=3,
        max_try=2,
    )

    plans = validator.build_probe_plans(profile, args)

    assert plans == [
        validator.ProbePlan(
            capability="manual",
            task="验证 YouTube 自定义搜索",
            query="lofi hip hop",
            candidate_limit=3,
            variant_limit=2,
        )
    ]


def test_search_probe_plan_filters_account_linking_candidate() -> None:
    plan = validator.ProbePlan(
        capability="search",
        task="验证 YouTube 搜索",
        query="Never Gonna Give You Up",
        candidate_limit=8,
        variant_limit=6,
    )
    account_linking = validator.Candidate(
        index=0,
        kind="deeplink",
        package="com.google.android.youtube",
        description="account linking",
        uri_template="vnd.youtube.gdi:",
        component="com.google.android.youtube/.AccountLinkingActivity",
    )
    search = validator.Candidate(
        index=1,
        kind="intent",
        package="com.google.android.youtube",
        description="search",
        action="android.intent.action.SEARCH",
        component="com.google.android.youtube/.ResultsActivity",
    )

    assert validator.candidate_matches_plan(account_linking, plan) is False
    assert validator.candidate_matches_plan(search, plan) is True


def test_open_page_probe_plan_does_not_duplicate_search_candidates() -> None:
    plan = validator.ProbePlan(
        capability="open_page",
        task="验证 YouTube 页面",
        query="",
        candidate_limit=8,
        variant_limit=6,
    )
    search_play = validator.Candidate(
        index=1,
        kind="intent",
        package="com.google.android.youtube",
        description="play from search",
        action="android.media.action.MEDIA_PLAY_FROM_SEARCH",
        component="com.google.android.youtube/.MediaSearchActivity",
    )
    watch = validator.Candidate(
        index=2,
        kind="deeplink",
        package="com.google.android.youtube",
        description="watch video",
        uri_template="vnd.youtube:",
        component="com.google.android.youtube/.UrlActivity",
    )

    assert validator.candidate_matches_plan(search_play, plan) is False
    assert validator.candidate_matches_plan(watch, plan) is True


def test_candidates_for_plan_prefers_deeplink_before_generic_search_intents() -> None:
    plan = validator.ProbePlan(
        capability="search",
        task="验证 B站 搜索",
        query="敢杀我的马",
        candidate_limit=3,
        variant_limit=4,
    )
    generic_intent = validator.Candidate(
        index=1,
        kind="intent",
        package="tv.danmaku.bili",
        description="generic search",
        action="android.intent.action.SEARCH",
        component="tv.danmaku.bili/.FavoriteBoxActivity",
    )
    deeplink = validator.Candidate(
        index=2,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="search route",
        uri_template="bilibili://search",
        component="tv.danmaku.bili/.Router",
    )

    candidates = validator.candidates_for_plan([generic_intent, deeplink], plan)

    assert candidates[0] == deeplink
    assert candidates[1] == generic_intent


def test_infer_candidate_capabilities_covers_web_publish_camera_poi_widget() -> None:
    candidates = [
        validator.Candidate(
            index=0,
            kind="deeplink",
            package="com.android.chrome",
            description="Chrome URL route",
            uri_template="googlechrome://navigate?url=https://example.com",
        ),
        validator.Candidate(
            index=1,
            kind="deeplink",
            package="com.xingin.xhs",
            description="embedded webview",
            uri_template="xhsdiscover://extweb",
        ),
        validator.Candidate(
            index=2,
            kind="intent",
            package="com.google.android.youtube",
            description="share media for upload",
            action="android.intent.action.SEND",
            component="com.google.android.youtube/.UploadActivity",
            mime_type="video/mp4",
        ),
        validator.Candidate(
            index=6,
            kind="intent",
            package="com.google.android.youtube",
            description="multiple uploads entry",
            action="android.intent.action.SEND_MULTIPLE",
            component="com.google.android.youtube/.Shell_MultipleUploadsActivity",
        ),
        validator.Candidate(
            index=3,
            kind="deeplink",
            package="com.xingin.xhs",
            description="QR scanner",
            uri_template="xhsdiscover://qrscan",
        ),
        validator.Candidate(
            index=4,
            kind="deeplink",
            package="com.xingin.xhs",
            description="nearby POI",
            uri_template="xhsdiscover://poi/location",
        ),
        validator.Candidate(
            index=5,
            kind="intent",
            package="md.obsidian",
            description="app widget quick action",
            action="android.appwidget.action.APPWIDGET_CONFIGURE",
            component="md.obsidian/.WidgetConfigureActivity",
        ),
    ]

    capabilities = set().union(
        *(set(validator.infer_candidate_capabilities(candidate)) for candidate in candidates)
    )

    assert {
        "browser_web",
        "web_container",
        "publish_upload",
        "camera_scan_effect",
        "poi_location",
        "widget_quick_action",
    } <= capabilities
    assert "publish_upload" in validator.infer_candidate_capabilities(candidates[3])


def test_file_document_is_not_inferred_from_image_search_but_uses_mime() -> None:
    file_plan = validator.ProbePlan(
        capability="file_document",
        task="验证文件入口",
        query="",
        candidate_limit=8,
        variant_limit=4,
    )
    image_search = validator.Candidate(
        index=0,
        kind="deeplink",
        package="com.xingin.xhs",
        description="image search",
        uri_template="xhsdiscover://image_search",
    )
    image_share = validator.Candidate(
        index=1,
        kind="intent",
        package="com.simplemobiletools.gallery.pro",
        description="open image",
        action="android.intent.action.SEND",
        mime_type="image/jpeg",
    )

    assert "file_document" not in validator.infer_candidate_capabilities(image_search)
    assert validator.candidate_matches_plan(image_search, file_plan) is False
    assert "file_document" in validator.infer_candidate_capabilities(image_share)
    assert validator.candidate_matches_plan(image_share, file_plan) is True


def test_settings_system_does_not_claim_app_notification_routes() -> None:
    app_notification = validator.Candidate(
        index=0,
        kind="deeplink",
        package="com.xingin.xhs",
        description="interaction notification",
        uri_template="xhsdiscover://message/notifications",
    )
    system_notification = validator.Candidate(
        index=1,
        kind="intent",
        package="com.android.settings",
        description="Android notification settings",
        action="android.settings.NOTIFICATION_SETTINGS",
        component="com.android.settings/.Settings$NotificationSettingsActivity",
    )

    assert "settings_system" not in validator.infer_candidate_capabilities(app_notification)
    assert "social" in validator.infer_candidate_capabilities(app_notification)
    assert "settings_system" in validator.infer_candidate_capabilities(system_notification)


def test_non_search_plans_exclude_search_like_candidates() -> None:
    plan = validator.ProbePlan(
        capability="content",
        task="验证内容页",
        query="",
        candidate_limit=8,
        variant_limit=4,
    )
    play_from_search = validator.Candidate(
        index=0,
        kind="intent",
        package="com.google.android.youtube",
        description="play from search",
        action="android.media.action.MEDIA_PLAY_FROM_SEARCH",
        component="com.google.android.youtube/.MediaSearchActivity",
    )
    watch = validator.Candidate(
        index=1,
        kind="deeplink",
        package="com.google.android.youtube",
        description="watch video",
        uri_template="vnd.youtube://watch",
    )

    assert validator.candidate_matches_plan(play_from_search, plan) is False
    assert validator.candidates_for_plan([play_from_search, watch], plan) == [watch]


def test_auto_probe_plans_skip_debug_remote_and_transit_noise() -> None:
    plan = validator.ProbePlan(
        capability="browser_web",
        task="验证 Web 页面",
        query="",
        candidate_limit=8,
        variant_limit=4,
    )
    noise_candidates = [
        validator.Candidate(
            index=0,
            kind="intent",
            package="com.android.chrome",
            description="dummy action",
            action="com.example.dummy.action.TEST",
        ),
        validator.Candidate(
            index=1,
            kind="intent",
            package="com.android.chrome",
            description="remote action receiver",
            action="com.google.android.apps.chrome.REMOTE_ACTION",
        ),
        validator.Candidate(
            index=2,
            kind="deeplink",
            package="com.android.chrome",
            description="debug web route",
            uri_template="googlechrome://debug",
        ),
        validator.Candidate(
            index=3,
            kind="deeplink",
            package="com.xingin.xhs",
            description="getui push transit",
            uri_template="xhsdiscover://getui/push/transit",
        ),
        validator.Candidate(
            index=5,
            kind="deeplink",
            package="com.xingin.xhs",
            description="rn autotest route",
            uri_template="xhsdiscover://rnautotest",
        ),
    ]
    valid = validator.Candidate(
        index=4,
        kind="deeplink",
        package="com.android.chrome",
        description="Chrome URL route",
        uri_template="googlechrome://navigate?url=https://example.com",
    )

    assert all(
        validator.candidate_matches_plan(candidate, plan) is False for candidate in noise_candidates
    )
    assert validator.candidates_for_plan([*noise_candidates, valid], plan) == [valid]


def test_deeplink_variants_include_encoded_chinese_query() -> None:
    candidate = validator.Candidate(
        index=0,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="search",
        uri_template="bilibili://search",
        component="tv.danmaku.bili/.Router",
    )

    variants = validator.variants_for_deeplink(candidate, query="敢杀我的马", max_try=6)

    assert variants[0].label == "raw_package"
    assert variants[1].label == "raw_component"
    assert any(
        "%E6%95%A2%E6%9D%80%E6%88%91%E7%9A%84%E9%A9%AC" in (variant.uri or "")
        for variant in variants
    )
    assert all(variant.package == "tv.danmaku.bili" for variant in variants)


def test_deeplink_variants_plus_encoding_with_space() -> None:
    candidate = validator.Candidate(
        index=0,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="search",
        uri_template="bilibili://search",
        component="tv.danmaku.bili/.Router",
    )

    variants = validator.variants_for_deeplink(candidate, query="phone 壳", max_try=12)

    assert any("phone+%E5%A3%B3" in (variant.uri or "") for variant in variants)


def test_intent_variants_include_query_extras() -> None:
    candidate = validator.Candidate(
        index=1,
        kind="intent",
        package="tv.danmaku.bili",
        description="search",
        action="android.intent.action.SEARCH",
        component="tv.danmaku.bili/.SearchActivity",
    )

    variants = validator.variants_for_intent(candidate, query="敢杀我的马", max_try=5)

    assert variants[0].label == "component_no_extra"
    assert any(("query", "敢杀我的马") in variant.extras for variant in variants)
    assert any(variant.component is None for variant in variants)


def test_build_launch_args_quotes_shell_values_with_spaces() -> None:
    variant = validator.Variant(
        label="package_extra_query",
        kind="intent",
        package="com.google.android.youtube",
        action="android.intent.action.SEARCH",
        extras=(("query", "Never Gonna Give You Up"),),
    )

    args = validator.build_launch_args(variant)

    assert args[args.index("--es") + 2] == "'Never Gonna Give You Up'"
    assert args[-2:] == ["-p", "com.google.android.youtube"]


def test_upload_intent_variants_include_probe_media_payload() -> None:
    candidate = validator.Candidate(
        index=0,
        kind="intent",
        package="com.google.android.youtube",
        description="YouTube upload",
        action="android.intent.action.SEND",
        component="com.google.android.youtube/.UploadActivity",
        mime_type="image/*",
    )

    variants = validator.variants_for_intent(candidate, query="", max_try=8)
    stream_variant = next(
        variant for variant in variants if variant.label == "component_probe_media_stream"
    )
    args = validator.build_launch_args(stream_variant)

    assert stream_variant.mime_type == "image/*"
    assert ("android.intent.extra.STREAM", validator.PROBE_UPLOAD_URI) in stream_variant.extras
    assert "--grant-read-uri-permission" in args
    assert "--eu" in args
    assert args[args.index("--eu") + 2] == validator.PROBE_UPLOAD_URI


def test_chrome_candidate_records_add_synthetic_browser_and_search_urls() -> None:
    profile = AppShortcutProfile(package="com.android.chrome")

    records = validator.candidate_records(profile, include_risky=False)
    uris = {record.uri_template for record in records}
    plans = validator.build_probe_plans(
        profile,
        SimpleNamespace(
            task="",
            query="",
            include_risky=False,
            max_candidates=8,
            max_try=5,
            max_probe_plans=4,
        ),
    )

    assert "https://example.com" in uris
    assert "https://www.google.com/search" in uris
    assert any(plan.capability == "search" for plan in plans)
    assert any(plan.capability == "browser_web" for plan in plans)


def test_parse_json_object_accepts_fenced_json() -> None:
    parsed = validator.parse_json_object(
        """```json
        {"usable": true, "status": "page_validated", "description": "B站搜索视频"}
        ```"""
    )

    assert parsed["usable"] is True
    assert parsed["description"] == "B站搜索视频"


def test_result_to_validation_record_for_intent() -> None:
    candidate = validator.Candidate(
        index=1,
        kind="intent",
        package="tv.danmaku.bili",
        description="search",
        action="android.intent.action.SEARCH",
        component="tv.danmaku.bili/.SearchActivity",
    )
    result = validator.ProbeResult(
        candidate=candidate,
        status="page_validated",
        description="B站搜索视频",
        parameters=["query"],
        best_variant={
            "label": "component_extra_query",
            "kind": "intent",
            "package": "tv.danmaku.bili",
            "action": "android.intent.action.SEARCH",
            "component": "tv.danmaku.bili/.SearchActivity",
            "extras": [["query", "{{query}}"]],
        },
    )

    record = validator.result_to_validation_record(result)

    assert record == {
        "package": "tv.danmaku.bili",
        "kind": "intent",
        "status": "page_validated",
        "description": "B站搜索视频",
        "parameters": ["query"],
        "valid_state": validator.SHORTCUT_SKIP_VALID_STATE,
        "intent_action": "android.intent.action.SEARCH",
        "extras": [["query", "{{query}}"]],
        "component": "tv.danmaku.bili/.SearchActivity",
    }


def test_result_to_validation_record_templates_query_payload() -> None:
    candidate = validator.Candidate(
        index=1,
        kind="intent",
        package="tv.danmaku.bili",
        description="search",
        action="android.intent.action.SEARCH",
        component="tv.danmaku.bili/.SearchActivity",
    )
    result = validator.ProbeResult(
        candidate=candidate,
        status="page_validated",
        description="B站搜索视频",
        parameters=["query=敢杀我的马"],
        best_variant={
            "label": "component_extra_query",
            "kind": "intent",
            "package": "tv.danmaku.bili",
            "action": "android.intent.action.SEARCH",
            "component": "tv.danmaku.bili/.SearchActivity",
            "extras": [["query", "敢杀我的马"]],
        },
    )

    record = validator.result_to_validation_record(result, query="敢杀我的马")

    assert record["parameters"] == ["query"]
    assert record["extras"] == [["query", "{{query}}"]]


def test_result_to_validation_record_templates_probe_media_payload() -> None:
    candidate = validator.Candidate(
        index=1,
        kind="intent",
        package="com.google.android.youtube",
        description="upload",
        action="android.intent.action.SEND",
        component="com.google.android.youtube/.UploadActivity",
    )
    result = validator.ProbeResult(
        candidate=candidate,
        status="page_validated",
        description="YouTube 上传入口",
        parameters=[],
        best_variant={
            "label": "component_probe_media_stream",
            "kind": "intent",
            "package": "com.google.android.youtube",
            "action": "android.intent.action.SEND",
            "component": "com.google.android.youtube/.UploadActivity",
            "mime_type": "image/*",
            "extras": [["android.intent.extra.STREAM", validator.PROBE_UPLOAD_URI]],
            "flags": ["--grant-read-uri-permission"],
        },
    )

    record = validator.result_to_validation_record(result)

    assert record["parameters"] == ["media_uri"]
    assert record["extras"] == [["android.intent.extra.STREAM", "{{media_uri}}"]]


def test_result_to_validation_record_drops_unused_query_parameter() -> None:
    candidate = validator.Candidate(
        index=0,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="search",
        uri_template="bilibili://search",
    )
    result = validator.ProbeResult(
        candidate=candidate,
        status="page_validated",
        description="打开B站搜索页",
        parameters=["query=敢杀我的马"],
        best_variant={
            "label": "raw_package",
            "kind": "deeplink",
            "package": "tv.danmaku.bili",
            "uri": "bilibili://search",
        },
    )

    record = validator.result_to_validation_record(result, query="敢杀我的马")

    assert record["parameters"] == []
    assert record["uri_template"] == "bilibili://search"


def test_promote_results_forces_shortcut_valid_state(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    async def fake_add_validated_shortcut_skill(library, record, *, require_page_validated=True):
        calls.append(dict(record))
        return "added", "shortcut:di:pkg:test"

    monkeypatch.setattr(
        validator, "add_validated_shortcut_skill", fake_add_validated_shortcut_skill
    )
    args = SimpleNamespace(
        promote=True,
        skill_store_root=tmp_path,
        allow_launchable_promote=False,
    )
    records = [
        {
            "package": "pkg",
            "kind": "intent",
            "status": "page_validated",
            "description": "打开搜索页",
            "valid_state": "old page state",
            "intent_action": "android.intent.action.SEARCH",
        }
    ]

    promotions = asyncio.run(validator.promote_results(args, records))

    assert records[0]["valid_state"] == "old page state"
    assert calls[0]["valid_state"] == validator.SHORTCUT_SKIP_VALID_STATE
    assert promotions[0]["record"]["valid_state"] == validator.SHORTCUT_SKIP_VALID_STATE


def test_write_sidecar_records_probe_plan(tmp_path: Path) -> None:
    profile = AppShortcutProfile(package="tv.danmaku.bili")
    plan = validator.ProbePlan(
        capability="search",
        task="验证 B站搜索",
        query="敢杀我的马",
        candidate_limit=5,
        variant_limit=4,
    )
    result = validator.ProbeResult(
        candidate=validator.Candidate(
            index=0,
            kind="deeplink",
            package="tv.danmaku.bili",
            description="search",
            uri_template="bilibili://search",
        ),
        status="page_validated",
        description="B站搜索视频",
        parameters=["query"],
        best_variant={
            "label": "keyword_raw",
            "kind": "deeplink",
            "package": "tv.danmaku.bili",
            "uri": "bilibili://search?keyword=敢杀我的马",
        },
        probe_plan=validator.probe_plan_to_dict(plan),
    )
    args = SimpleNamespace(
        serial="emulator-5554",
        cache=Path("shortcut_cache/tv.danmaku.bili.json"),
        task="",
        query="",
        execute=True,
        llm_model="qwen-vl-plus",
    )
    path = tmp_path / "sidecar.json"

    validator.write_sidecar(
        path=path,
        profile=profile,
        args=args,
        results=[result],
        promotions=[],
        probe_plans=[plan],
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["probe_plans"] == [validator.probe_plan_to_dict(plan)]
    assert data["stopped_early"] is None
    assert data["results"][0]["probe_plan"] == validator.probe_plan_to_dict(plan)
    assert (
        data["results"][0]["validation_record"]["uri_template"]
        == "bilibili://search?keyword={{query}}"
    )


def test_launch_variant_skips_evidence_capture_when_disabled(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeAdb:
        def run(self, *args, timeout=10.0):
            calls.append(" ".join(args))
            return 0, "Status: ok", False

    monkeypatch.setattr(
        validator, "foreground_activity", lambda adb: "tv.danmaku.bili/.SearchActivity"
    )
    monkeypatch.setattr(
        validator,
        "capture_screenshot",
        lambda adb, path: (_ for _ in ()).throw(AssertionError("screenshot")),
    )
    monkeypatch.setattr(
        validator, "capture_ui_tree", lambda adb: (_ for _ in ()).throw(AssertionError("ui tree"))
    )

    result = validator.launch_variant(
        FakeAdb(),
        validator.Variant(
            label="raw",
            kind="deeplink",
            package="tv.danmaku.bili",
            uri="bilibili://search",
        ),
        artifacts_dir=tmp_path,
        index=1,
        capture_evidence=False,
        settle_seconds=0,
    )

    assert result["target_package"] is True
    assert result["screenshot_path"] is None
    assert result["ui_tree"] == ""
    assert len(calls) == 2


def test_launch_variant_prepares_probe_media_payload(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    class FakeAdb:
        def run(self, *args, timeout=10.0):
            calls.append(tuple(args))
            return 0, "Status: ok", False

    monkeypatch.setattr(
        validator, "foreground_activity", lambda adb: "com.google.android.youtube/.UploadActivity"
    )

    result = validator.launch_variant(
        FakeAdb(),
        validator.Variant(
            label="component_probe_media_stream",
            kind="intent",
            package="com.google.android.youtube",
            action="android.intent.action.SEND",
            component="com.google.android.youtube/.UploadActivity",
            mime_type="image/*",
            extras=(("android.intent.extra.STREAM", validator.PROBE_UPLOAD_URI),),
            flags=("--grant-read-uri-permission",),
        ),
        artifacts_dir=tmp_path,
        index=1,
        capture_evidence=False,
        settle_seconds=0,
    )

    assert result["target_package"] is True
    assert any(
        call[:1] == ("push",) and call[-1] == validator.PROBE_UPLOAD_REMOTE_PATH for call in calls
    )


def test_validate_candidate_breaks_after_launchable_without_verifier(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = validator.Candidate(
        index=0,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="search",
        uri_template="bilibili://search",
    )
    args = SimpleNamespace(
        query="敢杀我的马",
        max_try=5,
        execute=True,
        llm_base_url="",
        llm_model="",
    )
    launched: list[str] = []

    monkeypatch.setattr(validator, "resolve_variant", lambda adb, variant: {"ok": True})

    def fake_launch(
        adb, variant, *, artifacts_dir, index, capture_evidence=False, settle_seconds=2.0
    ):
        launched.append(variant.label)
        assert capture_evidence is False
        return {"target_package": True, "foreground": "tv.danmaku.bili/.SearchActivity"}

    monkeypatch.setattr(validator, "launch_variant", fake_launch)

    result = validator.validate_candidate(object(), candidate, args=args, artifacts_dir=tmp_path)

    assert result.status == "launchable"
    assert launched == ["raw_package"]


def test_validate_candidate_normalizes_usable_launchable_to_page_validated(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = validator.Candidate(
        index=0,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="podcast",
        uri_template="bilibili://podcast",
    )
    args = SimpleNamespace(
        query="",
        max_try=2,
        execute=True,
        llm_base_url="http://example.test/v1",
        llm_model="qwen-vl",
        llm_api_key="ok",
        llm_api_key_env="DASHSCOPE_API_KEY",
        llm_temperature=0.0,
        task="验证 B站 内容页",
    )

    monkeypatch.setattr(validator, "resolve_variant", lambda adb, variant: {"ok": True})
    monkeypatch.setattr(
        validator,
        "launch_variant",
        lambda adb, variant, *, artifacts_dir, index, capture_evidence=False, settle_seconds=2.0: {
            "target_package": True,
            "foreground": "tv.danmaku.bili/.PodcastActivity",
        },
    )
    monkeypatch.setattr(
        validator,
        "verify_with_llm",
        lambda **kwargs: {
            "usable": True,
            "status": "launchable",
            "description": "B站听视频页面",
            "parameters": [],
            "payload_preserved": True,
            "reason": "opened",
        },
    )

    result = validator.validate_candidate(object(), candidate, args=args, artifacts_dir=tmp_path)

    assert result.status == "page_validated"
    assert result.variants[0]["normalized_status"] == "page_validated"


def test_validate_candidate_stops_on_verifier_error(monkeypatch, tmp_path: Path) -> None:
    candidate = validator.Candidate(
        index=0,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="search",
        uri_template="bilibili://search",
    )
    args = SimpleNamespace(
        query="敢杀我的马",
        max_try=5,
        execute=True,
        llm_base_url="http://example.test/v1",
        llm_model="qwen-vl",
        llm_api_key="bad",
        llm_api_key_env="DASHSCOPE_API_KEY",
        llm_temperature=0.0,
        task="验证 B站 搜索",
    )
    launched: list[str] = []

    monkeypatch.setattr(validator, "resolve_variant", lambda adb, variant: {"ok": True})

    def fake_launch(
        adb, variant, *, artifacts_dir, index, capture_evidence=False, settle_seconds=2.0
    ):
        launched.append(variant.label)
        return {"target_package": True, "foreground": "tv.danmaku.bili/.SearchActivity"}

    monkeypatch.setattr(validator, "launch_variant", fake_launch)
    monkeypatch.setattr(
        validator,
        "verify_with_llm",
        lambda **kwargs: {"usable": False, "status": "verifier_error", "reason": "401 invalid key"},
    )

    result = validator.validate_candidate(object(), candidate, args=args, artifacts_dir=tmp_path)

    assert result.status == "verifier_error"
    assert result.reason == "401 invalid key"
    assert launched == ["raw_package"]


def test_plan_satisfied_stops_per_capability_on_page_validated_or_launchable_without_verifier() -> (
    None
):
    candidate = validator.Candidate(index=0, kind="deeplink", package="pkg", description="desc")
    page_validated = validator.ProbeResult(candidate=candidate, status="page_validated")
    launchable = validator.ProbeResult(candidate=candidate, status="launchable")

    no_verifier_args = SimpleNamespace(execute=True, llm_base_url="", llm_model="")
    verifier_args = SimpleNamespace(
        execute=True, llm_base_url="http://example.test/v1", llm_model="qwen-vl"
    )

    assert validator.plan_satisfied(page_validated, verifier_args) is True
    assert validator.plan_satisfied(launchable, no_verifier_args) is True
    assert validator.plan_satisfied(launchable, verifier_args) is False


def test_candidates_for_plan_skips_globally_validated_candidates() -> None:
    plan = validator.ProbePlan(
        capability="social",
        task="验证社交页",
        query="",
        candidate_limit=8,
        variant_limit=4,
    )
    publish = validator.Candidate(
        index=0,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="publish",
        uri_template="bilibili://following2/publishInfo",
    )
    im = validator.Candidate(
        index=1,
        kind="deeplink",
        package="tv.danmaku.bili",
        description="im",
        uri_template="bilibili://im",
    )

    candidates = validator.candidates_for_plan(
        [publish, im],
        plan,
        skip_candidate_keys={validator.candidate_key(publish)},
    )

    assert publish not in candidates
    assert im in candidates


def test_dedupe_validation_records_keeps_first_matching_shortcut() -> None:
    records = [
        {
            "package": "com.google.android.youtube",
            "kind": "intent",
            "status": "page_validated",
            "description": "usage",
            "intent_action": "com.google.android.apps.wellbeing.VIEW_APP_USAGE",
            "component": "com.google.android.youtube/.WatchWhileActivity",
        },
        {
            "package": "com.google.android.youtube",
            "kind": "intent",
            "status": "page_validated",
            "description": "usage duplicate",
            "intent_action": "com.google.android.apps.wellbeing.VIEW_APP_USAGE",
            "component": "com.google.android.youtube/.WatchWhileActivity",
        },
    ]

    assert validator.dedupe_validation_records(records) == [records[0]]


def test_validate_verifier_config_fails_when_env_name_does_not_resolve(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    args = SimpleNamespace(
        llm_base_url="http://example.test/v1",
        llm_model="qwen-vl",
        llm_api_key="",
        llm_api_key_env="DASHSCOPE_API_KEY",
    )

    try:
        validator.validate_verifier_config(args)
    except SystemExit as exc:
        assert "--llm-api-key-env DASHSCOPE_API_KEY" in str(exc)
    else:
        raise AssertionError("validate_verifier_config should fail without an API key")


def test_verify_with_llm_returns_verifier_error_on_api_exception(monkeypatch) -> None:
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("rate limit")

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FailingCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))

    verdict = validator.verify_with_llm(
        base_url="http://example.test/v1",
        model="test",
        api_key="test",
        temperature=0.0,
        task="test",
        candidate=validator.Candidate(index=0, kind="deeplink", package="pkg", description="desc"),
        variant=validator.Variant(label="raw", kind="deeplink", package="pkg", uri="x://y"),
        launch={"foreground": "pkg/.A"},
    )

    assert verdict["usable"] is False
    assert verdict["status"] == "verifier_error"
