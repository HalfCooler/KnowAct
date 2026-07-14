"""Validate Android shortcut cache entries on a connected device.

The module keeps shortcut discovery and validation separate:

* ``shortcut_cache/*.json`` remains the static discovery artifact.
* runtime validation writes device-specific sidecars.
* only page-validated records are promoted to ``skills.py`` by default.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus, urlparse, urlunparse

from guiclaw.skills.deeplink import (
    AppShortcutProfile,
    add_validated_shortcut_skill,
)
from guiclaw.skills.flat import DEFAULT_SKILLS_STORE_DIR, FlatSkillLibrary

VIEW_ACTION = "android.intent.action.VIEW"
BROWSABLE_CATEGORY = "android.intent.category.BROWSABLE"
SHORTCUT_SKIP_VALID_STATE = "No need to verify"
DEFAULT_QUERY_KEYS = ("keyword", "query", "q", "text", "word")
DEFAULT_INTENT_EXTRA_KEYS = (
    "query",
    "keyword",
    "text",
    "android.intent.extra.TEXT",
    "user_query",
)
ANDROID_EXTRA_STREAM = "android.intent.extra.STREAM"
BROWSER_PACKAGES = frozenset({"com.android.chrome"})
PROBE_UPLOAD_REMOTE_PATH = "/sdcard/Download/nanobot_probe_upload.png"
PROBE_UPLOAD_URI = f"file://{PROBE_UPLOAD_REMOTE_PATH}"
PROBE_UPLOAD_LOCAL_NAME = "nanobot_probe_upload.png"
PROBE_UPLOAD_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
GRANT_READ_URI_PERMISSION_FLAG = "--grant-read-uri-permission"
RISKY_TOKENS = (
    "pay",
    "bilipay",
    "recharge",
    "share",
    "push",
    "authorize",
    "auth",
    "accountlinking",
    "account linking",
    "login",
    "sso",
    "nfc",
    "sign",
    "order",
    "purchase",
)
DEFAULT_MAX_PROBE_PLANS = 12
GENERIC_INTENT_ACTIONS = (
    "android.intent.action.search",
    "android.media.action.media_play_from_search",
    "android.intent.action.send",
    "android.intent.action.send_multiple",
    "android.intent.action.view",
    "android.intent.action.pick",
    "android.intent.action.get_content",
)
FILE_DOCUMENT_ACTIONS = (
    "android.intent.action.open_document",
    "android.intent.action.create_document",
    "android.intent.action.get_content",
    "android.intent.action.pick",
)
PUBLISH_UPLOAD_ACTIONS = (
    "android.intent.action.send",
    "android.intent.action.send_multiple",
)
FILE_DOCUMENT_MIME_PREFIXES = (
    "application/",
    "audio/",
    "image/",
    "text/",
    "video/",
)
PROBE_NOISE_SUBSTRINGS = (
    "autotest",
    "debug",
    "dummy.action",
    "getui",
    "remote_action",
)
PROBE_NOISE_PREFIX_TERMS = (
    "autotest",
    "debug",
    "getui",
)
PROBE_NOISE_EXACT_TERMS = (
    "push",
    "transit",
)
CAPABILITY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "search",
        (
            "search",
            "query",
            "keyword",
            "find",
            "results",
            "搜索",
            "media_search",
            "media_play_from_search",
        ),
    ),
    ("live", ("live", "liveroom", "livearea", "直播")),
    (
        "social",
        (
            "message",
            "messages",
            "im",
            "chat",
            "following",
            "following2",
            "follow",
            "fans",
            "interaction",
            "关注",
            "动态",
            "消息",
        ),
    ),
    (
        "commerce",
        (
            "market",
            "shop",
            "store",
            "details",
            "billing",
            "vip",
            "charge",
            "purchase",
            "pay",
            "order",
            "会员",
            "支付",
        ),
    ),
    ("game", ("game", "游戏")),
    (
        "content",
        (
            "video",
            "watch",
            "play",
            "article",
            "music",
            "podcast",
            "story",
            "note",
            "feed",
            "pgc",
            "bangumi",
            "topic",
            "shorts",
            "内容",
        ),
    ),
    (
        "profile",
        (
            "profile",
            "space",
            "author",
            "user",
            "account",
            "people",
            "person",
            "contact",
            "个人",
            "空间",
        ),
    ),
    (
        "collection",
        (
            "favorite",
            "favorites",
            "bookmark",
            "history",
            "collection",
            "playlist",
            "download",
            "收藏",
            "历史",
        ),
    ),
    (
        "settings_system",
        (
            "settings",
            "setting",
            "wifi",
            "bluetooth",
            "network",
            "display",
            "brightness",
            "privacy",
            "system",
            "android.settings",
        ),
    ),
    ("clock_alarm_timer", ("clock", "alarm", "timer", "deskclock")),
    ("calendar_event", ("calendar", "schedule", "agenda")),
    ("contacts_people", ("contacts", "contact", "people", "dial", "call", "tel:")),
    (
        "file_document",
        (
            "file",
            "files",
            "document",
            "documents",
            "open_document",
            "create_document",
            "get_content",
            "openable",
            "mime",
            "picker",
        ),
    ),
    (
        "browser_web",
        (
            "http",
            "https",
            "url",
            "browser",
            "chrome",
            "googlechrome",
            "customtab",
            "customtabs",
            "trusted_web_activity",
            "webapp",
            "webapps",
            "webapk",
            "translate",
        ),
    ),
    (
        "web_container",
        (
            "webview",
            "web_view",
            "web_container",
            "webcontainer",
            "extweb",
            "hybrid",
            "h5",
            "jsbridge",
            "miniapp",
            "miniprogram",
            "miniversion",
            "reactnative",
            "react_native",
            "rn",
            "rnpage",
            "rn_page",
        ),
    ),
    (
        "publish_upload",
        (
            "upload",
            "publish",
            "post",
            "compose",
            "create_video",
            "creator",
            "draft",
            "notes_draft",
            "notes_draft_box",
        ),
    ),
    (
        "camera_scan_effect",
        (
            "camera",
            "scan",
            "scanner",
            "qrcode",
            "qrscan",
            "barcode",
            "effect",
            "filter",
            "face",
            "face_photo",
            "skin_detection",
            "ar_skin_detection",
            "ar",
        ),
    ),
    (
        "poi_location",
        (
            "poi",
            "location",
            "locations",
            "nearby",
            "localfeed",
            "map",
            "maps",
            "geo:",
            "place",
            "places",
            "city",
        ),
    ),
    (
        "widget_quick_action",
        (
            "widget",
            "appwidget",
            "remoteviews",
            "shortcut",
            "quick_action",
            "quicksettings",
            "quick_settings",
            "qs_tile",
        ),
    ),
    ("app_entry", ("home", "main", "launch", "launcher", "open")),
)
CAPABILITY_LABELS = {
    "search": "搜索结果页并保留查询词",
    "live": "直播页面",
    "social": "关注/消息/动态页面",
    "commerce": "商店/会员/交易相关页面",
    "game": "游戏中心页面",
    "content": "视频/文章/音乐等内容页面",
    "profile": "个人资料/用户空间页面",
    "collection": "收藏/历史/合集页面",
    "settings_system": "系统设置页面",
    "clock_alarm_timer": "闹钟/计时器页面",
    "calendar_event": "日历/日程页面",
    "contacts_people": "联系人/拨号页面",
    "file_document": "文件/文档/媒体页面",
    "browser_web": "浏览器/Web 页面",
    "web_container": "内置 Web 容器页面",
    "publish_upload": "发布/上传入口页面",
    "camera_scan_effect": "相机/扫码/特效页面",
    "poi_location": "地点/附近/位置页面",
    "widget_quick_action": "桌面组件/快捷操作入口",
    "app_entry": "应用首页/入口页面",
    "open_page": "通用页面",
    "other": "其他可打开页面",
}


@dataclass(frozen=True)
class Candidate:
    index: int
    kind: str
    package: str
    description: str
    uri_template: str | None = None
    action: str | None = None
    component: str | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class Variant:
    label: str
    kind: str
    package: str
    uri: str | None = None
    action: str | None = None
    component: str | None = None
    mime_type: str | None = None
    categories: tuple[str, ...] = ()
    extras: tuple[tuple[str, Any], ...] = ()
    flags: tuple[str, ...] = ()


@dataclass
class ProbeResult:
    candidate: Candidate
    variants: list[dict[str, Any]] = field(default_factory=list)
    status: str = "static_only"
    name: str = ""
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    best_variant: dict[str, Any] | None = None
    reason: str = ""
    probe_plan: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProbePlan:
    capability: str
    task: str
    query: str
    candidate_limit: int
    variant_limit: int


@dataclass(frozen=True)
class ShortcutPostprocessItem:
    record: dict[str, Any]
    result: ProbeResult
    plan: ProbePlan


def add_validation_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_cache: bool = True,
    include_execute: bool = True,
) -> None:
    if include_cache:
        parser.add_argument(
            "--cache", type=Path, required=True, help="Path to shortcut_cache/<package>.json"
        )
    if include_execute:
        parser.add_argument(
            "--execute", action="store_true", help="Actually launch resolved variants."
        )
    parser.add_argument(
        "--serial", default="", help="ADB device serial. Defaults to adb's selected device."
    )
    parser.add_argument(
        "--task", default="", help="Optional target task, e.g. '在B站搜索敢杀我的马'."
    )
    parser.add_argument(
        "--query", default="", help="Optional payload text used to test search/text variants."
    )
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-probe-plans", type=int, default=DEFAULT_MAX_PROBE_PLANS)
    parser.add_argument("--max-try", type=int, default=5)
    parser.add_argument(
        "--include-risky", action="store_true", help="Do not skip pay/share/push/auth candidates."
    )
    parser.add_argument("--validation-root", type=Path, default=None)
    parser.add_argument(
        "--promote", action="store_true", help="Promote page-validated shortcuts into skills.py."
    )
    parser.add_argument("--allow-launchable-promote", action="store_true")
    parser.add_argument("--skill-store-root", type=Path, default=DEFAULT_SKILLS_STORE_DIR)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument(
        "--shortcut-postprocess",
        choices=("off", "rules", "llm"),
        default="rules",
        help=(
            "Post-process validated shortcut records before promotion. "
            "'rules' generalizes known probe payloads and merges duplicate pages; "
            "'llm' also asks the verifier LLM to refine names/descriptions."
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_validation_arguments(parser)
    return parser.parse_args(argv)


def load_profile(cache_path: Path) -> AppShortcutProfile:
    return AppShortcutProfile.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))


def validation_root_for(cache_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    parent = cache_path.expanduser().parent
    if parent.name == "shortcut_cache":
        return parent.parent / "shortcut_cache_validation"
    return parent / "shortcut_cache_validation"


def candidate_records(profile: AppShortcutProfile, *, include_risky: bool) -> list[Candidate]:
    records: list[Candidate] = []
    for i, dl in enumerate(profile.deep_links):
        candidate = Candidate(
            index=i,
            kind="deeplink",
            package=profile.package,
            description=dl.description,
            uri_template=dl.uri_template,
            component=dl.component,
        )
        if include_risky or not is_risky_candidate(candidate):
            records.append(candidate)
    offset = len(profile.deep_links)
    for i, di in enumerate(profile.deep_intents):
        candidate = Candidate(
            index=offset + i,
            kind="intent",
            package=profile.package,
            description=di.description,
            action=di.action,
            component=di.component,
            mime_type=di.mime_type,
        )
        if include_risky or not is_risky_candidate(candidate):
            records.append(candidate)
    records.extend(synthetic_candidate_records(profile, records))
    unique_records: dict[tuple[Any, ...], Candidate] = {}
    for candidate in records:
        unique_records.setdefault(candidate_key(candidate), candidate)
    return sorted(unique_records.values(), key=candidate_priority, reverse=True)


def synthetic_candidate_records(
    profile: AppShortcutProfile, existing: list[Candidate]
) -> list[Candidate]:
    if profile.package not in BROWSER_PACKAGES:
        return []
    existing_keys = {candidate_key(candidate) for candidate in existing}
    candidates = [
        Candidate(
            index=-1000,
            kind="deeplink",
            package=profile.package,
            description="Synthetic browser probe URL for ACTION_VIEW",
            uri_template="https://example.com",
        ),
        Candidate(
            index=-999,
            kind="deeplink",
            package=profile.package,
            description="Synthetic browser search URL for ACTION_VIEW",
            uri_template="https://www.google.com/search",
        ),
    ]
    return [candidate for candidate in candidates if candidate_key(candidate) not in existing_keys]


def build_probe_plans(profile: AppShortcutProfile, args: argparse.Namespace) -> list[ProbePlan]:
    candidates = candidate_records(profile, include_risky=args.include_risky)
    if args.task or args.query:
        inferred_capability = infer_primary_capability(candidates)
        query = args.query or default_probe_query(profile.package, inferred_capability)
        return [
            ProbePlan(
                capability="manual",
                task=args.task or default_probe_task(profile.package, inferred_capability, query),
                query=query,
                candidate_limit=args.max_candidates,
                variant_limit=args.max_try,
            )
        ]

    capabilities = infer_probe_capabilities(candidates)
    max_probe_plans = max(1, int(getattr(args, "max_probe_plans", DEFAULT_MAX_PROBE_PLANS)))

    plans: list[ProbePlan] = []
    for capability in capabilities[:max_probe_plans]:
        query = default_probe_query(profile.package, capability)
        plans.append(
            ProbePlan(
                capability=capability,
                task=default_probe_task(profile.package, capability, query),
                query=query,
                candidate_limit=args.max_candidates,
                variant_limit=args.max_try,
            )
        )
    return plans


def infer_primary_capability(candidates: list[Candidate]) -> str:
    if any(looks_query_like(candidate) for candidate in candidates):
        return "search"
    capabilities = infer_probe_capabilities(candidates)
    return capabilities[0] if capabilities else "open_page"


def infer_probe_capabilities(candidates: list[Candidate]) -> list[str]:
    ranked: dict[str, tuple[int, int]] = {}
    for candidate in candidates:
        if is_probe_noise_candidate(candidate):
            continue
        for capability in infer_candidate_capabilities(candidate):
            count, score = ranked.get(capability, (0, -10_000))
            ranked[capability] = (
                count + 1,
                max(score, capability_priority(candidate, capability)),
            )
    if not ranked:
        return ["open_page"]
    return [
        capability
        for capability, _ in sorted(
            ranked.items(),
            key=lambda item: (
                capability_specificity(item[0]),
                item[1][0],
                item[1][1],
                -capability_order(item[0]),
            ),
            reverse=True,
        )
    ]


def infer_candidate_capabilities(candidate: Candidate) -> tuple[str, ...]:
    text = candidate_search_text(candidate)
    capabilities: list[str] = []
    for capability, tokens in CAPABILITY_RULES:
        if matches_any_capability_token(text, tokens):
            capabilities.append(capability)
    if looks_file_document_candidate(candidate):
        capabilities.append("file_document")
    if looks_publish_upload_candidate(candidate):
        capabilities.append("publish_upload")
    if looks_query_like(candidate):
        capabilities = [capability for capability in capabilities if capability == "search"]
    if looks_open_page_candidate(candidate):
        capabilities.append("open_page")
    if not capabilities:
        capabilities.append("other")
    return tuple(dict.fromkeys(capabilities))


def capability_priority(candidate: Candidate, capability: str) -> int:
    score, _ = candidate_priority(candidate)
    route_text = " ".join(
        str(item or "")
        for item in (
            candidate.uri_template,
            candidate.component,
            candidate.description,
        )
    ).lower()
    action = (candidate.action or "").lower()
    tokens = capability_tokens(capability)
    if candidate.kind == "deeplink":
        score += 25
    if candidate.kind == "intent":
        score -= 5
    if matches_any_capability_token(route_text, tokens):
        score += 35
    if matches_any_capability_token(action, tokens):
        score += 15
    if is_generic_intent(candidate) and not matches_any_capability_token(route_text, tokens):
        score -= 40
    if capability in ("other", "open_page"):
        score -= 30
    if capability == "search" and looks_query_like(candidate):
        score += 20
    if looks_query_like(candidate) and capability != "search":
        score -= 80
    if is_probe_noise_candidate(candidate):
        score -= 200
    return score


def capability_tokens(capability: str) -> tuple[str, ...]:
    for name, tokens in CAPABILITY_RULES:
        if name == capability:
            return tokens
    if capability == "open_page":
        return ("view", "open", "detail", "watch", "play", "video", "home", "首页", "详情")
    return ()


def matches_any_capability_token(text: str, tokens: tuple[str, ...]) -> bool:
    return any(matches_capability_token(text, token) for token in tokens)


def matches_capability_token(text: str, token: str) -> bool:
    lowered = text.lower()
    token = token.lower()
    if not token:
        return False
    if not token.isascii() or any(char in token for char in ".:_"):
        return token in lowered
    terms = re.split(r"[^a-z0-9\u4e00-\u9fff]+", lowered)
    return any(term == token or (len(token) >= 5 and term.startswith(token)) for term in terms)


def capability_order(capability: str) -> int:
    preferred = [
        "search",
        "app_entry",
        "settings_system",
        "clock_alarm_timer",
        "calendar_event",
        "contacts_people",
        "live",
        "social",
        "content",
        "profile",
        "collection",
        "file_document",
        "browser_web",
        "web_container",
        "publish_upload",
        "camera_scan_effect",
        "poi_location",
        "widget_quick_action",
        "commerce",
        "game",
        "open_page",
        "other",
    ]
    try:
        return preferred.index(capability)
    except ValueError:
        return len(preferred)


def capability_specificity(capability: str) -> int:
    if capability == "search":
        return 4
    if capability in {
        "browser_web",
        "web_container",
        "publish_upload",
        "camera_scan_effect",
        "poi_location",
        "widget_quick_action",
        "settings_system",
        "clock_alarm_timer",
        "calendar_event",
        "contacts_people",
    }:
        return 3
    if capability == "other":
        return 0
    if capability == "open_page":
        return 1
    return 2


def default_probe_query(package: str, capability: str) -> str:
    if capability != "search":
        return ""
    if package == "tv.danmaku.bili":
        return "敢杀我的马"
    if package == "com.google.android.youtube":
        return "Never Gonna Give You Up"
    return "hello world"


def default_probe_task(package: str, capability: str, query: str) -> str:
    app = display_app_name(package)
    if capability == "search":
        return f"验证 {app} 搜索 deeplink/intent 是否能打开搜索结果页并保留查询词，query={query}"
    label = CAPABILITY_LABELS.get(capability, CAPABILITY_LABELS["open_page"])
    return f"验证 {app} deeplink/intent 是否能打开{label}"


def display_app_name(package: str) -> str:
    names = {
        "tv.danmaku.bili": "B站",
        "com.google.android.youtube": "YouTube",
    }
    return names.get(package, package)


def candidate_matches_plan(candidate: Candidate, plan: ProbePlan) -> bool:
    if plan.capability == "manual":
        return True
    if is_probe_noise_candidate(candidate):
        return False
    if plan.capability == "search":
        return looks_query_like(candidate)
    if looks_query_like(candidate):
        return False
    if plan.capability == "open_page":
        return looks_open_page_candidate(candidate)
    return plan.capability in infer_candidate_capabilities(candidate)


def candidates_for_plan(
    candidates: list[Candidate],
    plan: ProbePlan,
    *,
    skip_candidate_keys: set[tuple[Any, ...]] | None = None,
) -> list[Candidate]:
    skip_candidate_keys = skip_candidate_keys or set()
    matched = [
        candidate
        for candidate in candidates
        if candidate_key(candidate) not in skip_candidate_keys
        and candidate_matches_plan(candidate, plan)
    ]
    return sorted(
        matched,
        key=lambda candidate: candidate_priority_for_plan(candidate, plan),
        reverse=True,
    )[: plan.candidate_limit]


def candidate_priority_for_plan(candidate: Candidate, plan: ProbePlan) -> tuple[int, int]:
    if plan.capability == "manual":
        score, tie = candidate_priority(candidate)
    else:
        score = capability_priority(candidate, plan.capability)
        _, tie = candidate_priority(candidate)
    if candidate.kind == "deeplink":
        score += 10
    elif is_generic_intent(candidate):
        score -= 10
    return score, tie


def looks_open_page_candidate(candidate: Candidate) -> bool:
    return looks_open_page_like(candidate) and not looks_query_like(candidate)


def looks_open_page_like(candidate: Candidate) -> bool:
    text = candidate_search_text(candidate)
    return any(
        token in text
        for token in ("view", "open", "detail", "watch", "play", "video", "home", "首页", "详情")
    )


def candidate_search_text(candidate: Candidate) -> str:
    return " ".join(
        str(item or "")
        for item in (
            candidate.uri_template,
            candidate.action,
            candidate.component,
            candidate.description,
            candidate.mime_type,
        )
    ).lower()


def candidate_key(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.kind,
        candidate.package,
        candidate.uri_template or "",
        candidate.action or "",
        candidate.component or "",
        candidate.mime_type or "",
    )


def is_generic_intent(candidate: Candidate) -> bool:
    action = (candidate.action or "").lower()
    return candidate.kind == "intent" and action in GENERIC_INTENT_ACTIONS


def looks_file_document_candidate(candidate: Candidate) -> bool:
    action = (candidate.action or "").lower()
    mime_type = (candidate.mime_type or "").lower()
    if mime_type and (
        mime_type == "*/*"
        or any(mime_type.startswith(prefix) for prefix in FILE_DOCUMENT_MIME_PREFIXES)
        or "document" in mime_type
    ):
        return True
    if action not in FILE_DOCUMENT_ACTIONS:
        return False
    text = candidate_search_text(candidate)
    return matches_any_capability_token(
        text,
        (
            "file",
            "files",
            "document",
            "documents",
            "open_document",
            "get_content",
            "openable",
            "mime",
            "picker",
        ),
    )


def looks_publish_upload_candidate(candidate: Candidate) -> bool:
    action = (candidate.action or "").lower()
    if action not in PUBLISH_UPLOAD_ACTIONS:
        return False
    if candidate.mime_type:
        return True
    text = candidate_search_text(candidate)
    return any(token in text for token in ("upload", "publish", "post", "compose", "draft"))


def is_probe_noise_candidate(candidate: Candidate) -> bool:
    text = candidate_search_text(candidate)
    if any(token in text for token in PROBE_NOISE_SUBSTRINGS):
        return True
    terms = re.split(r"[^a-z0-9\u4e00-\u9fff]+", text)
    for term in terms:
        if term in PROBE_NOISE_EXACT_TERMS:
            return True
        if any(term.startswith(prefix) for prefix in PROBE_NOISE_PREFIX_TERMS):
            return True
    return False


def probe_plan_to_dict(plan: ProbePlan) -> dict[str, Any]:
    return {
        "capability": plan.capability,
        "task": plan.task,
        "query": plan.query,
        "candidate_limit": plan.candidate_limit,
        "variant_limit": plan.variant_limit,
    }


def is_risky_candidate(candidate: Candidate) -> bool:
    text = candidate_search_text(candidate)
    return any(token in text for token in RISKY_TOKENS)


def candidate_priority(candidate: Candidate) -> tuple[int, int]:
    uri_and_description = f"{candidate.uri_template or ''} {candidate.description or ''}".lower()
    action = (candidate.action or "").lower()
    component = (candidate.component or "").lower()
    score = 0
    if is_probe_noise_candidate(candidate):
        score -= 200
    if candidate.kind == "deeplink" and has_route_word(
        uri_and_description, ("search", "query", "keyword", "搜索")
    ):
        score += 120
    elif action == "android.intent.action.search":
        score += 90
    elif has_route_word(uri_and_description, ("search", "query", "keyword", "搜索")):
        score += 80
    elif "search" in component:
        score += 70
    if has_route_word(uri_and_description, ("browser", "home", "首页")):
        score += 35
    if has_route_word(uri_and_description, ("detail", "详情")):
        score += 20
    if candidate.kind == "deeplink":
        score += 8
    return score, -candidate.index


def has_route_word(text: str, words: tuple[str, ...]) -> bool:
    tokens = set(re.split(r"[^a-z0-9\u4e00-\u9fff]+", text.lower()))
    return any(word in tokens or (not word.isascii() and word in text) for word in words)


def variants_for_candidate(candidate: Candidate, *, query: str, max_try: int) -> list[Variant]:
    if candidate.kind == "deeplink":
        return variants_for_deeplink(candidate, query=query, max_try=max_try)
    return variants_for_intent(candidate, query=query, max_try=max_try)


def variants_for_deeplink(candidate: Candidate, *, query: str, max_try: int) -> list[Variant]:
    uri = candidate.uri_template or ""
    variants: list[Variant] = []

    def add(label: str, value: str, *, component: str | None = None) -> None:
        if not value:
            return
        variant = Variant(
            label=label,
            kind="deeplink",
            package=candidate.package,
            uri=value,
            component=component,
            categories=(BROWSABLE_CATEGORY,),
        )
        if variant not in variants:
            variants.append(variant)

    add("raw_package", uri)
    if candidate.component:
        add("raw_component", uri, component=candidate.component)

    if query and looks_query_like(candidate):
        for key in DEFAULT_QUERY_KEYS:
            add(f"{key}_encoded", uri_with_query(uri, key, quote(query, safe="")))
            add(f"{key}_plus", uri_with_query(uri, key, quote_plus(query)))
            add(f"{key}_raw", uri_with_query(uri, key, query))

    return variants[:max_try]


def variants_for_intent(candidate: Candidate, *, query: str, max_try: int) -> list[Variant]:
    action = candidate.action or ""
    variants: list[Variant] = []

    def add(
        label: str,
        *,
        component: str | None,
        extras: tuple[tuple[str, Any], ...] = (),
        mime_type: str | None = None,
        flags: tuple[str, ...] = (),
    ) -> None:
        if not action:
            return
        variant = Variant(
            label=label,
            kind="intent",
            package=candidate.package,
            action=action,
            component=component,
            mime_type=mime_type if mime_type is not None else candidate.mime_type,
            extras=extras,
            flags=flags,
        )
        if variant not in variants:
            variants.append(variant)

    add("component_no_extra", component=candidate.component)
    add("package_no_extra", component=None)
    if needs_probe_media_payload(candidate):
        stream_extra = ((ANDROID_EXTRA_STREAM, PROBE_UPLOAD_URI),)
        stream_mime_type = candidate.mime_type or "image/*"
        stream_flags = (GRANT_READ_URI_PERMISSION_FLAG,)
        add(
            "component_probe_media_stream",
            component=candidate.component,
            extras=stream_extra,
            mime_type=stream_mime_type,
            flags=stream_flags,
        )
        add(
            "package_probe_media_stream",
            component=None,
            extras=stream_extra,
            mime_type=stream_mime_type,
            flags=stream_flags,
        )
    if query and looks_query_like(candidate):
        for key in DEFAULT_INTENT_EXTRA_KEYS:
            add(f"component_extra_{key}", component=candidate.component, extras=((key, query),))
            add(f"package_extra_{key}", component=None, extras=((key, query),))
    return variants[:max_try]


def looks_query_like(candidate: Candidate) -> bool:
    text = candidate_search_text(candidate)
    return any(
        token in text
        for token in (
            "search",
            "query",
            "keyword",
            "media_search",
            "media_play_from_search",
            "搜索",
        )
    )


def needs_probe_media_payload(candidate: Candidate) -> bool:
    action = (candidate.action or "").lower()
    text = candidate_search_text(candidate)
    return action in PUBLISH_UPLOAD_ACTIONS or any(
        token in text for token in ("upload", "internal_upload")
    )


def uri_with_query(uri: str, key: str, value: str) -> str:
    parsed = urlparse(uri)
    sep = "&" if parsed.query else ""
    query = f"{parsed.query}{sep}{key}={value}"
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment)
    )


class Adb:
    def __init__(self, serial: str = "") -> None:
        self.serial = serial.strip()

    def run(self, *args: str, timeout: float = 10.0) -> tuple[int, str, bool]:
        command = ["adb"]
        if self.serial:
            command.extend(["-s", self.serial])
        command.extend(args)
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
            )
            return completed.returncode, (completed.stdout + completed.stderr).strip(), False
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            if isinstance(output, bytes):
                output = output.decode("utf-8", "ignore")
            return 124, output.strip(), True


def resolve_variant(adb: Adb, variant: Variant) -> dict[str, Any]:
    if variant.kind == "deeplink":
        args = [
            "shell",
            "cmd",
            "package",
            "resolve-activity",
            "--brief",
            "-a",
            VIEW_ACTION,
            "-c",
            BROWSABLE_CATEGORY,
            "-d",
            adb_shell_quote(variant.uri or ""),
        ]
    else:
        args = [
            "shell",
            "cmd",
            "package",
            "resolve-activity",
            "--brief",
            "-a",
            variant.action or "",
        ]
        if variant.mime_type:
            args.extend(["-t", variant.mime_type])
        if variant.component:
            args.extend(["-n", variant.component])
    rc, output, timed_out = adb.run(*args, timeout=5.0)
    ok = is_resolve_ok(output)
    return {
        "ok": ok,
        "rc": rc,
        "timed_out": timed_out,
        "command": args,
        "output": output,
        "target_package": variant.package in output,
    }


def is_resolve_ok(output: str) -> bool:
    lowered = (output or "").lower()
    return bool(
        output.strip()
        and "no activity found" not in lowered
        and "unable to resolve" not in lowered
        and "error:" not in lowered
    )


def launch_variant(
    adb: Adb,
    variant: Variant,
    *,
    artifacts_dir: Path,
    index: int,
    capture_evidence: bool = False,
    settle_seconds: float = 2.0,
) -> dict[str, Any]:
    adb.run("shell", "am", "force-stop", variant.package, timeout=5.0)
    time.sleep(0.4)
    prepare_variant_payload(adb, variant, artifacts_dir)
    args = build_launch_args(variant)
    rc, output, timed_out = adb.run(*args, timeout=15.0)
    time.sleep(settle_seconds)
    foreground = foreground_activity(adb)
    screenshot_path = None
    ui_tree = ""
    ui_sample: list[str] = []
    if capture_evidence:
        screenshot_path = capture_screenshot(adb, artifacts_dir / f"variant_{index:03d}.png")
        ui_tree, ui_sample = capture_ui_tree(adb)
    return {
        "rc": rc,
        "timed_out": timed_out,
        "command": args,
        "output": output,
        "foreground": foreground,
        "screenshot_path": str(screenshot_path) if screenshot_path else None,
        "ui_sample": ui_sample,
        "ui_tree": ui_tree,
        "target_package": foreground.startswith(f"{variant.package}/"),
    }


def build_launch_args(variant: Variant) -> list[str]:
    if variant.kind == "deeplink":
        args = [
            "shell",
            "am",
            "start",
            "-W",
            *variant.flags,
            "-a",
            VIEW_ACTION,
            "-c",
            BROWSABLE_CATEGORY,
            "-d",
            adb_shell_quote(variant.uri or ""),
        ]
    else:
        args = ["shell", "am", "start", "-W", *variant.flags, "-a", variant.action or ""]
        if variant.uri:
            args.extend(["-d", adb_shell_quote(variant.uri)])
        if variant.mime_type:
            args.extend(["-t", variant.mime_type])
        for category in variant.categories:
            args.extend(["-c", category])
        for key, value in variant.extras:
            args.extend(android_extra_args(key, value))
    if variant.component:
        args.extend(["-n", variant.component])
    else:
        args.extend(["-p", variant.package])
    return args


def android_extra_args(key: str, value: Any) -> list[str]:
    text = "" if value is None else str(value)
    if key == ANDROID_EXTRA_STREAM and looks_uri_value(text):
        return ["--eu", key, adb_shell_quote(text)]
    return ["--es", key, adb_shell_quote(text)]


def adb_shell_quote(value: str) -> str:
    return shlex.quote(value)


def looks_uri_value(value: str) -> bool:
    return value.startswith(("content://", "file://", "http://", "https://"))


def prepare_variant_payload(adb: Adb, variant: Variant, artifacts_dir: Path) -> None:
    if not variant_uses_probe_upload_media(variant):
        return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    local_path = artifacts_dir / PROBE_UPLOAD_LOCAL_NAME
    if not local_path.exists():
        local_path.write_bytes(base64.b64decode(PROBE_UPLOAD_PNG_BASE64))
    adb.run("shell", "mkdir", "-p", str(Path(PROBE_UPLOAD_REMOTE_PATH).parent), timeout=5.0)
    adb.run("push", str(local_path), PROBE_UPLOAD_REMOTE_PATH, timeout=10.0)


def variant_uses_probe_upload_media(variant: Variant | dict[str, Any]) -> bool:
    extras: Any
    if isinstance(variant, Variant):
        extras = variant.extras
    else:
        extras = variant.get("extras") or []
    return any(
        len(item) == 2 and item[0] == ANDROID_EXTRA_STREAM and item[1] == PROBE_UPLOAD_URI
        for item in extras
    )


def foreground_activity(adb: Adb) -> str:
    _, output, _ = adb.run("shell", "dumpsys", "window", "windows", timeout=5.0)
    match = re.search(r"mCurrentFocus=Window\{[^ ]+ [^ ]+ ([^/ ]+)/([^}\s]+)", output) or re.search(
        r"mFocusedApp=.* ([^/ ]+)/([^\s}]+)", output
    )
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    _, output, _ = adb.run("shell", "dumpsys", "activity", "activities", timeout=5.0)
    match = re.search(r"topResumedActivity=.* ([^/ ]+)/([^\s}]+)", output)
    return f"{match.group(1)}/{match.group(2)}" if match else ""


def capture_screenshot(adb: Adb, path: Path) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["adb"]
    if adb.serial:
        command.extend(["-s", adb.serial])
    command.extend(["exec-out", "screencap", "-p"])
    try:
        completed = subprocess.run(command, capture_output=True, timeout=8.0, check=False)
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    path.write_bytes(completed.stdout)
    return path


def capture_ui_tree(adb: Adb) -> tuple[str, list[str]]:
    remote = "/sdcard/window_dump.xml"
    adb.run("shell", "rm", "-f", remote, timeout=3.0)
    adb.run("shell", "uiautomator", "dump", "--compressed", remote, timeout=15.0)
    _, xml, _ = adb.run("exec-out", "cat", remote, timeout=8.0)
    values: list[str] = []
    for attr in ("text", "content-desc", "resource-id"):
        values.extend(re.findall(attr + r'="([^"]{1,100})"', xml))
    sample: list[str] = []
    for value in values:
        if value and value not in sample and len(sample) < 40:
            sample.append(value)
    return xml, sample


def verify_with_llm(
    *,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    task: str,
    candidate: Candidate,
    variant: Variant,
    launch: dict[str, Any],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "task": task,
                    "candidate": candidate_to_dict(candidate),
                    "variant": variant_to_dict(variant),
                    "foreground": launch.get("foreground"),
                    "adb_output": summarize_text(str(launch.get("output") or ""), 1200),
                    "ui_sample": launch.get("ui_sample") or [],
                    "ui_tree_excerpt": summarize_text(str(launch.get("ui_tree") or ""), 4000),
                },
                ensure_ascii=False,
            ),
        }
    ]
    screenshot = launch.get("screenshot_path")
    if screenshot:
        image_data = Path(screenshot).read_bytes()
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image_data).decode("ascii")
                },
            }
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You validate Android deeplink/intent probe results. "
                "Return only one JSON object with fields: "
                "usable (boolean), status ('page_validated' or 'launchable' or 'failed'), "
                "name (short English function name, e.g. 'bili_search', 'taobao_cart'; required when usable=true), "
                "description (short natural-language capability, e.g. 'B站搜索视频'), "
                "parameters (array of strings), "
                "payload_preserved (boolean), reason (short string), "
                "next_variant_hint (string or null). "
                "Set usable=true only if the screenshot/UI indicates the intended app page opened "
                "and the payload, such as query text, was preserved when relevant. "
                "When usable=true for the intended page, set status='page_validated'; "
                "use status='launchable' only for target-package launches whose page purpose is unclear. "
                "The `name` field must be a concise English identifier (snake_case, max 30 chars) "
                "derived from the app and page function, e.g. 'bili_scan', 'taobao_search'. "
                "Do not treat a query visible only in search history or suggestions as payload_preserved; "
                "payload_preserved means the current input/result page reflects that payload. "
                "Use a concise natural-language description such as 'B站搜索视频'. "
                "Do not mention adb, URI, component, or implementation details in description or name."
            ),
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except Exception as exc:
        return {
            "usable": False,
            "status": "verifier_error",
            "reason": str(exc)[:200],
        }
    text = response.choices[0].message.content or ""
    return parse_json_object(text)


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        from json_repair import loads as json_repair_loads

        data = json_repair_loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def verifier_enabled(args: argparse.Namespace) -> bool:
    return bool(args.llm_base_url and args.llm_model)


def verifier_api_key(args: argparse.Namespace) -> str:
    return args.llm_api_key or os.environ.get(args.llm_api_key_env, "")


def validate_verifier_config(args: argparse.Namespace) -> None:
    if not verifier_enabled(args) or verifier_api_key(args):
        return
    raise SystemExit(
        "LLM verifier is enabled but no API key was resolved. "
        "Use --llm-api-key-env with the environment variable name, e.g. "
        "--llm-api-key-env DASHSCOPE_API_KEY, or pass --llm-api-key directly."
    )


def args_for_probe_plan(args: argparse.Namespace, plan: ProbePlan) -> argparse.Namespace:
    plan_args = argparse.Namespace(**vars(args))
    plan_args.task = plan.task
    plan_args.query = plan.query
    plan_args.max_try = plan.variant_limit
    return plan_args


def validate_candidate(
    adb: Adb,
    candidate: Candidate,
    *,
    args: argparse.Namespace,
    artifacts_dir: Path,
) -> ProbeResult:
    result = ProbeResult(candidate=candidate, description=candidate.description)
    variants = variants_for_candidate(candidate, query=args.query, max_try=args.max_try)
    hint = ""
    for attempt, variant in enumerate(reorder_variants(variants, hint), start=1):
        variant_record = {
            "attempt": attempt,
            "variant": variant_to_dict(variant),
            "resolve": resolve_variant(adb, variant),
        }
        result.variants.append(variant_record)
        if variant_record["resolve"]["ok"]:
            result.status = "resolved"
        else:
            continue
        if not args.execute:
            continue
        llm_enabled = verifier_enabled(args)
        launch = launch_variant(
            adb,
            variant,
            artifacts_dir=artifacts_dir,
            index=len(result.variants),
            capture_evidence=llm_enabled,
        )
        variant_record["launch"] = launch
        if launch.get("target_package"):
            result.status = "launchable"
            result.best_variant = variant_to_dict(variant)
            if not llm_enabled:
                break
        if llm_enabled:
            verdict = verify_with_llm(
                base_url=args.llm_base_url,
                model=args.llm_model,
                api_key=verifier_api_key(args),
                temperature=args.llm_temperature,
                task=args.task,
                candidate=candidate,
                variant=variant,
                launch=launch,
            )
            variant_record["verifier"] = verdict
            if verdict.get("name"):
                result.name = str(verdict["name"]).strip()
            if verdict.get("description"):
                result.description = str(verdict["description"])
            if isinstance(verdict.get("parameters"), list):
                result.parameters = normalize_parameters(verdict["parameters"], query=args.query)
            hint = str(verdict.get("next_variant_hint") or "")
            if verdict.get("status") == "verifier_error":
                result.status = "verifier_error"
                result.reason = str(verdict.get("reason") or "verifier_error")
                break
            if verdict.get("usable") is True:
                if (
                    args.query
                    and looks_query_like(candidate)
                    and verdict.get("payload_preserved") is not True
                    and not variant_contains_query(variant, args.query)
                ):
                    continue
                normalized_status = normalize_verifier_status(verdict)
                variant_record["normalized_status"] = normalized_status
                result.status = normalized_status
                result.best_variant = variant_to_dict(variant)
                result.reason = str(verdict.get("reason") or "")
                break
    if not result.best_variant and result.variants:
        for variant_record in reversed(result.variants):
            if variant_record.get("resolve", {}).get("ok"):
                result.best_variant = variant_record["variant"]
                break
    return result


def normalize_verifier_status(verdict: dict[str, Any]) -> str:
    status = str(verdict.get("status") or "").strip() or "page_validated"
    if verdict.get("usable") is True and status in {"launchable", "page_validated"}:
        return "page_validated"
    return status


def reorder_variants(variants: list[Variant], hint: str) -> list[Variant]:
    if not hint:
        return variants
    lowered = hint.lower()
    preferred = [variant for variant in variants if lowered in variant.label.lower()]
    rest = [variant for variant in variants if variant not in preferred]
    return [*preferred, *rest]


def plan_satisfied(result: ProbeResult, args: argparse.Namespace) -> bool:
    if result.status == "page_validated":
        return True
    return bool(args.execute and not verifier_enabled(args) and result.status == "launchable")


def result_to_validation_record(result: ProbeResult, *, query: str = "") -> dict[str, Any] | None:
    variant = result.best_variant
    if not variant:
        return None
    variant = template_probe_media_payload(template_query_payload(variant, query=query))
    placeholders = placeholder_names(variant)
    parameters = normalize_parameters(result.parameters, query=query)
    if query and "query" in placeholders and "query" not in parameters:
        parameters.append("query")
    if "media_uri" in placeholders and "media_uri" not in parameters:
        parameters.append("media_uri")
    parameters = [name for name in parameters if name in placeholders]
    record = {
        "package": result.candidate.package,
        "kind": result.candidate.kind,
        "status": result.status,
        "description": result.description,
        "parameters": parameters,
        "valid_state": SHORTCUT_SKIP_VALID_STATE,
    }
    if result.name:
        record["name"] = result.name
    if result.candidate.kind == "deeplink":
        record["uri_template"] = variant.get("uri")
    else:
        record["intent_action"] = variant.get("action")
        if variant.get("uri"):
            record["uri_template"] = variant.get("uri")
        if variant.get("mime_type"):
            record["mime_type"] = variant.get("mime_type")
        if variant.get("categories"):
            record["categories"] = variant.get("categories")
        if variant.get("extras"):
            record["extras"] = variant.get("extras")
    if variant.get("component"):
        record["component"] = variant.get("component")
    return record


def shortcut_skip_valid_state_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    if str(updated.get("kind") or "").strip() in {"deeplink", "intent"}:
        updated["valid_state"] = SHORTCUT_SKIP_VALID_STATE
    return updated


def force_shortcut_skip_valid_state(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [shortcut_skip_valid_state_record(record) for record in records]


def normalize_parameters(values: Any, *, query: str = "") -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if not text or text.startswith(("android.", "com.")):
            continue
        if "=" in text:
            key, raw_value = text.split("=", 1)
            text = key.strip() if not query or raw_value.strip() == query else ""
        name = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
        if name and re.match(r"^[A-Za-z_]\w*$", name) and name not in result:
            result.append(name)
    return result


def template_query_payload(value: Any, *, query: str) -> Any:
    if not query:
        return value
    if isinstance(value, str):
        for item in (quote(query, safe=""), quote_plus(query), query):
            value = value.replace(item, "{{query}}")
        return value
    if isinstance(value, list):
        return [template_query_payload(item, query=query) for item in value]
    if isinstance(value, tuple):
        return tuple(template_query_payload(item, query=query) for item in value)
    if isinstance(value, dict):
        return {key: template_query_payload(item, query=query) for key, item in value.items()}
    return value


def template_probe_media_payload(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(PROBE_UPLOAD_URI, "{{media_uri}}")
    if isinstance(value, list):
        return [template_probe_media_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(template_probe_media_payload(item) for item in value)
    if isinstance(value, dict):
        return {key: template_probe_media_payload(item) for key, item in value.items()}
    return value


def placeholder_names(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(re.findall(r"\{\{(\w+)\}\}", value))
    if isinstance(value, dict):
        names: set[str] = set()
        for item in value.values():
            names.update(placeholder_names(item))
        return names
    if isinstance(value, (list, tuple)):
        names: set[str] = set()
        for item in value:
            names.update(placeholder_names(item))
        return names
    return set()


def variant_contains_query(variant: Variant, query: str) -> bool:
    if not query:
        return False
    return "{{query}}" in json.dumps(
        template_query_payload(variant_to_dict(variant), query=query), ensure_ascii=False
    )


def dedupe_validation_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = validation_record_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def validation_record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("package") or "",
        record.get("kind") or "",
        record.get("uri_template") or "",
        record.get("intent_action") or "",
        record.get("component") or "",
        record.get("mime_type") or "",
        json.dumps(record.get("categories") or [], ensure_ascii=False, sort_keys=True),
        json.dumps(record.get("extras") or [], ensure_ascii=False, sort_keys=True),
    )


async def promote_results(
    args: argparse.Namespace, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not args.promote or not records:
        return []
    store_root = args.skill_store_root or DEFAULT_SKILLS_STORE_DIR
    library = FlatSkillLibrary(store_dir=store_root.expanduser())
    outcomes: list[dict[str, Any]] = []
    for record in force_shortcut_skip_valid_state(records):
        decision, skill_id = await add_validated_shortcut_skill(
            library,
            record,
            require_page_validated=not args.allow_launchable_promote,
        )
        outcomes.append({"decision": decision, "skill_id": skill_id, "record": record})
    return outcomes


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "index": candidate.index,
        "kind": candidate.kind,
        "package": candidate.package,
        "description": candidate.description,
        "uri_template": candidate.uri_template,
        "action": candidate.action,
        "component": candidate.component,
        "mime_type": candidate.mime_type,
    }


def variant_to_dict(variant: Variant) -> dict[str, Any]:
    return {
        "label": variant.label,
        "kind": variant.kind,
        "package": variant.package,
        "uri": variant.uri,
        "action": variant.action,
        "component": variant.component,
        "mime_type": variant.mime_type,
        "categories": list(variant.categories),
        "extras": [list(item) for item in variant.extras],
        "flags": list(variant.flags),
    }


def summarize_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text or "").strip("_") or "default"


def write_sidecar(
    *,
    path: Path,
    profile: AppShortcutProfile,
    args: argparse.Namespace,
    results: list[ProbeResult],
    promotions: list[dict[str, Any]],
    probe_plans: list[ProbePlan],
    stopped_early: str | None = None,
    postprocess_report: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "package": profile.package,
        "device_serial": args.serial,
        "cache_path": str(args.cache),
        "task": args.task,
        "query": args.query,
        "probe_plans": [probe_plan_to_dict(plan) for plan in probe_plans],
        "execute": args.execute,
        "verifier_model": args.llm_model or None,
        "stopped_early": stopped_early,
        "postprocess": postprocess_report or {},
        "created_at": time.time(),
        "results": [
            {
                "candidate": candidate_to_dict(result.candidate),
                "status": result.status,
                "description": result.description,
                "parameters": result.parameters,
                "best_variant": result.best_variant,
                "reason": result.reason,
                "probe_plan": result.probe_plan,
                "variants": result.variants,
                "validation_record": result_to_validation_record(
                    result, query=result_probe_query(result, args)
                ),
            }
            for result in results
        ],
        "promotions": promotions,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def result_probe_query(result: ProbeResult, args: argparse.Namespace) -> str:
    if result.probe_plan:
        return str(result.probe_plan.get("query") or "")
    return args.query


# ---------------------------------------------------------------------------
# Post-processing pipeline
# ---------------------------------------------------------------------------


def _plan_for_result(result: ProbeResult, probe_plans: list[ProbePlan]) -> ProbePlan:
    """Match a result's probe_plan dict back to the original ProbePlan object."""
    plan_record = result.probe_plan or {}
    for plan in probe_plans:
        if probe_plan_to_dict(plan) == plan_record:
            return plan
    return ProbePlan(
        capability=str(plan_record.get("capability") or "manual"),
        task=str(plan_record.get("task") or ""),
        query=str(plan_record.get("query") or ""),
        candidate_limit=int(plan_record.get("candidate_limit") or 1),
        variant_limit=int(plan_record.get("variant_limit") or 1),
    )


def postprocess_validation_records(
    args: argparse.Namespace,
    items: list[ShortcutPostprocessItem],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the post-processing pipeline: generalize → merge → dedupe → optional LLM refine."""
    mode = str(getattr(args, "shortcut_postprocess", "rules") or "rules").strip().lower()
    raw_records = [copy.deepcopy(item.record) for item in items]
    report: dict[str, Any] = {
        "mode": mode,
        "input_count": len(raw_records),
        "rule_changes": [],
        "merged_groups": [],
        "llm": None,
    }
    if mode == "off":
        records = dedupe_validation_records(raw_records)
        records = force_shortcut_skip_valid_state(records)
        report["output_count"] = len(records)
        return records, report

    generalized: list[dict[str, Any]] = []
    for item in items:
        record, changes = generalize_shortcut_record(item.record, item.result, item.plan)
        generalized.append(record)
        report["rule_changes"].extend(changes)

    merged, merge_report = merge_shortcut_records(generalized)
    report["merged_groups"] = merge_report
    records = dedupe_validation_records(merged)

    if mode == "llm":
        records, llm_report = refine_shortcut_records_with_llm(args, records)
        report["llm"] = llm_report

    records = force_shortcut_skip_valid_state(records)
    report["output_count"] = len(records)
    return records, report


def generalize_shortcut_record(
    record: dict[str, Any],
    result: ProbeResult,
    plan: ProbePlan,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply rule-based generalization to a single validation record."""
    updated = copy.deepcopy(record)
    changes: list[dict[str, Any]] = []
    capability = str(getattr(plan, "capability", "") or "")
    package = str(updated.get("package") or "")
    kind = str(updated.get("kind") or "")

    # Chrome browser probe URL → {{url}}
    if (
        package == "com.android.chrome"
        and kind == "deeplink"
        and _is_fixed_browser_probe_url(updated.get("uri_template"), capability)
    ):
        before = copy.deepcopy(updated)
        updated["uri_template"] = "{{url}}"
        updated["parameters"] = _merge_record_parameters(updated, ["url"])
        updated["name"] = "chrome_open_url"
        updated["description"] = "Open an arbitrary full URL in Chrome"
        updated["valid_state"] = SHORTCUT_SKIP_VALID_STATE
        changes.append(_postprocess_change("generalize_browser_url", before, updated))

    # Chrome MEDIA_SEARCH → chrome_open_search_page
    if _is_chrome_media_search_record(updated):
        before = copy.deepcopy(updated)
        updated["name"] = "chrome_open_search_page"
        updated["description"] = "Open Chrome search UI without a prefilled query"
        updated["valid_state"] = SHORTCUT_SKIP_VALID_STATE
        if before != updated:
            changes.append(_postprocess_change("rename_chrome_media_search", before, updated))

    # Deeplink with {{query}} → ensure query parameter
    if kind == "deeplink" and "{{query}}" in str(updated.get("uri_template") or ""):
        before = copy.deepcopy(updated)
        updated["parameters"] = _merge_record_parameters(updated, ["query"])
        if package == "com.android.chrome":
            updated["name"] = "chrome_search"
            updated["description"] = "Search the web in Chrome"
            updated["valid_state"] = SHORTCUT_SKIP_VALID_STATE
        if before != updated:
            changes.append(_postprocess_change("ensure_query_parameter", before, updated))

    # Google Messages SMS compose → generalize recipient + body
    if _is_messages_sms_compose_record(updated):
        before = copy.deepcopy(updated)
        updated["uri_template"] = _generalize_smsto_uri(updated.get("uri_template"))
        updated["extras"] = _replace_or_add_extra(
            updated.get("extras"),
            "sms_body",
            "{{body}}",
        )
        updated["parameters"] = _merge_record_parameters(updated, ["recipient", "body"])
        updated["name"] = "messages_compose_sms"
        updated["description"] = "Open SMS compose page with recipient and message body prefilled"
        updated["valid_state"] = SHORTCUT_SKIP_VALID_STATE
        if before != updated:
            changes.append(_postprocess_change("generalize_sms_body", before, updated))

    # Media picker → generalize mime_type
    if _is_media_picker_record(updated):
        before = copy.deepcopy(updated)
        params = _normalized_result_parameters(result)
        if "mime_type" in params:
            updated["mime_type"] = "{{mime_type}}"
            updated["parameters"] = _merge_record_parameters(updated, ["mime_type"])
            updated["name"] = _media_picker_name(updated)
            updated["description"] = _media_picker_description(updated)
            updated["valid_state"] = SHORTCUT_SKIP_VALID_STATE
        if before != updated:
            changes.append(_postprocess_change("generalize_media_picker_mime", before, updated))

    updated["valid_state"] = SHORTCUT_SKIP_VALID_STATE
    return updated, changes


def merge_shortcut_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group and merge media picker records; pass through everything else."""
    media_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for record in records:
        if _is_media_picker_record(record):
            media_groups.setdefault(_media_picker_merge_key(record), []).append(record)
        else:
            passthrough.append(record)

    merged = list(passthrough)
    report: list[dict[str, Any]] = []
    for key, group in media_groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        merged_record = merge_media_picker_group(group)
        merged.append(merged_record)
        report.append(
            {
                "type": "merge_media_picker",
                "key": list(key),
                "input_count": len(group),
                "input_names": [record.get("name") for record in group],
                "output_name": merged_record.get("name"),
                "output_parameters": merged_record.get("parameters", []),
            }
        )

    return merged, report


def merge_media_picker_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge a group of media picker records that share the same merge key."""
    preferred = sorted(
        group,
        key=lambda record: (
            str(record.get("mime_type") or "") != "image/*",
            str(record.get("name") or ""),
        ),
    )[0]
    merged = copy.deepcopy(preferred)
    mime_types = sorted(
        {
            str(record.get("mime_type") or "").strip()
            for record in group
            if str(record.get("mime_type") or "").strip()
        }
    )
    if len(mime_types) > 1:
        merged["mime_type"] = "{{mime_type}}"
        merged["parameters"] = _merge_record_parameters(merged, ["mime_type"])
    merged["name"] = _media_picker_name(merged)
    merged["description"] = _media_picker_description(merged)
    merged["valid_state"] = SHORTCUT_SKIP_VALID_STATE
    return merged


def refine_shortcut_records_with_llm(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Optionally refine shortcut skill names/descriptions with an LLM pass."""
    report: dict[str, Any] = {"enabled": True, "applied": 0, "error": None}
    if not verifier_enabled(args):
        report["enabled"] = False
        report["error"] = "--shortcut-postprocess llm requires --llm-base-url and --llm-model"
        return records, report
    api_key = verifier_api_key(args)
    if not api_key:
        report["enabled"] = False
        report["error"] = "LLM postprocess requested but API key is missing"
        return records, report

    compact_records = [
        {
            "index": index,
            "package": record.get("package"),
            "kind": record.get("kind"),
            "name": record.get("name"),
            "description": record.get("description"),
            "parameters": record.get("parameters", []),
            "valid_state": SHORTCUT_SKIP_VALID_STATE,
            "uri_template": record.get("uri_template"),
            "intent_action": record.get("intent_action"),
            "component": record.get("component"),
            "mime_type": record.get("mime_type"),
            "extras": record.get("extras"),
        }
        for index, record in enumerate(records)
    ]
    prompt = {
        "task": (
            "Refine Android shortcut skill metadata after validation. "
            'Return JSON only: {"records":[{"index":0,"name":"...",'
            '"description":"..."}]}. '
            "Do not change payload fields, packages, status, parameters, valid_state, or indices. "
            f"All promoted shortcut records must keep valid_state={SHORTCUT_SKIP_VALID_STATE!r}. "
            "Prefer concise distinct names and descriptions that help an agent choose the right shortcut."
        ),
        "records": compact_records,
    }
    try:
        from openai import OpenAI

        client = OpenAI(base_url=args.llm_base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=args.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You refine shortcut skill names/descriptions. Return only JSON.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=args.llm_temperature,
        )
        payload = parse_json_object(response.choices[0].message.content or "")
    except Exception as exc:
        report["error"] = str(exc)[:300]
        return records, report

    proposals = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(proposals, list):
        report["error"] = "LLM response did not include records list"
        return records, report

    updated = [copy.deepcopy(record) for record in records]
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        index = proposal.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(updated):
            continue
        for key in ("name", "description"):
            value = proposal.get(key)
            if isinstance(value, str) and value.strip():
                if key == "name":
                    value = _safe_skill_name(value)
                    if not value:
                        continue
                updated[index][key] = value.strip()
        updated[index]["valid_state"] = SHORTCUT_SKIP_VALID_STATE
        report["applied"] += 1

    return force_shortcut_skip_valid_state(updated), report


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _is_fixed_browser_probe_url(uri: Any, capability: str) -> bool:
    text = str(uri or "").strip()
    if capability != "browser_web":
        return False
    return text in {"https://example.com", "http://example.com", "https://www.example.com"}


def _is_media_picker_record(record: dict[str, Any]) -> bool:
    action = str(record.get("intent_action") or record.get("action") or "")
    component = str(record.get("component") or "")
    mime_type = str(record.get("mime_type") or "")
    return (
        str(record.get("kind") or "") == "intent"
        and action in {"android.intent.action.GET_CONTENT", "android.intent.action.PICK"}
        and bool(component)
        and bool(mime_type)
    )


def _is_chrome_media_search_record(record: dict[str, Any]) -> bool:
    action = str(record.get("intent_action") or record.get("action") or "")
    return (
        str(record.get("package") or "") == "com.android.chrome"
        and str(record.get("kind") or "") == "intent"
        and action == "android.intent.action.MEDIA_SEARCH"
    )


def _is_messages_sms_compose_record(record: dict[str, Any]) -> bool:
    action = str(record.get("intent_action") or record.get("action") or "")
    uri = str(record.get("uri_template") or "")
    return (
        str(record.get("package") or "") == "com.google.android.apps.messaging"
        and str(record.get("kind") or "") == "intent"
        and action == "android.intent.action.SENDTO"
        and (uri.startswith("smsto:") or uri.startswith("sms:"))
    )


def _generalize_smsto_uri(uri_template: Any) -> str:
    uri = str(uri_template or "")
    if uri.startswith("smsto:") or uri.startswith("sms:"):
        scheme = uri.split(":", 1)[0]
        return f"{scheme}:{{{{recipient}}}}"
    return "smsto:{{recipient}}"


def _replace_or_add_extra(extras: Any, key: str, value: Any) -> list[list[Any]]:
    output: list[list[Any]] = []
    replaced = False
    for item in extras or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        current_key = str(item[0])
        if current_key == key:
            output.append([current_key, value])
            replaced = True
        else:
            output.append([item[0], item[1]])
    if not replaced:
        output.append([key, value])
    return output


def _media_picker_merge_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("package") or "",
        record.get("kind") or "",
        record.get("intent_action") or record.get("action") or "",
        record.get("component") or "",
        json.dumps(record.get("categories") or [], ensure_ascii=False, sort_keys=True),
        json.dumps(record.get("extras") or [], ensure_ascii=False, sort_keys=True),
    )


def _media_picker_name(record: dict[str, Any]) -> str:
    package = str(record.get("package") or "")
    if package == "gallery.photomanager.picturegalleryapp.imagegallery":
        return "gallery_pick_media"
    if package == "com.google.android.documentsui":
        return "documents_pick_file"
    return _safe_skill_name(f"{package.rsplit('.', 1)[-1]}_pick_media")


def _media_picker_description(record: dict[str, Any]) -> str:
    package = str(record.get("package") or "")
    if package == "gallery.photomanager.picturegalleryapp.imagegallery":
        return "Open Gallery media picker for a MIME type such as image/* or video/*"
    if package == "com.google.android.documentsui":
        return "Open Android file picker"
    return "Open media picker"


def _merge_record_parameters(record: dict[str, Any], names: list[str]) -> list[str]:
    merged: list[str] = []
    for name in [*record.get("parameters", []), *names]:
        text = str(name).strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def _normalized_result_parameters(result: ProbeResult) -> set[str]:
    output: set[str] = set()
    for name in getattr(result, "parameters", []) or []:
        text = str(name).strip()
        if text:
            output.add(text)
    return output


def _postprocess_change(
    change_type: str, before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": change_type,
        "before": _compact_record_for_report(before),
        "after": _compact_record_for_report(after),
    }


def _compact_record_for_report(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "package": record.get("package"),
        "kind": record.get("kind"),
        "name": record.get("name"),
        "parameters": record.get("parameters", []),
        "uri_template": record.get("uri_template"),
        "intent_action": record.get("intent_action"),
        "component": record.get("component"),
        "mime_type": record.get("mime_type"),
        "extras": record.get("extras", []),
        "description": record.get("description"),
    }


def _safe_skill_name(value: str) -> str:
    text = re.sub(r"\W+", "_", value.strip().lower()).strip("_")
    if not text:
        return ""
    if text[0].isdigit():
        text = f"skill_{text}"
    return text[:60]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_verifier_config(args)
    args.cache = args.cache.expanduser()
    profile = load_profile(args.cache)
    root = validation_root_for(args.cache, args.validation_root)
    serial_label = safe_name(args.serial or "default")
    package_label = safe_name(profile.package)
    artifacts_dir = root / f"{package_label}.{serial_label}.artifacts"
    sidecar_path = root / f"{package_label}.{serial_label}.json"
    adb = Adb(args.serial)

    probe_plans = build_probe_plans(profile, args)
    candidates = candidate_records(profile, include_risky=args.include_risky)
    results: list[ProbeResult] = []
    validated_candidate_keys: set[tuple[Any, ...]] = set()
    stopped_early: str | None = None
    for plan in probe_plans:
        plan_args = args_for_probe_plan(args, plan)
        plan_candidates = candidates_for_plan(
            candidates, plan, skip_candidate_keys=validated_candidate_keys
        )
        for candidate in plan_candidates:
            result = validate_candidate(adb, candidate, args=plan_args, artifacts_dir=artifacts_dir)
            result.probe_plan = probe_plan_to_dict(plan)
            results.append(result)
            if result.status == "verifier_error":
                stopped_early = f"verifier_error: {result.reason}"
                break
            if plan_satisfied(result, plan_args):
                if result.status == "page_validated":
                    validated_candidate_keys.add(candidate_key(candidate))
                break
        if stopped_early:
            break

    validation_items: list[ShortcutPostprocessItem] = []
    for result in results:
        record = result_to_validation_record(result, query=result_probe_query(result, args))
        if record is None:
            continue
        if record["status"] != "page_validated" and not (
            args.allow_launchable_promote and record["status"] == "launchable"
        ):
            continue
        plan = _plan_for_result(result, probe_plans)
        validation_items.append(ShortcutPostprocessItem(record=record, result=result, plan=plan))

    validation_records, postprocess_report = postprocess_validation_records(args, validation_items)
    promotions = asyncio.run(promote_results(args, validation_records))
    write_sidecar(
        path=sidecar_path,
        profile=profile,
        args=args,
        results=results,
        promotions=promotions,
        probe_plans=probe_plans,
        stopped_early=stopped_early,
        postprocess_report=postprocess_report,
    )

    print(f"cache: {args.cache}")
    print(f"package: {profile.package}")
    print(f"candidates_tested: {len(results)}")
    print(
        "probe_plans:",
        json.dumps([probe_plan_to_dict(plan) for plan in probe_plans], ensure_ascii=False),
    )
    print(f"sidecar: {sidecar_path}")
    print("status_counts:", status_counts(results))
    if stopped_early:
        print(f"stopped_early: {stopped_early}")
    if postprocess_report.get("mode") != "off":
        print(
            f"postprocess: input={postprocess_report.get('input_count')} output={postprocess_report.get('output_count')}"
        )
    if promotions:
        print("promotions:", json.dumps(promotions, ensure_ascii=False, indent=2))
    return 0


def status_counts(results: list[ProbeResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
