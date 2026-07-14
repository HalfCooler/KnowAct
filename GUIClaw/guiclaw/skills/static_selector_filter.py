"""
guiclaw.skills.static_selector_filter
=====================================
Shared filters for keeping graph identity tied to static UI chrome instead of
dynamic feed/content snapshots.
"""

from __future__ import annotations

import re
from typing import Any

_STRUCTURAL_TEXTS = {
    "home",
    "profile",
    "settings",
    "search",
    "orders",
    "cart",
    "cancel",
    "ok",
    "done",
    "save",
    "back",
    "close",
    "wi-fi",
    "首页",
    "主页",
    "我",
    "我的",
    "关注",
    "推荐",
    "热点",
    "游戏库",
    "消息",
    "设置",
    "搜索",
    "取消",
    "确认",
    "确定",
    "完成",
    "保存",
    "返回",
    "关闭",
    "黑盒商城",
    "我的订单",
    "全部订单",
    "成功订单",
    "失败订单",
    "购物车",
}

_STRUCTURAL_RESOURCE_RE = re.compile(
    r"(^|_)(nav|tab|tabs|toolbar|appbar|actionbar|menu|btn|button|search|"
    r"digit|digits|key|keys|keypad|keyboard|number|num|"
    r"hotel|htl|query|querybox|keyword|keybox|city|date|filter|"
    r"settings|home|profile|mine|me|order|orders|cart|mall|back|close|"
    r"bottom|top|rb|rg|radio|checkbox|switch|edit|input|wifi|wi_fi|network)($|_)",
    re.IGNORECASE,
)
_DYNAMIC_RESOURCE_RE = re.compile(
    r"(^|_)(post|feed|article|comment|reply|nickname|avatar|price|count|"
    r"content|desc|description|message|cell|item|list|recycler|rank|"
    r"recommend|news|title)($|_)",
    re.IGNORECASE,
)
_SAFE_TITLE_RESOURCE_RE = re.compile(
    r"(^|_)(toolbar|appbar|actionbar|titlebar|page)_?title($|_)",
    re.IGNORECASE,
)
_SAFE_RESULT_RESOURCE_RE = re.compile(
    r"(^|_)((search|hotel|order|orders)_?)?results?_(list|recycler|container|view)($|_)"
    r"|(^|_)(list|recycler|container|view)_results?($|_)",
    re.IGNORECASE,
)

_TIME_RE = re.compile(
    r"(\b\d{1,2}:\d{2}\b|\b\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}|"
    r"\d+\s*(分钟前|小时前|天前|周前|月前|秒前|评论|回复|点赞|浏览|阅读|收藏|粉丝|万|k|K)\b|"
    r"(刚刚|昨天|今日|今天)\s*\d{0,2}:?\d{0,2})"
)
_PRICE_OR_METRIC_RE = re.compile(r"([￥¥$]\s*\d|\d+\s*[%折]|[+-]?\d+(?:\.\d+)?\s*(MB|GB|KB))", re.IGNORECASE)
_SENTENCE_PUNCT_RE = re.compile(r"[，。；：、,.!?！？]")
_GENERIC_MEDIA_TEXTS = {"ad", "ads", "advertisement", "image", "图片", "图像", "广告", "广告素材"}


def static_control_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    text = _clean_text(node.get("text"))
    content_desc = _clean_text(node.get("content_desc"))
    resource_id = _clean_text(node.get("resource_id"))
    clickable = bool(node.get("clickable"))

    resource_static = bool(resource_id and is_static_resource_id(resource_id))
    resource_dynamic = bool(resource_id and is_dynamic_resource_id(resource_id))
    text_static = bool(text and is_static_text(text))
    content_static = bool(content_desc and is_static_text(content_desc))

    if resource_dynamic and not resource_static and not _is_structural_text(text):
        return None
    if not (resource_static or content_static or (text_static and (clickable or not resource_id))):
        return None

    control: dict[str, Any] = {}
    if text_static:
        control["text"] = text
    if content_static:
        control["content_desc"] = content_desc
    if resource_static:
        control["resource_id"] = resource_id
    bounds = node.get("bounds")
    if isinstance(bounds, str) and bounds.strip():
        control["bounds"] = bounds.strip()
    return control or None


def static_selector_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    control = static_control_from_node(node)
    if not control:
        return None
    selector: dict[str, Any] = {}
    if control.get("resource_id"):
        selector["resource_id"] = control["resource_id"]
    elif control.get("content_desc"):
        selector["content_desc"] = control["content_desc"]
    elif control.get("text"):
        selector["text"] = control["text"]
    if selector and node.get("clickable"):
        selector["clickable"] = True
    return selector or None


def selector_is_static(selector: Any) -> bool:
    if not isinstance(selector, dict):
        return False
    resource_id = _clean_text(selector.get("resource_id"))
    content_desc = _clean_text(selector.get("content_desc"))
    text = _clean_text(selector.get("text"))
    xpath = _clean_text(selector.get("xpath"))
    if resource_id:
        return is_static_resource_id(resource_id)
    if content_desc:
        return is_static_text(content_desc)
    if text:
        return is_static_text(text)
    return bool(xpath and not is_dynamic_text(xpath))


def is_static_resource_id(value: Any) -> bool:
    resource_id = _clean_text(value)
    if not resource_id:
        return False
    name = _resource_name(resource_id)
    if _SAFE_TITLE_RESOURCE_RE.search(name) or _SAFE_RESULT_RESOURCE_RE.search(name):
        return True
    if _DYNAMIC_RESOURCE_RE.search(name):
        return False
    return bool(_STRUCTURAL_RESOURCE_RE.search(name))


def is_dynamic_resource_id(value: Any) -> bool:
    resource_id = _clean_text(value)
    if not resource_id:
        return False
    name = _resource_name(resource_id)
    if _SAFE_TITLE_RESOURCE_RE.search(name) or _SAFE_RESULT_RESOURCE_RE.search(name):
        return False
    return bool(_DYNAMIC_RESOURCE_RE.search(name))


def is_static_text(value: Any) -> bool:
    text = _clean_text(value)
    if not text or is_dynamic_text(text):
        return False
    if _is_structural_text(text):
        return True
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_chars:
        return len(text) <= 10 and not _SENTENCE_PUNCT_RE.search(text)
    words = re.findall(r"[A-Za-z0-9]+", text)
    return len(text) <= 28 and len(words) <= 3


def is_dynamic_text(value: Any) -> bool:
    text = _clean_text(value)
    if not text:
        return True
    if text.strip().casefold() in _GENERIC_MEDIA_TEXTS or "广告" in text:
        return True
    if _TIME_RE.search(text) or _PRICE_OR_METRIC_RE.search(text):
        return True
    if sum(ch.isdigit() for ch in text) >= 2:
        return True
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_chars and len(text) > 14:
        return True
    if not cjk_chars and len(text) > 48:
        return True
    if _SENTENCE_PUNCT_RE.search(text) and len(text) > 8:
        return True
    return False


def _is_structural_text(value: Any) -> bool:
    text = _clean_text(value)
    return bool(text and text.casefold() in _STRUCTURAL_TEXTS)


def _resource_name(resource_id: str) -> str:
    name = resource_id.split("/")[-1].split(":")[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _clean_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None
