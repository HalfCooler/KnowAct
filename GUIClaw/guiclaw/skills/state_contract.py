"""
guiclaw.skills.state_contract
=============================
Deterministic helpers for machine-checkable GUI state contracts.

Canonical contracts use the v5 shape:

``anchor``
    Strong identity fields. ``app_package`` is required when known; optional
    ``activity_class`` / ``fragment_class`` are exact-match hints.
``signature``
    Stable UI selectors split into ``required`` and ``forbidden`` elements.
``mask_rules``
    Explicit dynamic-content masks used during extraction/versioning.
``fingerprint``
    ``sha256(canonical_json(normalized_contract_without_fingerprint))``.  It
    is used for dedup/version/cache only, never as the runtime match itself.

Legacy contracts shaped as ``{"app": ..., "must_exist": ...}`` are accepted
and normalized into the canonical schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from guiclaw.skills.static_selector_filter import (
    selector_is_static,
    static_selector_from_node,
)

_SELECTOR_KEYS = frozenset({
    "text",
    "content_desc",
    "resource_id",
    "class",
    "xpath",
})
_STATE_FLAGS = ("visible", "clickable", "enabled", "focused", "scrollable")
_STATE_FLAG_SET = frozenset(_STATE_FLAGS)
_SELECTOR_MATCH_THRESHOLD = 0.72


@dataclass(frozen=True)
class ContractEvalResult:
    """Detailed state-contract evaluation result for trace diagnostics."""

    passed: bool | None
    score: float
    failed_required: list[dict[str, Any]] = field(default_factory=list)
    matched_required: list[dict[str, Any]] = field(default_factory=list)
    unknown_required: list[dict[str, Any]] = field(default_factory=list)
    failed_forbidden: list[dict[str, Any]] = field(default_factory=list)
    unknown_forbidden: list[dict[str, Any]] = field(default_factory=list)
    evidence_coverage: float = 0.0
    reason: str = ""


def normalize_state_contract(contract: Any) -> dict[str, Any] | None:
    """Return a canonical v5 state contract or ``None`` when empty/invalid."""
    if not isinstance(contract, dict):
        return None

    anchor = _normalize_anchor(contract)
    required, forbidden = _normalize_signature(contract)
    mask_rules = _normalize_mask_rules(contract.get("mask_rules"))

    if not anchor and not required and not forbidden:
        return None

    normalized: dict[str, Any] = {
        "anchor": anchor,
        "signature": {
            "required": required,
            "forbidden": forbidden,
        },
        "mask_rules": mask_rules,
    }
    normalized["fingerprint"] = _compute_fingerprint(normalized)
    return normalized


def state_contract_fingerprint(contract: Any) -> str:
    """Return the canonical SHA-256 fingerprint for *contract*.

    The stored ``fingerprint`` field, when present, is ignored and recomputed
    from normalized content.
    """
    normalized = normalize_state_contract(contract)
    if not normalized:
        return ""
    return str(normalized["fingerprint"])


def evaluate_state_contract_detail(
    contract: Any,
    *,
    observation: Any | None = None,
    foreground_app: str | None = None,
    observation_extra: dict[str, Any] | None = None,
    selector_threshold: float = _SELECTOR_MATCH_THRESHOLD,
) -> ContractEvalResult:
    """Evaluate a state contract and return explainable match diagnostics."""
    normalized = normalize_state_contract(contract)
    if not normalized:
        return ContractEvalResult(
            passed=None,
            score=0.0,
            evidence_coverage=0.0,
            reason="invalid_or_empty_contract",
        )

    actual_app = (
        foreground_app
        or getattr(observation, "foreground_app", None)
        or _dict_get(observation, "foreground_app")
        or _dict_get(observation, "app")
    )
    extra = (
        observation_extra
        if observation_extra is not None
        else getattr(observation, "extra", None)
        or _dict_get(observation, "extra")
        or {}
    )
    extra_dict = extra if isinstance(extra, dict) else {}
    index = _UiIndex.from_extra(extra_dict)

    checked: list[float] = []
    matched_required: list[dict[str, Any]] = []
    failed_required: list[dict[str, Any]] = []
    unknown_required: list[dict[str, Any]] = []
    failed_forbidden: list[dict[str, Any]] = []
    unknown_forbidden: list[dict[str, Any]] = []
    total_checks = 0

    anchor = normalized.get("anchor", {})
    expected_app = _clean_string(anchor.get("app_package"))
    if expected_app:
        total_checks += 1
        if not actual_app:
            unknown_required.append({
                "anchor": {"app_package": expected_app},
                "reason": "missing_foreground_app",
            })
        elif _normalize_app(actual_app) != _normalize_app(expected_app):
            failed_required.append({
                "anchor": {"app_package": expected_app},
                "actual": actual_app,
                "reason": "app_package_mismatch",
            })
        else:
            checked.append(1.0)
            matched_required.append({"anchor": {"app_package": expected_app}})

    for key in ("activity_class", "fragment_class"):
        expected = _clean_string(anchor.get(key))
        if not expected:
            continue
        actual = _clean_string(
            _dict_get(extra_dict, key)
            or _dict_get(extra_dict, key.replace("_class", ""))
            or _dict_get(observation, key)
        )
        if actual and _normalize_text(actual) != _normalize_text(expected):
            failed_required.append({
                "anchor": {key: expected},
                "actual": actual,
                "reason": f"{key}_mismatch",
            })
        elif actual:
            checked.append(1.0)
            matched_required.append({"anchor": {key: expected}})

    signature = normalized.get("signature", {})
    for element in signature.get("required", []):
        total_checks += 1
        score = _element_score(element, index)
        if score is None:
            unknown_required.append({
                "element": element,
                "reason": "selector_or_state_evidence_missing",
            })
            continue
        if score < selector_threshold:
            failed_required.append({
                "element": element,
                "score": score,
                "reason": "required_element_mismatch",
            })
            continue
        checked.append(score)
        matched_required.append({"element": element, "score": score})

    for element in signature.get("forbidden", []):
        total_checks += 1
        score = _element_score(element, index)
        if score is None:
            unknown_forbidden.append({
                "element": element,
                "reason": "selector_or_state_evidence_missing",
            })
            continue
        if score >= selector_threshold:
            failed_forbidden.append({
                "element": element,
                "score": score,
                "reason": "forbidden_element_present",
            })
            continue
        checked.append(1.0)

    score_value = sum(checked) / len(checked) if checked else 0.0
    coverage = len(checked) / total_checks if total_checks else 0.0

    if failed_required or failed_forbidden:
        reason = "failed_required" if failed_required else "failed_forbidden"
        return ContractEvalResult(
            passed=False,
            score=0.0,
            failed_required=failed_required,
            matched_required=matched_required,
            unknown_required=unknown_required,
            failed_forbidden=failed_forbidden,
            unknown_forbidden=unknown_forbidden,
            evidence_coverage=coverage,
            reason=reason,
        )
    if unknown_required or unknown_forbidden:
        reason = "unknown_required" if unknown_required else "unknown_forbidden"
        return ContractEvalResult(
            passed=None,
            score=score_value,
            failed_required=failed_required,
            matched_required=matched_required,
            unknown_required=unknown_required,
            failed_forbidden=failed_forbidden,
            unknown_forbidden=unknown_forbidden,
            evidence_coverage=coverage,
            reason=reason,
        )
    if checked:
        return ContractEvalResult(
            passed=True,
            score=score_value,
            failed_required=failed_required,
            matched_required=matched_required,
            unknown_required=unknown_required,
            failed_forbidden=failed_forbidden,
            unknown_forbidden=unknown_forbidden,
            evidence_coverage=coverage,
            reason="matched",
        )
    return ContractEvalResult(
        passed=None,
        score=0.0,
        failed_required=failed_required,
        matched_required=matched_required,
        unknown_required=unknown_required,
        failed_forbidden=failed_forbidden,
        unknown_forbidden=unknown_forbidden,
        evidence_coverage=coverage,
        reason="no_checks",
    )


def evaluate_state_contract(
    contract: Any,
    *,
    observation: Any | None = None,
    foreground_app: str | None = None,
    observation_extra: dict[str, Any] | None = None,
) -> bool | None:
    """Evaluate a state contract against an observation.

    Returns:
        ``True`` when all deterministic checks pass.
        ``False`` when any deterministic check fails.
        ``None`` when there is not enough structured evidence.
    """
    result = evaluate_state_contract_detail(
        contract,
        observation=observation,
        foreground_app=foreground_app,
        observation_extra=observation_extra,
    )
    return result.passed


def infer_interaction_target(action: Any, observation: Any) -> dict[str, Any] | None:
    """Return the pre-action UI node targeted by a pointer action."""
    action_type = _normalize_text(_action_get(action, "action_type"))
    if action_type not in {"tap", "long_press", "double_tap"}:
        return None

    extra = getattr(observation, "extra", None) or _dict_get(observation, "extra") or {}
    if not isinstance(extra, dict) or not isinstance(extra.get("ui_tree"), list):
        return None

    action_payload = {
        "x": _action_get(action, "x"),
        "y": _action_get(action, "y"),
        "relative": bool(_action_get(action, "relative")),
    }
    point_extra = dict(extra)
    point_extra.setdefault("screen_width", getattr(observation, "screen_width", None) or _dict_get(observation, "screen_width"))
    point_extra.setdefault("screen_height", getattr(observation, "screen_height", None) or _dict_get(observation, "screen_height"))
    point = _step_point(action_payload, point_extra)
    if point is None:
        return None

    hits: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for node in extra.get("ui_tree") or []:
        if not isinstance(node, dict):
            continue
        bounds = _parse_bounds(node.get("bounds"))
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        x, y = point
        if not (left <= x <= right and top <= y <= bottom):
            continue
        selector = static_selector_from_node(node)
        if not selector or not selector_is_static(selector):
            continue
        selector = {key: value for key, value in selector.items() if key in _SELECTOR_KEYS}
        if not selector:
            continue
        hits.append((max(1.0, (right - left) * (bottom - top)), node, selector))

    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    min_area = hits[0][0]
    smallest = [(node, selector) for area, node, selector in hits if area == min_area]
    if len({_selector_key(selector) for _, selector in smallest}) != 1:
        return None

    node, selector = smallest[0]
    app = getattr(observation, "foreground_app", None) or _dict_get(observation, "foreground_app") or _dict_get(observation, "app")
    contract = normalize_state_contract({
        "anchor": {"app_package": app} if _clean_string(app) else {},
        "signature": {
            "required": [{"selector": selector, "state": ["visible", "clickable"]}],
            "forbidden": [],
        },
        "mask_rules": [],
    })
    if contract is None:
        return None
    return {
        "selector": selector,
        "bounds": node.get("bounds"),
        "match_method": "coordinate_hit",
        "confidence": 0.9,
        "state_contract": contract,
    }


def infer_focused_input_contract(
    observation_extra: dict[str, Any] | None,
    *,
    app: str | None = None,
) -> dict[str, Any] | None:
    """Infer a focused editable input contract from Android UI-tree metadata."""
    if not isinstance(observation_extra, dict):
        return None
    focused_inputs = [
        node for node in observation_extra.get("ui_tree") or []
        if _is_focused_input_node(node)
    ]
    if len(focused_inputs) != 1:
        return None

    node = focused_inputs[0]
    selector: dict[str, str] = {}
    resource_id = _clean_string(node.get("resource_id"))
    content_desc = _clean_string(node.get("content_desc"))
    class_name = _clean_string(node.get("class"))
    if resource_id:
        selector["resource_id"] = resource_id
    elif content_desc:
        selector["content_desc"] = content_desc
    else:
        return None
    if class_name:
        selector["class"] = class_name

    anchor: dict[str, str] = {}
    clean_app = _clean_string(app)
    if clean_app and clean_app.lower() not in {"unknown", "app_package_or_name"}:
        anchor["app_package"] = clean_app

    return normalize_state_contract({
        "anchor": anchor,
        "signature": {
            "required": [{
                "selector": selector,
                "state": ["visible", "enabled", "focused"],
            }],
            "forbidden": [],
        },
        "mask_rules": [],
    })


def _is_focused_input_node(node: Any) -> bool:
    if not isinstance(node, dict) or not node.get("focused"):
        return False
    class_name = _clean_string(node.get("class")).lower()
    return "edittext" in class_name


def _normalize_anchor(contract: dict[str, Any]) -> dict[str, Any]:
    raw = contract.get("anchor") if isinstance(contract.get("anchor"), dict) else {}
    anchor: dict[str, Any] = {}

    app = (
        raw.get("app_package")
        or raw.get("app")
        or contract.get("app_package")
        or contract.get("app")
    )
    app_text = _clean_string(app)
    if app_text and app_text.lower() not in {"unknown", "app_package_or_name"}:
        anchor["app_package"] = app_text

    activity = (
        raw.get("activity_class")
        or raw.get("activity")
        or contract.get("activity_class")
        or contract.get("activity")
    )
    activity_text = _clean_string(activity)
    if activity_text:
        anchor["activity_class"] = activity_text

    fragment = (
        raw.get("fragment_class")
        or raw.get("fragment")
        or contract.get("fragment_class")
        or contract.get("fragment")
    )
    fragment_text = _clean_string(fragment)
    if fragment_text:
        anchor["fragment_class"] = fragment_text

    return anchor


def _normalize_signature(contract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signature = contract.get("signature") if isinstance(contract.get("signature"), dict) else {}
    required = _normalize_elements(
        signature.get("required")
        or signature.get("required_elements")
        or contract.get("required_elements")
        or contract.get("must_exist")
    )
    forbidden = _normalize_elements(
        signature.get("forbidden")
        or signature.get("forbidden_elements")
        or contract.get("forbidden_elements")
        or contract.get("must_not_exist")
        or contract.get("must_not_exist")
    )
    return _dedupe_elements(required), _dedupe_elements(forbidden)


def _normalize_elements(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        return []

    elements: list[dict[str, Any]] = []
    for item in raw_items:
        element = _normalize_element(item)
        if element:
            elements.append(element)
    return elements


def _normalize_element(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        text = _clean_string(value)
        return {"selector": {"text": text}, "state": ["visible"]} if text else None
    if not isinstance(value, dict):
        return None

    if isinstance(value.get("selector"), dict):
        selector_raw = dict(value["selector"])
        state_raw = value.get("state")
    else:
        selector_raw = dict(value)
        state_raw = value.get("state")

    selector, selector_state = _normalize_selector(selector_raw)
    state = _normalize_state_flags(state_raw)
    if not state:
        state = selector_state
    if selector and "visible" not in state:
        state = [*state, "visible"]

    if not selector and not state:
        return None
    return {"selector": selector, "state": _sort_state_flags(state)}


def _normalize_selector(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    selector: dict[str, Any] = {}
    state: list[str] = []
    for key in _SELECTOR_KEYS:
        text = _clean_string(value.get(key))
        if text:
            selector[key] = text
    for key in _STATE_FLAGS:
        if key in value and bool(value.get(key)):
            state.append(key)
    return selector, _sort_state_flags(state)


def _normalize_state_flags(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = [value] if isinstance(value, str) else value
    if not isinstance(raw_items, list):
        return []
    flags = [
        str(item).strip().lower()
        for item in raw_items
        if str(item).strip().lower() in _STATE_FLAG_SET
    ]
    return _sort_state_flags(flags)


def _sort_state_flags(flags: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(_STATE_FLAGS)}
    return sorted(set(flags), key=lambda item: order.get(item, len(order)))


def _normalize_mask_rules(value: Any) -> list[Any]:
    if value is None:
        return []
    raw_items = [value] if isinstance(value, (str, dict)) else value
    if not isinstance(raw_items, list):
        return []
    out: list[Any] = []
    for item in raw_items:
        if isinstance(item, str):
            text = _clean_string(item)
            if text:
                out.append(text)
        elif isinstance(item, dict):
            cleaned = {
                str(k): _clean_string(v)
                for k, v in sorted(item.items())
                if _clean_string(v)
            }
            if cleaned:
                out.append(cleaned)
    return _dedupe_mask_rules(out)


def _compute_fingerprint(normalized: dict[str, Any]) -> str:
    payload = {
        "anchor": normalized.get("anchor", {}),
        "signature": normalized.get("signature", {"required": [], "forbidden": []}),
        "mask_rules": normalized.get("mask_rules", []),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dedupe_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for element in elements:
        key = _element_key(element)
        if key in seen:
            continue
        out.append(element)
        seen.add(key)
    return sorted(out, key=_element_key)


def _dedupe_mask_rules(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        key = _canonical_json(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return sorted(out, key=_canonical_json)


def _element_key(element: dict[str, Any]) -> str:
    return _canonical_json({
        "selector": element.get("selector", {}),
        "state": element.get("state", []),
    })


def _element_score(element: dict[str, Any], index: "_UiIndex") -> float | None:
    if not index.has_evidence:
        return None

    selector = element.get("selector") if isinstance(element.get("selector"), dict) else {}
    states = set(element.get("state") or [])
    fields = _selector_fields(selector)

    node_score = _selector_node_state_score(fields, states, index)
    if node_score is not None:
        return node_score

    flat_state_score = _selector_flat_state_score(fields, states, index)
    if flat_state_score is not None:
        return flat_state_score

    if any(state in states for state in ("clickable", "focused", "enabled")):
        return None
    if "scrollable" in states and not fields:
        return 1.0 if index.has_scrollable else 0.0

    selector_score = _selector_value_score(selector, index)
    if selector and selector_score is None:
        return None

    scores: list[float] = []
    if selector:
        scores.append(selector_score or 0.0)
    if "visible" in states:
        scores.append(selector_score if selector else 1.0)
    if "scrollable" in states:
        scores.append(1.0 if index.has_scrollable else 0.0)

    return min(scores) if scores else None


def _selector_value_score(selector: dict[str, Any], index: "_UiIndex") -> float | None:
    fields = _selector_fields(selector)
    if not fields:
        return 1.0

    node_score = _selector_node_score(fields, index)
    if node_score is not None:
        return node_score

    scores: list[float] = []
    for key, value in fields:
        score = _selector_field_score(key, value, index)
        if score is None:
            return None
        scores.append(score)
    return min(scores) if scores else 1.0


def _selector_fields(selector: dict[str, Any]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for key in ("resource_id", "content_desc", "text", "class", "xpath"):
        value = _clean_string(selector.get(key))
        if value:
            fields.append((key, value))
    return fields


def _selector_node_state_score(
    fields: list[tuple[str, str]],
    states: set[str],
    index: "_UiIndex",
) -> float | None:
    if not index.ui_nodes:
        return None

    if not fields:
        if "scrollable" in states:
            return 1.0 if index.has_scrollable else 0.0
        return None

    best = 0.0
    saw_state_unknown = False
    saw_state_known = False
    for node in index.ui_nodes:
        selector_score = _score_node_fields(node, fields)
        if selector_score <= 0:
            continue
        state_score = _score_node_states(node, states)
        if state_score is None:
            saw_state_unknown = True
            continue
        saw_state_known = True
        best = max(best, min(selector_score, state_score))

    if best > 0:
        return best
    if saw_state_known:
        return 0.0
    return None if saw_state_unknown else 0.0


def _selector_node_score(fields: list[tuple[str, str]], index: "_UiIndex") -> float | None:
    if not index.ui_nodes:
        return None
    best = 0.0
    for node in index.ui_nodes:
        best = max(best, _score_node_fields(node, fields))
    return best


def _selector_flat_state_score(
    fields: list[tuple[str, str]],
    states: set[str],
    index: "_UiIndex",
) -> float | None:
    if not fields or not states:
        return None
    if any(state == "scrollable" for state in states):
        return None

    selector_score = _selector_value_score_without_nodes(fields, index)
    if selector_score is None or selector_score <= 0:
        return None

    scores = [selector_score]
    for state in states:
        if state == "visible":
            scores.append(selector_score)
        elif state == "clickable":
            state_score = _selector_flat_text_score(fields, index.clickable_text)
            if state_score is None:
                return None
            scores.append(state_score)
        elif state == "focused":
            state_score = _selector_flat_text_score(fields, index.focused_text)
            if state_score is None:
                return None
            scores.append(state_score)
        elif state == "enabled":
            if not index.has_enabled_evidence:
                return None
            state_score = _selector_flat_text_score(fields, index.enabled_text)
            if state_score is None:
                return None
            scores.append(state_score)
    return min(scores)


def _selector_value_score_without_nodes(
    fields: list[tuple[str, str]],
    index: "_UiIndex",
) -> float | None:
    scores: list[float] = []
    for key, value in fields:
        score = _selector_field_score(key, value, index)
        if score is None:
            return None
        scores.append(score)
    return min(scores) if scores else 1.0


def _selector_flat_text_score(fields: list[tuple[str, str]], labels: list[str]) -> float | None:
    if not labels:
        return None
    scores: list[float] = []
    for key, value in fields:
        if key in {"text", "content_desc"}:
            score = _selector_text_score(key, value, labels)
        elif key in {"resource_id", "class", "xpath"}:
            score = _exact_selector_score(value, labels)
        else:
            score = 0.0
        if score <= 0:
            return None
        scores.append(score)
    return min(scores) if scores else None


def _score_node_fields(node: dict[str, Any], fields: list[tuple[str, str]]) -> float:
    scores: list[float] = []
    for key, value in fields:
        score = _selector_text_score(key, value, [_clean_string(node.get(key))])
        if score <= 0:
            return 0.0
        scores.append(score)
    return min(scores) if scores else 1.0


def _score_node_states(node: dict[str, Any], states: set[str]) -> float | None:
    scores: list[float] = []
    for state in states:
        if state == "visible":
            scores.append(1.0)
        elif state == "scrollable":
            if "scrollable" not in node:
                return None
            scores.append(1.0 if _truthy(node.get("scrollable")) else 0.0)
        elif state in {"clickable", "focused", "enabled"}:
            if state not in node:
                return None
            scores.append(1.0 if _truthy(node.get(state)) else 0.0)
    return min(scores) if scores else 1.0


def _selector_field_score(key: str, value: str, index: "_UiIndex") -> float | None:
    if key == "text":
        return _selector_text_score(key, value, index.visible_text + index.all_text)
    if key == "content_desc":
        return _selector_text_score(key, value, index.content_desc + index.all_text)
    if key == "resource_id":
        return _exact_selector_score(value, index.resource_ids)
    if key == "class":
        return _exact_selector_score(value, index.class_names)
    if key == "xpath":
        return _exact_selector_score(value, index.xpath_values)
    return None


def _exact_selector_score(needle: str, haystack: list[str]) -> float:
    needle_norm = _normalize_text(needle)
    if not needle_norm:
        return 0.0
    for item in haystack:
        if _normalize_text(item) == needle_norm:
            return 1.0
    return 0.0


def _selector_text_score(key: str, needle: str, haystack: list[str]) -> float:
    if key in {"resource_id", "xpath", "class"}:
        return _exact_selector_score(needle, haystack)
    if key == "content_desc":
        return _best_text_score(needle, haystack, allow_fuzzy=True)
    if key == "text":
        return _best_text_score(needle, haystack, allow_fuzzy=True)
    return 0.0


def _best_text_score(
    needle: str,
    haystack: list[str],
    *,
    allow_fuzzy: bool = False,
) -> float:
    needle_norm = _normalize_text(needle)
    if not needle_norm:
        return 0.0
    best = 0.0
    for item in haystack:
        item_norm = _normalize_text(item)
        if not item_norm:
            continue
        if needle_norm == item_norm:
            return 1.0
    if not allow_fuzzy or len(needle_norm) < 4:
        return 0.0

    needle_tokens = set(re.findall(r"\w+", needle_norm))
    for item in haystack:
        item_norm = _normalize_text(item)
        if not item_norm:
            continue
        if needle_norm in item_norm or item_norm in needle_norm:
            if min(len(needle_norm), len(item_norm)) >= 4:
                best = max(best, 0.92)
            continue
        item_tokens = set(re.findall(r"\w+", item_norm))
        if needle_tokens and item_tokens:
            overlap = len(needle_tokens & item_tokens) / len(needle_tokens | item_tokens)
            if overlap:
                best = max(best, 0.55 + 0.35 * overlap)
    return best


def _step_point(step_payload: dict[str, Any], extra: dict[str, Any]) -> tuple[float, float] | None:
    params = step_payload.get("parameters")
    if not isinstance(params, dict):
        params = step_payload
    x = _float_value(params.get("x"))
    y = _float_value(params.get("y"))
    if x is None or y is None:
        return None
    if params.get("relative"):
        width = _float_value(extra.get("screen_width"))
        height = _float_value(extra.get("screen_height"))
        if width is None or height is None:
            return None
        ui_tree = extra.get("ui_tree")
        if isinstance(ui_tree, list) and ui_tree:
            tree_width: float | None = None
            tree_height: float | None = None
            for node in ui_tree:
                if not isinstance(node, dict):
                    continue
                bounds = _parse_bounds(node.get("bounds"))
                if bounds is None:
                    continue
                tree_width = max(tree_width or 0.0, bounds[2])
                tree_height = max(tree_height or 0.0, bounds[3])
            if tree_width and tree_height:
                if tree_width > width * 1.2:
                    width = tree_width
                if tree_height > height * 1.2:
                    height = tree_height
        x = x / 1000.0 * width
        y = y / 1000.0 * height
    return x, y


def _parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        coords = [_float_value(item) for item in value]
        if all(coord is not None for coord in coords):
            left = float(coords[0])
            top = float(coords[1])
            right = float(coords[2])
            bottom = float(coords[3])
            return left, top, right, bottom
        return None
    text = _clean_string(value)
    match = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", text)
    if not match:
        return None
    left, top, right, bottom = (float(item) for item in match.groups())
    return left, top, right, bottom


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class _UiIndex:
    visible_text: list[str] = field(default_factory=list)
    content_desc: list[str] = field(default_factory=list)
    resource_ids: list[str] = field(default_factory=list)
    clickable_text: list[str] = field(default_factory=list)
    focused_text: list[str] = field(default_factory=list)
    enabled_text: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    xpath_values: list[str] = field(default_factory=list)
    ui_nodes: list[dict[str, Any]] = field(default_factory=list)
    all_text: list[str] = field(default_factory=list)
    has_scrollable: bool = False
    has_enabled_evidence: bool = False
    has_evidence: bool = False

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> "_UiIndex":
        index = cls(
            visible_text=_string_list(extra.get("visible_text")),
            content_desc=_string_list(extra.get("content_desc")),
            resource_ids=_string_list(extra.get("resource_ids")),
            clickable_text=_string_list(extra.get("clickable_text")),
            focused_text=_string_list(extra.get("focused_text")),
            enabled_text=_string_list(extra.get("enabled_text")),
            class_names=_string_list(extra.get("class_names")),
            xpath_values=_string_list(extra.get("xpaths")),
            has_scrollable=_truthy(extra.get("scrollable_present")),
            has_enabled_evidence=(
                "enabled_text" in extra
                or "enabled_present" in extra
            ),
        )
        for node in extra.get("ui_tree") or []:
            if not isinstance(node, dict):
                if isinstance(node, str):
                    index.all_text.append(node)
                continue
            text = _clean_string(node.get("text"))
            content_desc = _clean_string(node.get("content_desc"))
            resource_id = _clean_string(node.get("resource_id"))
            class_name = _clean_string(node.get("class"))
            xpath = _clean_string(node.get("xpath"))
            label = text or content_desc or resource_id
            node_labels = _dedupe(
                value
                for value in (text, content_desc, resource_id, class_name, xpath)
                if value
            )
            ui_node = {
                key: value
                for key, value in (
                    ("text", text),
                    ("content_desc", content_desc),
                    ("resource_id", resource_id),
                    ("class", class_name),
                    ("xpath", xpath),
                )
                if value
            }
            for state_key in ("clickable", "focused", "enabled", "scrollable"):
                if state_key in node:
                    ui_node[state_key] = bool(node.get(state_key))
            if ui_node:
                index.ui_nodes.append(ui_node)
            if text:
                index.visible_text.append(text)
            if content_desc:
                index.content_desc.append(content_desc)
            if resource_id:
                index.resource_ids.append(resource_id)
            if class_name:
                index.class_names.append(class_name)
            if xpath:
                index.xpath_values.append(xpath)
            if node.get("clickable"):
                index.clickable_text.extend(node_labels or ([label] if label else []))
            if node.get("focused"):
                index.focused_text.extend(node_labels or ([label] if label else []))
            if "enabled" in node:
                index.has_enabled_evidence = True
                if node.get("enabled"):
                    index.enabled_text.extend(node_labels or ([label] if label else []))
            if node.get("scrollable"):
                index.has_scrollable = True

        index.visible_text = _dedupe(index.visible_text)
        index.content_desc = _dedupe(index.content_desc)
        index.resource_ids = _dedupe(index.resource_ids)
        index.clickable_text = _dedupe(index.clickable_text)
        index.focused_text = _dedupe(index.focused_text)
        index.enabled_text = _dedupe(index.enabled_text)
        index.class_names = _dedupe(index.class_names)
        index.xpath_values = _dedupe(index.xpath_values)
        index.ui_nodes = _dedupe_node_dicts(index.ui_nodes)
        index.all_text = _dedupe(
            index.all_text
            + index.visible_text
            + index.content_desc
            + index.resource_ids
            + index.clickable_text
            + index.focused_text
            + index.enabled_text
            + index.class_names
            + index.xpath_values
        )
        index.has_evidence = bool(
            index.all_text
            or extra.get("ui_tree_node_count")
            or index.has_scrollable
        )
        return index


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    return _dedupe(text for item in items if (text := _clean_string(item)))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _normalize_text(value: Any) -> str:
    return _clean_string(value).casefold()


def _normalize_app(value: Any) -> str:
    return _normalize_text(value)


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_string(value)
        if not text:
            continue
        marker = text.casefold()
        if marker in seen:
            continue
        out.append(text)
        seen.add(marker)
    return out


def _dedupe_node_dicts(values: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        item: dict[str, Any] = {}
        for key, raw in value.items():
            if key in _STATE_FLAG_SET:
                item[str(key)] = bool(raw)
                continue
            text = _clean_string(raw)
            if text:
                item[str(key)] = text
        marker = _canonical_json(item)
        if not item or marker in seen:
            continue
        out.append(item)
        seen.add(marker)
    return out


def _selector_key(selector: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(selector.items()))


def _dict_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _action_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
