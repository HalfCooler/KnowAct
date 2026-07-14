"""Static Android shortcut extraction from AndroidManifest.xml only."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from guiclaw.skills.data import collect_placeholder_names

_ANDROID_NS = "http://schemas.android.com/apk/res/android"


@dataclass(frozen=True)
class ManifestIntentFilter:
    component: str
    actions: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    schemes: tuple[str, ...] = ()
    authorities: tuple[str, ...] = ()
    paths: tuple[tuple[str, str], ...] = ()
    mime_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeepLink:
    uri_template: str
    scheme: str
    host: str | None
    path: str | None
    component: str
    description: str
    path_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri_template": self.uri_template,
            "scheme": self.scheme,
            "host": self.host,
            "path": self.path,
            "component": self.component,
            "description": self.description,
            "path_kind": self.path_kind,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeepLink":
        return cls(
            uri_template=d["uri_template"],
            scheme=d["scheme"],
            host=d.get("host"),
            path=d.get("path"),
            component=d["component"],
            description=d.get("description", ""),
            path_kind=d.get("path_kind"),
        )


@dataclass(frozen=True)
class DeepIntent:
    action: str
    component: str
    mime_type: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "component": self.component,
            "mime_type": self.mime_type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeepIntent":
        return cls(
            action=d["action"],
            component=d["component"],
            mime_type=d.get("mime_type"),
            description=d.get("description", ""),
        )


@dataclass(frozen=True)
class AppShortcutProfile:
    package: str
    deep_links: tuple[DeepLink, ...] = ()
    deep_intents: tuple[DeepIntent, ...] = ()
    activity_aliases: tuple[tuple[str, str], ...] = ()
    manifest_meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "deep_links": [dl.to_dict() for dl in self.deep_links],
            "deep_intents": [di.to_dict() for di in self.deep_intents],
            "activity_aliases": list(self.activity_aliases),
            "manifest_meta": self.manifest_meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AppShortcutProfile":
        return cls(
            package=d["package"],
            deep_links=tuple(DeepLink.from_dict(x) for x in d.get("deep_links", [])),
            deep_intents=tuple(DeepIntent.from_dict(x) for x in d.get("deep_intents", [])),
            activity_aliases=tuple((a, t) for a, t in d.get("activity_aliases", [])),
            manifest_meta=d.get("manifest_meta"),
        )


@dataclass(frozen=True)
class ValidatedShortcut:
    package: str
    kind: str
    status: str
    description: str
    name: str | None = None
    uri_template: str | None = None
    action: str | None = None
    component: str | None = None
    mime_type: str | None = None
    categories: tuple[str, ...] = ()
    extras: tuple[tuple[str, Any], ...] = ()
    parameters: tuple[str, ...] = ()
    valid_state: str | None = None
    skill_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidatedShortcut":
        return cls(
            package=_clean_app(data.get("package")),
            kind=str(data.get("kind") or data.get("type") or "").strip(),
            status=str(data.get("status") or "").strip(),
            description=str(
                data.get("intent_summary") or data.get("description") or data.get("summary") or ""
            ).strip(),
            name=str(data["name"]).strip() if data.get("name") else None,
            uri_template=str(
                data.get("uri_template") or data.get("uri") or data.get("text") or ""
            ).strip()
            or None,
            action=str(data.get("intent_action") or data.get("action") or "").strip() or None,
            component=str(data.get("component") or "").strip() or None,
            mime_type=str(data.get("mime_type") or "").strip() or None,
            categories=_normalize_str_tuple(data.get("categories", ())),
            extras=_normalize_extras(data.get("extras", ())),
            parameters=_normalize_str_tuple(data.get("parameters", ())),
            valid_state=str(data.get("valid_state") or "").strip() or None,
            skill_id=str(data.get("skill_id") or "").strip() or None,
        )


async def extract_app_shortcuts(backend: Any, package: str) -> AppShortcutProfile:
    app = _clean_app(package)
    apk_path = await _pull_apk(backend, app)
    if apk_path is None:
        return AppShortcutProfile(package=app)

    manifest_root = _parse_manifest(apk_path)
    Path(apk_path).unlink(missing_ok=True)
    manifest_package = _clean_text(manifest_root.get("package")) or app
    filters = _extract_all_filters(manifest_root, manifest_package)
    return _profile_from_manifest(
        manifest_root,
        manifest_package,
        manifest_meta={"apk_path": apk_path, "filter_count": len(filters)},
        filters=filters,
    )


def extract_shortcuts_from_manifest(manifest_path: Path | str) -> AppShortcutProfile:
    """Extract static shortcut candidates from a decoded AndroidManifest.xml file."""
    path = Path(manifest_path).expanduser()
    manifest_root = ET.parse(path).getroot()
    package = _clean_text(manifest_root.get("package"))
    if not package:
        raise ValueError(f"manifest has no package: {path}")
    filters = _extract_all_filters(manifest_root, package)
    return _profile_from_manifest(
        manifest_root,
        package,
        manifest_meta={"manifest_path": str(path), "filter_count": len(filters)},
        filters=filters,
    )


def _profile_from_manifest(
    manifest_root: ET.Element,
    package: str,
    *,
    manifest_meta: dict[str, Any],
    filters: list[ManifestIntentFilter] | None = None,
) -> AppShortcutProfile:
    filters = filters if filters is not None else _extract_all_filters(manifest_root, package)

    aliases: list[tuple[str, str]] = []
    for alias in _find_elements(manifest_root, "activity-alias"):
        alias_component = _normalize_component_name(package, _android_attr(alias, "name"))
        target_component = _normalize_component_name(
            package, _android_attr(alias, "targetActivity")
        )
        if alias_component and target_component:
            aliases.append((alias_component, target_component))

    return AppShortcutProfile(
        package=package,
        deep_links=tuple(_classify_deep_links(filters)),
        deep_intents=tuple(_classify_deep_intents(filters)),
        activity_aliases=tuple(dict.fromkeys(aliases)),
        manifest_meta=manifest_meta,
    )


def validated_shortcut_to_skill(
    shortcut: ValidatedShortcut | dict[str, Any],
    *,
    require_page_validated: bool = True,
) -> Any | None:
    """Convert a validated deeplink/intent record into a one-step flat Skill."""
    from guiclaw.skills.data import Skill, SkillStep

    record = ValidatedShortcut.from_dict(shortcut) if isinstance(shortcut, dict) else shortcut
    if require_page_validated and record.status != "page_validated":
        return None
    if not require_page_validated and record.status not in {"page_validated", "launchable"}:
        return None

    package = _clean_app(record.package)
    if not package:
        raise ValueError("validated shortcut requires package")

    kind = record.kind.strip().lower()
    description = _clean_shortcut_description(record.description, package)
    name = record.name or _validated_skill_name(package, kind, description)
    tags = ("shortcut", "deeplink" if kind == "deeplink" else "intent", "validated")

    if kind == "deeplink":
        uri = record.uri_template or ""
        if not uri:
            raise ValueError("validated deeplink shortcut requires uri_template")
        parameters: dict[str, Any] = {"text": uri, "package": package}
        if record.component:
            parameters["component"] = record.component
        step = SkillStep(
            action_type="open_deeplink",
            target=uri,
            parameters=parameters,
            valid_state=record.valid_state,
        )
        raw = {
            "package": package,
            "kind": kind,
            "uri_template": uri,
            "component": record.component or "",
        }
        skill_id = (
            record.skill_id or f"shortcut:dl:{package}:{_stable_short_hash(_stable_json(raw))}"
        )
    elif kind == "intent":
        action = record.action or ""
        if not action:
            raise ValueError("validated intent shortcut requires action")
        parameters = {"intent_action": action, "package": package}
        if record.uri_template:
            parameters["text"] = record.uri_template
        if record.component:
            parameters["component"] = record.component
        if record.mime_type:
            parameters["mime_type"] = record.mime_type
        if record.categories:
            parameters["categories"] = record.categories
        if record.extras:
            parameters["extras"] = record.extras
        step = SkillStep(
            action_type="open_intent",
            target=action,
            parameters=parameters,
            valid_state=record.valid_state,
        )
        raw = {
            "package": package,
            "kind": kind,
            "action": action,
            "uri_template": record.uri_template or "",
            "component": record.component or "",
            "mime_type": record.mime_type or "",
            "categories": record.categories,
            "extras": record.extras,
        }
        skill_id = (
            record.skill_id or f"shortcut:di:{package}:{_stable_short_hash(_stable_json(raw))}"
        )
    else:
        raise ValueError(f"unsupported validated shortcut kind: {record.kind!r}")

    return Skill(
        skill_id=skill_id,
        name=name,
        description=description,
        app=package,
        platform="android",
        steps=(step,),
        parameters=_shortcut_parameters(record, step),
        tags=tags,
        success_count=1,
    )


async def add_validated_shortcut_skill(
    library: Any,
    shortcut: ValidatedShortcut | dict[str, Any],
    *,
    require_page_validated: bool = True,
) -> tuple[str, str | None]:
    skill_obj = validated_shortcut_to_skill(
        shortcut,
        require_page_validated=require_page_validated,
    )
    if skill_obj is None:
        return "SKIP_UNVALIDATED", None
    return await library.add_or_merge(skill_obj)


async def _pull_apk(backend: Any, package: str) -> str | None:
    output = await backend._run("shell", "pm", "path", package, timeout=10.0)
    apk_lines = [
        line.removeprefix("package:").strip()
        for line in str(output).splitlines()
        if line.strip().startswith("package:")
    ]
    if not apk_lines:
        return None

    safe_package = re.sub(r"[^A-Za-z0-9_.-]+", "_", package)
    with tempfile.NamedTemporaryFile(
        suffix=".apk", prefix=f"guiclaw-{safe_package}-", delete=False
    ) as fd:
        local_path = fd.name
    await backend._run("pull", apk_lines[0], str(local_path), timeout=30.0)
    return local_path


def _parse_manifest(apk_path: str) -> ET.Element:
    from pyaxmlparser import APK

    manifest = APK(apk_path).get_android_manifest_xml()
    if hasattr(manifest, "tag"):
        return manifest
    if isinstance(manifest, bytes):
        return ET.fromstring(manifest)
    return ET.fromstring(str(manifest))


def _extract_all_filters(manifest_root: ET.Element, package: str) -> list[ManifestIntentFilter]:
    filters: list[ManifestIntentFilter] = []
    component_tags = ("activity", "activity-alias")

    for component_element in manifest_root.iter():
        if _local_tag(component_element.tag) not in component_tags:
            continue
        component = _normalize_component_name(package, _android_attr(component_element, "name"))
        if not component:
            continue

        for intent_filter in component_element:
            if _local_tag(intent_filter.tag) != "intent-filter":
                continue
            actions: list[str] = []
            categories: list[str] = []
            schemes: list[str] = []
            authorities: list[str] = []
            paths: list[tuple[str, str]] = []
            mime_types: list[str] = []

            for child in intent_filter:
                tag = _local_tag(child.tag)
                if tag == "action":
                    actions.append(_android_attr(child, "name"))
                elif tag == "category":
                    categories.append(_android_attr(child, "name"))
                elif tag == "data":
                    scheme = _android_attr(child, "scheme")
                    host = _android_attr(child, "host")
                    port = _android_attr(child, "port")
                    if scheme:
                        schemes.append(scheme)
                    if host and port:
                        authorities.append(f"{host}:{port}")
                    elif host:
                        authorities.append(host)
                    mime_type = _android_attr(child, "mimeType")
                    if mime_type:
                        mime_types.append(mime_type)
                    for path_attr in (
                        "path",
                        "pathPrefix",
                        "pathPattern",
                        "pathAdvancedPattern",
                        "pathSuffix",
                    ):
                        path = _android_attr(child, path_attr)
                        if path:
                            paths.append((path_attr, path))

            filters.append(
                ManifestIntentFilter(
                    component=component,
                    actions=tuple(_dedupe(actions)),
                    categories=tuple(_dedupe(categories)),
                    schemes=tuple(_dedupe(schemes)),
                    authorities=tuple(_dedupe(authorities)),
                    paths=tuple(_dedupe_path_specs(paths)),
                    mime_types=tuple(_dedupe(mime_types)),
                )
            )

    return filters


def _classify_deep_links(filters: list[ManifestIntentFilter]) -> list[DeepLink]:
    results: list[DeepLink] = []
    seen: set[tuple[str, str | None, str | None, str | None, str]] = set()

    for item in filters:
        if "android.intent.action.VIEW" not in item.actions:
            continue
        if "android.intent.category.BROWSABLE" not in item.categories:
            continue
        for scheme in item.schemes:
            if scheme.casefold() in {"http", "https"}:
                continue
            hosts = item.authorities or ("",)
            if item.authorities:
                paths = item.paths or (("", ""),)
            else:
                paths = (("", ""),)
            for host in hosts:
                host_value = host or None
                for path_kind, path in paths:
                    if path_kind not in {"", "path", "pathPrefix"}:
                        continue
                    path_value = path or None
                    if host_value and path_value:
                        path_part = path if path.startswith("/") else f"/{path}"
                        uri_template = f"{scheme}://{host_value}{path_part}"
                    elif host_value:
                        uri_template = f"{scheme}://{host_value}"
                    else:
                        uri_template = f"{scheme}:"
                    if not _usable_data_uri(uri_template):
                        continue
                    key = (scheme, host_value, path_value, path_kind or None, item.component)
                    if key in seen:
                        continue
                    seen.add(key)
                    kind_label = f" with {path_kind}={path_value}" if path_kind else ""
                    results.append(
                        DeepLink(
                            uri_template=uri_template,
                            scheme=scheme,
                            host=host_value,
                            path=path_value,
                            component=item.component,
                            description=(
                                f"Static Android deep link candidate for {uri_template}{kind_label}; "
                                f"component={item.component}; not page-validated."
                            ),
                            path_kind=path_kind or None,
                        )
                    )

    return results


def _classify_deep_intents(filters: list[ManifestIntentFilter]) -> list[DeepIntent]:
    results: list[DeepIntent] = []
    seen: set[tuple[str, str, str | None]] = set()

    for item in filters:
        for action in item.actions:
            if action in {"android.intent.action.MAIN", "android.intent.action.VIEW"}:
                continue
            mime_types: tuple[str | None, ...] = item.mime_types or (None,)
            for mime_type in mime_types:
                key = (action, item.component, mime_type)
                if key in seen:
                    continue
                seen.add(key)
                mime_label = f"; mime_type={mime_type}" if mime_type else ""
                results.append(
                    DeepIntent(
                        action=action,
                        component=item.component,
                        mime_type=mime_type,
                        description=(
                            f"Static Android intent candidate for action={action}{mime_label}; "
                            f"component={item.component}; not page-validated."
                        ),
                    )
                )

    return results


def _local_tag(tag: str) -> str:
    text = str(tag)
    if "}" in text:
        return text.rsplit("}", 1)[1]
    if ":" in text:
        return text.split(":", 1)[1]
    return text


def _android_attr(element: ET.Element, name: str) -> str:
    value = element.get(f"{{{_ANDROID_NS}}}{name}")
    if value is None:
        value = element.get(f"android:{name}")
    if value is None:
        value = element.get(name)
    return _clean_text(value)


def _normalize_component_name(package: str, name: str) -> str:
    app = _clean_app(package)
    text = _clean_text(name)
    if not app or not text:
        return ""
    if text.startswith("."):
        text = f"{app}{text}"
    elif "." not in text:
        text = f"{app}.{text}"
    return f"{app}/{text}"


def _find_elements(root: ET.Element, tag: str) -> list[ET.Element]:
    return [element for element in root.iter() if _local_tag(element.tag) == tag]


def _dedupe(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _dedupe_path_specs(
    items: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for kind, value in items:
        cleaned_kind = str(kind).strip()
        cleaned_value = str(value).strip()
        key = (cleaned_kind, cleaned_value)
        if cleaned_kind and cleaned_value and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _normalize_extras(raw: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(raw, dict):
        iterable = raw.items()
    else:
        iterable = raw or ()
    extras: list[tuple[str, Any]] = []
    for item in iterable:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        key, value = item
        key_text = str(key).strip()
        if key_text:
            extras.append((key_text, value))
    return tuple(extras)


def _normalize_str_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        raw = (raw,)
    return tuple(str(item).strip() for item in (raw or ()) if str(item).strip())


def _clean_shortcut_description(description: str, package: str) -> str:
    text = re.sub(r"\s+", " ", _clean_text(description)).strip()
    lowered = text.lower()
    if (
        not text
        or "://" in text
        or "component=" in lowered
        or "android.intent." in lowered
        or "am start" in lowered
        or "adb " in lowered
    ):
        return f"{package} shortcut"
    return text[:120]


def _validated_skill_name(package: str, kind: str, description: str) -> str:
    package_tail = package.rsplit(".", 1)[-1]
    slug = _slug(f"{package_tail}_{description}", max_len=60)
    if slug == "shortcut":
        slug = _slug(f"{package_tail}_{kind}", max_len=60)
    return slug


def _shortcut_parameters(record: ValidatedShortcut, step: Any) -> tuple[str, ...]:
    names = set(record.parameters)
    names.update(collect_placeholder_names(step.target))
    names.update(collect_placeholder_names(step.parameters))
    return tuple(sorted(name for name in names if name))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _slug(value: str, *, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return (text or "shortcut")[:max_len]


def _clean_app(value: Any) -> str:
    return _clean_text(value)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _usable_data_uri(uri: str) -> bool:
    try:
        return bool(urlparse(uri).scheme)
    except Exception:
        return False
