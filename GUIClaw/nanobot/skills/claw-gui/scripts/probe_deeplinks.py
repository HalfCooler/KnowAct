#!/usr/bin/env python3
"""Generate and optionally execute Android deep-link probes."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

TASK_PROFILES = {
    "search": {
        "paths": ["search", "search/", "search/result", "search/results", "find"],
        "query_keys": ["keyword", "query", "q", "text"],
        "query_value": "test",
    },
    "detail": {
        "paths": ["detail/123", "item/123", "note/123", "post/123"],
        "query_keys": ["id", "item_id", "note_id"],
        "query_value": "123",
    },
    "profile": {
        "paths": ["profile/test", "user/test", "u/test", "member/test"],
        "query_keys": ["user_id", "uid", "id"],
        "query_value": "test",
    },
    "web": {
        "paths": ["open", "deeplink", "launch"],
        "query_keys": ["url", "target", "redirect"],
        "query_value": quote("https://example.com"),
    },
    "generic": {
        "paths": ["open", "home", ""],
        "query_keys": ["target", "id"],
        "query_value": "test",
    },
}


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _guess_task_profile(task: str) -> str:
    lowered = task.lower()
    checks = [
        ("search", ["search", "find", "query", "keyword", "搜", "检索"]),
        ("detail", ["detail", "note", "post", "item", "详情", "详情页"]),
        ("profile", ["profile", "user", "account", "主页", "个人", "用户"]),
        ("web", ["web", "browser", "url", "link", "网页"]),
    ]
    for name, needles in checks:
        if any(needle in lowered for needle in needles):
            return name
    return "generic"


def _guess_schemes(package: str) -> list[str]:
    parts = [segment for segment in package.split(".") if segment]
    candidates = []
    if parts:
        candidates.append(parts[-1])
        if len(parts) >= 2:
            candidates.append("".join(parts[-2:]))
    normalized = re.sub(r"[^a-z0-9]+", "", package.lower())
    if normalized:
        candidates.append(normalized)
    return _dedupe(candidates)


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _run_command(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode, output


def _classify_am_output(output: str) -> tuple[str, str]:
    lowered = output.lower()
    if "unable to resolve intent" in lowered or "error: activity not started" in lowered:
        return "unresolved", "Intent did not resolve to the target package."
    if "status: ok" in lowered and "activity:" in lowered:
        return "launched", "Activity launch succeeded."
    if "warning:" in lowered and "activity not started" in lowered:
        return "partial", "Command reached an activity boundary but did not clearly launch the expected target."
    if "starting: intent" in lowered:
        return "partial", "Intent was accepted but the launch outcome is ambiguous."
    return "unknown", "Need manual inspection of adb output."


def _extract_relevant_lines(output: str) -> list[str]:
    interesting = []
    for line in output.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in ("status:", "activity:", "warning:", "error:", "unable", "intent")):
            interesting.append(line.strip())
    return interesting[:6]


@dataclass
class Probe:
    uri: str
    kind: str
    command: str
    status: str
    evidence: list[str]
    note: str


@dataclass
class ComponentRecord:
    name: str
    kind: str
    exported: str | None = None
    permission: str | None = None
    process: str | None = None
    authorities: list[str] | None = None


def _build_candidates(
    package: str,
    task: str,
    mode: str,
    schemes: list[str],
    hosts: list[str],
    paths: list[str],
    query_keys: list[str],
    candidate_uris: list[str],
) -> list[tuple[str, str]]:
    profile_name = _guess_task_profile(task)
    profile = TASK_PROFILES[profile_name]
    merged_paths = _dedupe(paths + profile["paths"])
    merged_query_keys = _dedupe(query_keys + profile["query_keys"])
    query_value = profile["query_value"]

    if mode == "fast":
        merged_paths = merged_paths[:3]
        merged_query_keys = merged_query_keys[:2]

    guesses = _dedupe(schemes + _guess_schemes(package))
    seed_candidates: list[tuple[str, str]] = [(uri, "seed") for uri in candidate_uris]
    custom_candidates: list[tuple[str, str]] = []
    web_candidates: list[tuple[str, str]] = []

    for scheme in guesses:
        for path in merged_paths:
            base_uri = f"{scheme}://{path}" if path else f"{scheme}://"
            custom_candidates.append((base_uri, "custom_scheme"))
            for key in merged_query_keys:
                custom_candidates.append((f"{base_uri}?{key}={query_value}", "custom_scheme"))

    for host in hosts:
        base_host = host if host.startswith("http") else f"https://{host}"
        for path in merged_paths:
            path_part = f"/{path}" if path else ""
            base_uri = f"{base_host}{path_part}"
            web_candidates.append((base_uri, "web_link"))
            for key in merged_query_keys:
                web_candidates.append((f"{base_uri}?{key}={query_value}", "web_link"))

    limits = {"fast": {"custom": 8, "web": 4, "total": 12}, "investigate": {"custom": 18, "web": 12, "total": 30}}
    current = limits[mode]
    combined = seed_candidates + custom_candidates[: current["custom"]] + web_candidates[: current["web"]]
    return _dedupe_probe_candidates(combined, current["total"])


def _dedupe_probe_candidates(candidates: list[tuple[str, str]], total_limit: int) -> list[tuple[str, str]]:
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for uri, kind in candidates:
        if uri in seen:
            continue
        seen.add(uri)
        deduped.append((uri, kind))
        if len(deduped) >= total_limit:
            break
    return deduped


def _adb_prefix(serial: str | None) -> list[str]:
    command = ["adb"]
    if serial:
        command.extend(["-s", serial])
    return command


def _probe_uri(package: str, serial: str | None, uri: str, kind: str, no_exec: bool) -> Probe:
    command_parts = _adb_prefix(serial) + [
        "shell",
        "am",
        "start",
        "-W",
        "-a",
        "android.intent.action.VIEW",
        "-d",
        uri,
        package,
    ]
    command = _shell_join(command_parts)
    if no_exec:
        return Probe(
            uri=uri,
            kind=kind,
            command=command,
            status="planned",
            evidence=[],
            note="Dry run only; command not executed.",
        )

    returncode, output = _run_command(command_parts)
    status, note = _classify_am_output(output)
    if returncode != 0 and status == "unknown":
        status = "error"
    return Probe(
        uri=uri,
        kind=kind,
        command=command,
        status=status,
        evidence=_extract_relevant_lines(output),
        note=note,
    )


SECTION_LABELS = {
    "activities:": "activity",
    "services:": "service",
    "receivers:": "receiver",
    "providers:": "provider",
}


def _normalize_component_name(package: str, value: str) -> str:
    component = value.rstrip(":")
    if "/" not in component:
        return component
    pkg, cls = component.split("/", 1)
    if cls.startswith("."):
        cls = f"{pkg}{cls}"
    return f"{pkg}/{cls}"


def _match_component_line(package: str, line: str) -> str | None:
    patterns = [
        rf"(?P<component>{re.escape(package)}/[A-Za-z0-9_.$]+)",
        rf"(?P<component>{re.escape(package)}/\.[A-Za-z0-9_.$]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return _normalize_component_name(package, match.group("component"))
    return None


def _extract_unique_matches(pattern: str, text: str) -> list[str]:
    seen: set[str] = set()
    matches: list[str] = []
    for match in re.findall(pattern, text, flags=re.IGNORECASE):
        value = match if isinstance(match, str) else match[0]
        cleaned = value.strip().strip(",")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            matches.append(cleaned)
    return matches


def _parse_package_profile(package: str, output: str) -> dict[str, object]:
    authorities = _extract_unique_matches(r"authorit(?:y|ies)\s*[:=]\s*([^\s,;]+)", output)
    permissions = _extract_unique_matches(
        r"(?:permission|readPermission|writePermission)\s*[:=]\s*([A-Za-z0-9._]+)",
        output,
    )
    processes = _extract_unique_matches(r"process(?:Name)?\s*[:=]\s*([A-Za-z0-9._:]+)", output)
    browsable_lines = [
        line.strip()
        for line in output.splitlines()
        if any(token in line.lower() for token in ("view", "browsable", "scheme=", "host=", "path"))
    ][:16]
    view_actions = _extract_unique_matches(r"android\.intent\.action\.([A-Z_]+)", output)
    schemes = _extract_unique_matches(r"scheme(?:=|:)\s*([A-Za-z][A-Za-z0-9+.-]*)", output)
    hosts = _extract_unique_matches(r"host(?:=|:)\s*([A-Za-z0-9._-]+)", output)
    paths = _extract_unique_matches(r"path(?:Prefix|Pattern|Suffix)?(?:=|:)\s*([^\s,;]+)", output)

    components: list[ComponentRecord] = []
    current_section: str | None = None
    current_component: ComponentRecord | None = None

    for raw_line in output.splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if lowered in SECTION_LABELS:
            current_section = SECTION_LABELS[lowered]
            current_component = None
            continue

        component_name = _match_component_line(package, stripped)
        if component_name:
            current_component = ComponentRecord(name=component_name, kind=current_section or "component", authorities=[])
            components.append(current_component)
            continue

        if current_component is None:
            continue

        exported_match = re.search(r"exported\s*[:=]\s*(true|false)", stripped, flags=re.IGNORECASE)
        permission_match = re.search(
            r"(?:permission|readPermission|writePermission)\s*[:=]\s*([A-Za-z0-9._]+)",
            stripped,
            flags=re.IGNORECASE,
        )
        process_match = re.search(r"process(?:Name)?\s*[:=]\s*([A-Za-z0-9._:]+)", stripped, flags=re.IGNORECASE)
        authority_match = re.search(r"authorit(?:y|ies)\s*[:=]\s*([^\s,;]+)", stripped, flags=re.IGNORECASE)

        if exported_match:
            current_component.exported = exported_match.group(1).lower()
        if permission_match and not current_component.permission:
            current_component.permission = permission_match.group(1)
        if process_match and not current_component.process:
            current_component.process = process_match.group(1)
        if authority_match:
            current_component.authorities = _dedupe((current_component.authorities or []) + [authority_match.group(1)])

    exported_components: dict[str, list[str]] = {"activity": [], "service": [], "receiver": [], "provider": []}
    gated_components: list[dict[str, str]] = []
    for component in components:
        if component.kind in exported_components and component.exported == "true":
            exported_components[component.kind].append(component.name)
        if component.permission or component.exported == "false":
            gated_components.append(
                {
                    "name": component.name,
                    "kind": component.kind,
                    "exported": component.exported or "unknown",
                    "permission": component.permission or "",
                }
            )
        if component.authorities:
            authorities = _dedupe(authorities + component.authorities)
        if component.process:
            processes = _dedupe(processes + [component.process])

    return {
        "component_counts": {
            "activities": sum(1 for item in components if item.kind == "activity"),
            "services": sum(1 for item in components if item.kind == "service"),
            "receivers": sum(1 for item in components if item.kind == "receiver"),
            "providers": sum(1 for item in components if item.kind == "provider"),
        },
        "exported_components": {kind: values[:8] for kind, values in exported_components.items() if values},
        "provider_authorities": authorities[:8],
        "permissions": permissions[:12],
        "processes": processes[:8],
        "view_actions": view_actions[:8],
        "schemes": schemes[:8],
        "hosts": hosts[:8],
        "paths": paths[:8],
        "browsable_hints": browsable_lines,
        "gated_components": gated_components[:12],
    }


def _build_follow_up_commands(
    package: str,
    profile: dict[str, object],
    serial: str | None,
) -> list[str]:
    prefix = _shell_join(_adb_prefix(serial))
    commands: list[str] = [f"{prefix} shell dumpsys package {shlex.quote(package)}"]

    exported = profile.get("exported_components", {})
    if isinstance(exported, dict):
        for activity in exported.get("activity", [])[:2]:
            commands.append(f"{prefix} shell am start -W -n {shlex.quote(activity)}")
        for service in exported.get("service", [])[:1]:
            commands.append(f"{prefix} shell dumpsys activity service {shlex.quote(service)}")

    authorities = profile.get("provider_authorities", [])
    if isinstance(authorities, list):
        for authority in authorities[:2]:
            commands.append(f"{prefix} shell content query --uri content://{authority}")

    commands.append(f"{prefix} shell dumpsys activity top")
    commands.append(f"{prefix} shell dumpsys activity services")
    commands.append(f"{prefix} shell dumpsys package domain-preferred-apps")
    commands.append(
        f"{prefix} logcat -v time | grep -iE {shlex.quote(package + '|ActivityTaskManager|ActivityManager|SecurityException|Provider')}"
    )
    return _dedupe(commands)[:8]


def _inspect_package(
    package: str,
    serial: str | None,
    no_exec: bool,
    package_dump_file: str | None,
) -> dict[str, object]:
    package_cmd = _adb_prefix(serial) + ["shell", "dumpsys", "package", package]
    domain_cmd = _adb_prefix(serial) + ["shell", "dumpsys", "package", "domain-preferred-apps"]
    result: dict[str, object] = {
        "package_dump_command": _shell_join(package_cmd),
        "domain_check_command": _shell_join(domain_cmd),
    }
    package_output = ""

    if package_dump_file:
        result["package_dump_file"] = package_dump_file
        try:
            with open(package_dump_file, "r", encoding="utf-8") as handle:
                package_output = handle.read()
            result["package_dump_status"] = "ok"
        except OSError as exc:
            result["package_dump_status"] = "error"
            result["package_dump_error"] = str(exc)
    elif no_exec:
        result["note"] = "Dry run only; dumpsys commands not executed."
        return result
    else:
        package_rc, package_output = _run_command(package_cmd)
        result["package_dump_status"] = "ok" if package_rc == 0 else "error"

    if package_output:
        result["package_hints"] = _extract_dump_hints(package_output, package)
        result["package_profile"] = _parse_package_profile(package, package_output)
        result["recommended_commands"] = _build_follow_up_commands(
            package,
            result["package_profile"],
            serial,
        )

    if no_exec and package_dump_file:
        result["domain_check_status"] = "skipped"
        result["domain_hints"] = []
        return result

    domain_rc, domain_output = _run_command(domain_cmd)
    result["domain_check_status"] = "ok" if domain_rc == 0 else "error"
    result["domain_hints"] = _extract_dump_hints(domain_output, package)
    return result


def _extract_dump_hints(output: str, package: str) -> list[str]:
    hints = []
    lowered_package = package.lower()
    for line in output.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in ("view", "browsable", "scheme", "host", "path", lowered_package)):
            hints.append(line.strip())
    return hints[:12]


def _build_summary(probes: list[Probe], mode: str, system_checks: dict[str, object]) -> dict[str, object]:
    best = [probe.uri for probe in probes if probe.status == "launched"][:5]
    partial = [probe.uri for probe in probes if probe.status == "partial"][:5]
    invalid = [probe.uri for probe in probes if probe.status in {"unresolved", "error"}][:5]
    planned = [probe.uri for probe in probes if probe.status == "planned"][:5]
    profile = system_checks.get("package_profile", {})
    provider_authorities = profile.get("provider_authorities", []) if isinstance(profile, dict) else []
    exported = profile.get("exported_components", {}) if isinstance(profile, dict) else {}

    next_tries: list[str]
    if best:
        next_tries = ["Reuse the best launched URI and refine only one parameter or path segment at a time."]
    elif partial:
        next_tries = [
            "Keep the same scheme/host and vary one path level or parameter name.",
            "Inspect dumpsys output for BROWSABLE/path hints before widening the search.",
        ]
    elif planned:
        next_tries = [
            "Execute the top planned candidates on a connected device.",
            "If https links fall back to browser, inspect domain-preferred-apps before discarding the host.",
        ]
    else:
        next_tries = [
            "Add known schemes, hosts, or a direct candidate URI from documentation or trace context.",
            "Switch to investigate mode for a wider candidate set.",
        ]

    if mode == "fast" and not best:
        next_tries.append("Escalate to investigate mode if the fast candidate set does not reach the target state.")
    if provider_authorities:
        next_tries.append("Probe the discovered provider authorities with adb shell content query before widening URI guesses.")
    if isinstance(exported, dict) and exported.get("activity"):
        next_tries.append("Validate the likely exported activity entry points with am start -W -n to separate component reachability from deep-link routing.")
    recommended_commands = system_checks.get("recommended_commands", [])

    return {
        "best_candidates": best,
        "partial_matches": partial,
        "invalid_candidates": invalid,
        "next_tries": next_tries[:5],
        "recommended_commands": recommended_commands[:5] if isinstance(recommended_commands, list) else [],
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, help="Target Android package name.")
    parser.add_argument("--task", default="open target page", help="Short task description.")
    parser.add_argument("--serial", help="Optional adb device serial.")
    parser.add_argument("--mode", choices=["fast", "investigate"], default="fast")
    parser.add_argument("--scheme", action="append", default=[], help="Known custom scheme. Repeat for more.")
    parser.add_argument("--host", action="append", default=[], help="Known host. Repeat for more.")
    parser.add_argument("--path", action="append", default=[], help="Known candidate path. Repeat for more.")
    parser.add_argument("--query-key", action="append", default=[], help="Known query key. Repeat for more.")
    parser.add_argument("--candidate-uri", action="append", default=[], help="Direct URI seed. Repeat for more.")
    parser.add_argument("--package-dump-file", help="Parse an existing dumpsys package output instead of querying adb.")
    parser.add_argument("--no-exec", action="store_true", help="Only plan probes; do not run adb commands.")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    candidates = _build_candidates(
        package=args.package,
        task=args.task,
        mode=args.mode,
        schemes=args.scheme,
        hosts=args.host,
        paths=args.path,
        query_keys=args.query_key,
        candidate_uris=args.candidate_uri,
    )
    probes = [
        _probe_uri(args.package, args.serial, uri=uri, kind=kind, no_exec=args.no_exec)
        for uri, kind in candidates
    ]
    system_checks = _inspect_package(
        args.package,
        args.serial,
        no_exec=args.no_exec,
        package_dump_file=args.package_dump_file,
    )
    payload = {
        "inputs": {
            "package": args.package,
            "task": args.task,
            "serial": args.serial,
            "mode": args.mode,
            "no_exec": args.no_exec,
            "package_dump_file": args.package_dump_file,
            "schemes": _dedupe(args.scheme),
            "hosts": _dedupe(args.host),
            "paths": _dedupe(args.path),
            "query_keys": _dedupe(args.query_key),
            "candidate_uris": _dedupe(args.candidate_uri),
        },
        "probes": [
            {
                "uri": probe.uri,
                "kind": probe.kind,
                "command": probe.command,
                "status": probe.status,
                "evidence": probe.evidence,
                "note": probe.note,
            }
            for probe in probes
        ],
        "system_checks": system_checks,
        "summary": _build_summary(probes, args.mode, system_checks),
    }

    if args.format == "json":
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    print(f"Package: {args.package}")
    print(f"Task: {args.task}")
    print(f"Mode: {args.mode}")
    print()
    profile = system_checks.get("package_profile", {})
    if isinstance(profile, dict) and profile:
        print("Package profile:")
        counts = profile.get("component_counts", {})
        if isinstance(counts, dict):
            print(f"  components: {counts}")
        authorities = profile.get("provider_authorities", [])
        if authorities:
            print(f"  authorities: {', '.join(authorities[:4])}")
        processes = profile.get("processes", [])
        if processes:
            print(f"  processes: {', '.join(processes[:4])}")
        print()
    for probe in probes:
        print(f"[{probe.status}] {probe.uri}")
        print(f"  command: {probe.command}")
        if probe.evidence:
            print(f"  evidence: {' | '.join(probe.evidence)}")
        print(f"  note: {probe.note}")
    print()
    print("Next tries:")
    for item in payload["summary"]["next_tries"]:
        print(f"- {item}")
    if payload["summary"]["recommended_commands"]:
        print()
        print("Recommended commands:")
        for command in payload["summary"]["recommended_commands"]:
            print(f"- {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
