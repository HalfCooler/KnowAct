"""Infer, record, and optionally validate Android shortcuts from local manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from guiclaw.paths import DEFAULT_SHORTCUT_CACHE_DIR
from guiclaw.shortcut_validation import add_validation_arguments
from guiclaw.skills.deeplink import AppShortcutProfile, extract_shortcuts_from_manifest
from guiclaw.skills.flat import DEFAULT_SKILLS_STORE_DIR


def infer_and_record_shortcuts(source: Path | str, output_dir: Path | str) -> list[Path]:
    """Infer shortcut profiles from one manifest or a directory and write cache JSON files."""
    source_path = Path(source).expanduser()
    output_path = Path(output_dir).expanduser()
    manifests = _manifest_paths(source_path)
    profiles: list[AppShortcutProfile] = []
    packages: set[str] = set()
    for manifest in manifests:
        profile = extract_shortcuts_from_manifest(manifest)
        if profile.package in packages:
            raise ValueError(f"duplicate manifest package: {profile.package}")
        packages.add(profile.package)
        profiles.append(profile)

    output_path.mkdir(parents=True, exist_ok=True)
    cache_paths: list[Path] = []
    for profile in profiles:
        cache_path = output_path / f"{profile.package}.json"
        cache_path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        cache_paths.append(cache_path)
    return cache_paths


def _manifest_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"manifest source not found: {source}")
    manifests = sorted(source.rglob("AndroidManifest.xml"))
    if not manifests:
        raise ValueError(f"no AndroidManifest.xml files found under: {source}")
    return manifests


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="guiclaw shortcuts",
        description="Infer Android shortcuts from manifests or validate an existing cache JSON.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="AndroidManifest.xml file, manifest directory, or shortcut cache JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SHORTCUT_CACHE_DIR,
        help=f"Shortcut cache directory (default: {DEFAULT_SHORTCUT_CACHE_DIR})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Resolve, launch, and validate shortcut candidates on an ADB device.",
    )
    add_validation_arguments(parser, include_cache=False, include_execute=False)
    return parser.parse_args(argv)


def run_shortcut_validation(argv: list[str]) -> int:
    from guiclaw.shortcut_validation import main as validation_main

    return validation_main(argv)


def _validation_argv(args: argparse.Namespace, cache_path: Path) -> list[str]:
    argv = ["--cache", str(cache_path), "--execute"]
    for name in ("serial", "task", "query", "llm_base_url", "llm_model", "llm_api_key"):
        value = getattr(args, name)
        if value:
            argv.extend((f"--{name.replace('_', '-')}", str(value)))
    if args.llm_api_key_env != "OPENAI_API_KEY":
        argv.extend(("--llm-api-key-env", args.llm_api_key_env))
    if args.llm_temperature != 0.0:
        argv.extend(("--llm-temperature", str(args.llm_temperature)))
    for name, default in (
        ("max_candidates", 20),
        ("max_probe_plans", 12),
        ("max_try", 5),
    ):
        value = getattr(args, name)
        if value != default:
            argv.extend((f"--{name.replace('_', '-')}", str(value)))
    if args.validation_root is not None:
        argv.extend(("--validation-root", str(args.validation_root)))
    if args.skill_store_root != DEFAULT_SKILLS_STORE_DIR:
        argv.extend(("--skill-store-root", str(args.skill_store_root)))
    for name in ("include_risky", "promote", "allow_launchable_promote"):
        if getattr(args, name):
            argv.append(f"--{name.replace('_', '-')}")
    if args.shortcut_postprocess != "rules":
        argv.extend(("--shortcut-postprocess", args.shortcut_postprocess))
    return argv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser()
    cache_paths = (
        [source]
        if source.is_file() and source.suffix.lower() == ".json"
        else infer_and_record_shortcuts(source, args.output)
    )
    for cache_path in cache_paths:
        print(f"shortcut_cache: {cache_path}")
    if not args.validate:
        return 0
    for cache_path in cache_paths:
        exit_code = run_shortcut_validation(_validation_argv(args, cache_path))
        if exit_code:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
