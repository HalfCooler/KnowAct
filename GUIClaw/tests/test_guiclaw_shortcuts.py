from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

MANIFEST = """\
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.reader">
  <application>
    <activity android:name=".MainActivity">
      <intent-filter>
        <action android:name="android.intent.action.VIEW" />
        <category android:name="android.intent.category.BROWSABLE" />
        <data android:scheme="reader" android:host="search" android:pathPrefix="/items" />
      </intent-filter>
      <intent-filter>
        <action android:name="android.intent.action.SEARCH" />
      </intent-filter>
    </activity>
  </application>
</manifest>
"""


def _write_manifest(path: Path, *, package: str = "com.example.reader") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MANIFEST.replace("com.example.reader", package), encoding="utf-8")
    return path


def test_extract_shortcuts_from_local_manifest(tmp_path: Path) -> None:
    from guiclaw.skills.deeplink import extract_shortcuts_from_manifest

    manifest = _write_manifest(tmp_path / "manifest.xml")

    profile = extract_shortcuts_from_manifest(manifest)

    assert profile.package == "com.example.reader"
    assert profile.deep_links[0].uri_template == "reader://search/items"
    assert profile.deep_links[0].path_kind == "pathPrefix"
    assert profile.deep_intents[0].action == "android.intent.action.SEARCH"
    assert profile.manifest_meta == {"manifest_path": str(manifest), "filter_count": 2}


def test_infer_and_record_shortcuts_accepts_single_file(tmp_path: Path) -> None:
    from guiclaw.shortcuts import infer_and_record_shortcuts

    manifest = _write_manifest(tmp_path / "custom-name.xml")
    output = tmp_path / "shortcut_cache"

    cache_paths = infer_and_record_shortcuts(manifest, output)

    assert cache_paths == [output / "com.example.reader.json"]
    payload = json.loads(cache_paths[0].read_text(encoding="utf-8"))
    assert payload["package"] == "com.example.reader"
    assert payload["deep_links"][0]["uri_template"] == "reader://search/items"


def test_infer_and_record_shortcuts_recurses_android_manifests(tmp_path: Path) -> None:
    from guiclaw.shortcuts import infer_and_record_shortcuts

    source = tmp_path / "decoded"
    _write_manifest(source / "one" / "AndroidManifest.xml", package="com.example.one")
    _write_manifest(source / "two" / "nested" / "AndroidManifest.xml", package="com.example.two")
    _write_manifest(source / "ignored.xml", package="com.example.ignored")

    cache_paths = infer_and_record_shortcuts(source, tmp_path / "cache")

    assert [path.name for path in cache_paths] == ["com.example.one.json", "com.example.two.json"]


def test_infer_and_record_shortcuts_rejects_duplicate_packages(tmp_path: Path) -> None:
    from guiclaw.shortcuts import infer_and_record_shortcuts

    source = tmp_path / "decoded"
    _write_manifest(source / "one" / "AndroidManifest.xml")
    _write_manifest(source / "two" / "AndroidManifest.xml")

    with pytest.raises(ValueError, match="duplicate manifest package: com.example.reader"):
        infer_and_record_shortcuts(source, tmp_path / "cache")


def test_shortcuts_cli_does_not_validate_by_default(tmp_path: Path, monkeypatch) -> None:
    from guiclaw import shortcuts

    manifest = _write_manifest(tmp_path / "AndroidManifest.xml")

    def unexpected_validation(argv):
        raise AssertionError(f"validation unexpectedly called with {argv}")

    monkeypatch.setattr(shortcuts, "run_shortcut_validation", unexpected_validation)

    assert shortcuts.main([str(manifest), "--output", str(tmp_path / "cache")]) == 0


def test_shortcuts_cli_validate_routes_generated_cache(tmp_path: Path, monkeypatch) -> None:
    from guiclaw import shortcuts

    manifest = _write_manifest(tmp_path / "AndroidManifest.xml")
    calls: list[list[str]] = []
    monkeypatch.setattr(shortcuts, "run_shortcut_validation", lambda argv: calls.append(argv) or 0)

    exit_code = shortcuts.main(
        [
            str(manifest),
            "--output",
            str(tmp_path / "cache"),
            "--validate",
            "--serial",
            "emulator-5554",
        ]
    )

    assert exit_code == 0
    assert calls == [
        [
            "--cache",
            str(tmp_path / "cache" / "com.example.reader.json"),
            "--execute",
            "--serial",
            "emulator-5554",
        ]
    ]


def test_shortcuts_cli_validate_routes_existing_json_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from guiclaw import shortcuts

    cache_path = tmp_path / "com.example.reader.json"
    cache_path.write_text(json.dumps({"package": "com.example.reader"}), encoding="utf-8")
    calls: list[list[str]] = []

    def unexpected_inference(*args) -> list[Path]:
        raise AssertionError(f"manifest inference unexpectedly called with {args}")

    monkeypatch.setattr(shortcuts, "infer_and_record_shortcuts", unexpected_inference)
    monkeypatch.setattr(shortcuts, "run_shortcut_validation", lambda argv: calls.append(argv) or 0)

    exit_code = shortcuts.main([str(cache_path), "--validate", "--promote"])

    assert exit_code == 0
    assert calls == [["--cache", str(cache_path), "--execute", "--promote"]]


def test_guiclaw_cli_dispatches_shortcuts_subcommand(monkeypatch) -> None:
    from guiclaw import cli, shortcuts

    calls: list[list[str]] = []
    monkeypatch.setattr(shortcuts, "main", lambda argv: calls.append(argv) or 7)

    assert cli.main(["shortcuts", "AndroidManifest.xml"]) == 7
    assert calls == [["AndroidManifest.xml"]]


def test_shortcut_and_extraction_share_default_skills_file() -> None:
    from guiclaw import cli, shortcut_validation
    from guiclaw.skills.flat import DEFAULT_SKILLS_SOURCE_PATH
    from scripts import induce_compact_skills

    validation_args = shortcut_validation.parse_args(["--cache", "cache.json"])
    extraction_args = induce_compact_skills.parse_args([])

    assert DEFAULT_SKILLS_SOURCE_PATH == Path.home() / ".guiclaw" / "skill" / "skills.py"
    assert cli.DEFAULT_SKILLS_DIR / "skills.py" == DEFAULT_SKILLS_SOURCE_PATH
    assert validation_args.skill_store_root / "skills.py" == DEFAULT_SKILLS_SOURCE_PATH
    assert extraction_args.output == DEFAULT_SKILLS_SOURCE_PATH


def test_default_runtime_storage_uses_guiclaw_home() -> None:
    from guiclaw import cli, shortcuts
    from guiclaw.agent import GuiAgent

    guiclaw_home = Path.home() / ".guiclaw"
    parameters = inspect.signature(GuiAgent).parameters

    assert cli.DEFAULT_GUI_RUNS_DIR == guiclaw_home / "gui_runs"
    assert shortcuts.DEFAULT_SHORTCUT_CACHE_DIR == guiclaw_home / "shortcut_cache"
    assert parameters["artifacts_root"].default == guiclaw_home / "gui_runs"
    assert parameters["shortcut_cache_dir"].default == guiclaw_home / "shortcut_cache"


def test_guiclaw_data_dirs_support_relative_and_absolute_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from guiclaw import paths

    guiclaw_home = tmp_path / ".guiclaw"
    absolute = tmp_path / "external-runs"
    monkeypatch.setattr(paths, "DEFAULT_GUICLAW_HOME", guiclaw_home)

    assert paths.resolve_guiclaw_data_dir("custom-runs") == guiclaw_home / "custom-runs"
    assert paths.resolve_guiclaw_data_dir(absolute) == absolute


def test_default_nanobot_workspace_uses_shared_guiclaw_skills_file(tmp_path: Path) -> None:
    from guiclaw.skills.flat import DEFAULT_SKILLS_STORE_DIR
    from guiclaw.skills.normalization import get_gui_skill_store_root

    assert get_gui_skill_store_root(Path.home() / ".nanobot" / "workspace") == (
        DEFAULT_SKILLS_STORE_DIR
    )
    assert get_gui_skill_store_root(tmp_path) == tmp_path / "gui_skills"


def test_flat_repository_orders_shortcuts_before_extracted_skills(tmp_path: Path) -> None:
    from guiclaw.skills.data import Skill
    from guiclaw.skills.flat import FlatSkillRepository

    extracted = Skill(
        skill_id="compact:com.example.reader:search",
        name="search",
        description="Search content",
        app="com.example.reader",
        platform="android",
        tags=("compact", "compact_extracted"),
    )
    shortcut = Skill(
        skill_id="shortcut:dl:com.example.reader:search",
        name="open_search",
        description="Open search",
        app="com.example.reader",
        platform="android",
        tags=("shortcut", "deeplink", "validated"),
    )
    repository = FlatSkillRepository(tmp_path)

    repository.replace_all([extracted, shortcut])

    assert [skill.skill_id for skill in repository.list_all()] == [
        shortcut.skill_id,
        extracted.skill_id,
    ]
