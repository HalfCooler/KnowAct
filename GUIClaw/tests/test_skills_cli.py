from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from guiclaw import cli, skills_cli


def _write_config(path: Path) -> Path:
    path.write_text(
        """provider:
  base_url: https://small.example/v1
  model: small-model
postprocess_provider:
  base_url: https://post.example/v1
  model: post-model
embedding:
  base_url: https://embedding.example/v1
  model: embedding-model
""",
        encoding="utf-8",
    )
    return path


def _fake_providers(monkeypatch) -> None:
    monkeypatch.setattr(
        skills_cli,
        "OpenAICompatibleLLMProvider",
        lambda **kwargs: SimpleNamespace(settings=kwargs),
    )
    monkeypatch.setattr(
        skills_cli,
        "OpenAICompatibleEmbeddingProvider",
        lambda **kwargs: SimpleNamespace(settings=kwargs),
    )


def test_skills_optimize_promote_requires_validate(capsys) -> None:
    assert skills_cli.main(["optimize", "--promote", "--llm-model", "x"]) == 2
    assert "--promote requires --validate" in capsys.readouterr().err


def test_top_level_cli_dispatches_skills(monkeypatch) -> None:
    received: list[str] = []

    def fake_main(argv: list[str]) -> int:
        received.extend(argv)
        return 7

    monkeypatch.setattr(skills_cli, "main", fake_main)

    assert cli.main(["skills", "optimize", "--all", "--dry-run"]) == 7
    assert received == ["optimize", "--all", "--dry-run"]


def test_skills_optimize_dry_run_writes_report_without_store_mutation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    store = tmp_path / "skill"
    store.mkdir()
    source = store / "skills.py"
    source.write_text("# canonical\n", encoding="utf-8")
    calls: list[dict] = []

    class FakeLibrary:
        def __init__(self, **kwargs) -> None:
            calls.append({"init": kwargs})

        async def optimize_shortcut_prefixes(self, **kwargs):
            calls.append({"optimize": kwargs})
            return {
                "status": "processed",
                "candidate_count": 2,
                "activated": [],
                "proposed": ["compact:test"],
                "rejected": [],
            }

    monkeypatch.setattr(skills_cli, "FlatSkillLibrary", FakeLibrary)
    _fake_providers(monkeypatch)
    report_path = tmp_path / "report.json"
    config_path = _write_config(tmp_path / "config.yaml")

    exit_code = skills_cli.main(
        [
            "optimize",
            "--store",
            str(store),
            "--config",
            str(config_path),
            "--all",
            "--dry-run",
            "--report",
            str(report_path),
            "--llm-model",
            "qwen3.7-flash",
        ]
    )

    assert exit_code == 0
    assert calls[1] == {"optimize": {"persist": False, "embedding_top_k": 3}}
    assert source.read_text(encoding="utf-8") == "# canonical\n"
    assert not (store / "skill_optimizations.json").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["proposed"] == ["compact:test"]
    assert '"status": "processed"' in capsys.readouterr().out


def test_skills_optimize_uses_configured_postprocess_and_embedding(
    tmp_path: Path, monkeypatch
) -> None:
    _fake_providers(monkeypatch)
    args = skills_cli.parse_args(
        ["optimize", "--config", str(_write_config(tmp_path / "config.yaml"))]
    )

    llm, embedding, model, embedding_model = skills_cli._providers(args)

    assert model == "post-model"
    assert llm.settings["base_url"] == "https://post.example/v1"
    assert llm.settings["model"] == "post-model"
    assert embedding.settings["base_url"] == "https://embedding.example/v1"
    assert embedding.settings["model"] == "embedding-model"
    assert embedding_model == "embedding-model"


def test_skills_optimize_apply_enables_persistence(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []

    class FakeLibrary:
        def __init__(self, **_kwargs) -> None:
            pass

        async def optimize_shortcut_prefixes(self, **kwargs):
            calls.append(kwargs)
            return {"status": "processed", "activated": [], "proposed": []}

    monkeypatch.setattr(skills_cli, "FlatSkillLibrary", FakeLibrary)
    _fake_providers(monkeypatch)
    config_path = _write_config(tmp_path / "config.yaml")

    exit_code = skills_cli.main(
        [
            "optimize",
            "--store",
            str(tmp_path),
            "--config",
            str(config_path),
            "--apply",
            "--llm-base-url",
            "https://example.test/v1",
            "--llm-model",
            "qwen3.7-flash",
        ]
    )

    assert exit_code == 0
    assert calls == [{"persist": True, "embedding_top_k": 3}]


def test_skills_optimize_validate_promotes_only_after_validation(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict] = []

    class FakeValidator:
        async def shutdown(self) -> None:
            pass

    validator = FakeValidator()

    class FakeLibrary:
        def __init__(self, **_kwargs) -> None:
            pass

        async def optimize_shortcut_prefixes(self, **kwargs):
            calls.append(kwargs)
            return {"status": "processed", "validated": ["compact:test"]}

    monkeypatch.setattr(skills_cli, "FlatSkillLibrary", FakeLibrary)
    _fake_providers(monkeypatch)
    monkeypatch.setattr(skills_cli, "_build_validator", lambda *_args: validator)
    config_path = _write_config(tmp_path / "config.yaml")

    exit_code = skills_cli.main(
        [
            "optimize",
            "--store",
            str(tmp_path),
            "--config",
            str(config_path),
            "--validate",
            "--promote",
            "--llm-base-url",
            "https://example.test/v1",
            "--llm-model",
            "qwen3.7-flash",
        ]
    )

    assert exit_code == 0
    assert calls == [{"persist": True, "embedding_top_k": 3, "validator": validator}]
