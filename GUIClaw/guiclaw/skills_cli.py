"""Command-line maintenance operations for the flat GUI skill library."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from guiclaw.backends.adb import AdbBackend
from guiclaw.cli import (
    DEFAULT_CONFIG_PATH,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
    load_config,
)
from guiclaw.image_utils import scale_image
from guiclaw.skills.action_grounder import ActionGrounder
from guiclaw.skills.executor import ExecutionState, LLMStateValidator, SkillExecutor
from guiclaw.skills.flat import (
    CANONICAL_SKILLS_FILENAME,
    DEFAULT_SKILLS_STORE_DIR,
    FlatSkillLibrary,
)
from guiclaw.skills.observation_provider import AgentScreenshotProvider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="guiclaw skills")
    commands = parser.add_subparsers(dest="command", required=True)
    optimize = commands.add_parser(
        "optimize",
        help="Replace, delete, or keep UI skills using related deeplinks.",
    )
    optimize.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_SKILLS_STORE_DIR,
        help=f"Flat skill store (default: {DEFAULT_SKILLS_STORE_DIR})",
    )
    optimize.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Provider config (default: {DEFAULT_CONFIG_PATH})",
    )
    optimize.add_argument(
        "--all",
        action="store_true",
        help="Inspect all eligible UI skills (currently the default scope).",
    )
    mode = optimize.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate proposals without writing skill_optimizations.json (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply accepted prefix recipes and UI-skill deletions.",
    )
    optimize.add_argument(
        "--validate",
        action="store_true",
        help="Execute proposed optimized skills on the connected ADB device.",
    )
    optimize.add_argument(
        "--promote",
        action="store_true",
        help="Persist only proposals that pass --validate.",
    )
    optimize.add_argument("--serial", default="", help="ADB device serial.")
    optimize.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help="Directory for validation screenshots.",
    )
    optimize.add_argument("--embedding-top-k", type=int, default=3)
    optimize.add_argument("--report", type=Path, help="Write the JSON report to this path.")
    optimize.add_argument(
        "--llm-base-url",
        default="",
        help="Override postprocess_provider.base_url from config.",
    )
    optimize.add_argument(
        "--llm-model",
        default="",
        help="Override postprocess_provider.model from config.",
    )
    optimize.add_argument("--llm-api-key", default="", help=argparse.SUPPRESS)
    optimize.add_argument(
        "--llm-api-key-env",
        default="",
        help="Override API key with the named environment variable.",
    )
    optimize.add_argument("--llm-temperature", type=float, default=None)
    return parser.parse_args(argv)


def _providers(
    args: argparse.Namespace,
) -> tuple[OpenAICompatibleLLMProvider, OpenAICompatibleEmbeddingProvider, str, str]:
    config = load_config(args.config.expanduser())
    configured = config.postprocess_provider
    if configured is None:
        raise ValueError("postprocess_provider is required in config")
    base_url = str(args.llm_base_url).strip() or configured.base_url
    model = str(args.llm_model).strip() or configured.model
    api_key = str(args.llm_api_key).strip()
    if not api_key and args.llm_api_key_env:
        api_key = os.environ.get(args.llm_api_key_env, "")
    api_key = api_key or configured.api_key or ""
    llm = OpenAICompatibleLLMProvider(
        base_url=base_url,
        model=model,
        api_key=api_key or None,
        temperature=(
            args.llm_temperature if args.llm_temperature is not None else configured.temperature
        ),
        top_p=configured.top_p,
        reasoning_effort=configured.reasoning_effort,
        extra_body=configured.extra_body,
    )
    if config.embedding is None:
        raise ValueError("embedding provider is required in config")
    embedding = OpenAICompatibleEmbeddingProvider(
        base_url=config.embedding.base_url,
        model=config.embedding.model,
        api_key=config.embedding.api_key,
    )
    return llm, embedding, model, config.embedding.model


async def _run_optimize(args: argparse.Namespace) -> dict[str, Any]:
    store = args.store.expanduser()
    source = store / CANONICAL_SKILLS_FILENAME
    provider, embedding_provider, llm_model, embedding_model = _providers(args)
    args.resolved_llm_model = llm_model
    library = FlatSkillLibrary(
        store_dir=store,
        merge_llm=provider,
        embedding_provider=embedding_provider,
        embedding_signature=embedding_model,
    )
    persist = bool(args.apply or args.promote)
    validator = _build_validator(args, provider) if args.validate else None
    try:
        kwargs: dict[str, Any] = {
            "persist": persist,
            "embedding_top_k": args.embedding_top_k,
        }
        if validator is not None:
            kwargs["validator"] = validator
        report = await library.optimize_shortcut_prefixes(**kwargs)
    finally:
        if validator is not None:
            await validator.shutdown()
    report.update(
        {
            "mode": "apply" if persist else "dry-run",
            "validation_enabled": bool(args.validate),
            "store": str(store),
            "source": str(source),
            "providers": {
                "fusion_model": llm_model,
                "embedding_model": embedding_model,
            },
        }
    )
    return report


class _DeviceSkillValidator:
    def __init__(self, args: argparse.Namespace, llm: OpenAICompatibleLLMProvider) -> None:
        self.llm = llm
        self.backend = AdbBackend(
            serial=args.serial or None,
            use_scrcpy=False,
            collect_ui_tree=True,
            collect_ui_tree_nodes=True,
        )
        root = args.validation_root or Path(tempfile.mkdtemp(prefix="guiclaw-skill-validation-"))
        self.screenshots = AgentScreenshotProvider(self.backend, root.expanduser())
        self.state_validator = LLMStateValidator(llm)
        self.executor = SkillExecutor(
            backend=self.backend,
            state_validator=self.state_validator,
            action_grounder=ActionGrounder(
                llm=llm,
                model=args.resolved_llm_model,
                agent_profile="general_compact",
            ),
            screenshot_provider=self.screenshots,
        )

    async def __call__(self, skill, params: dict[str, str]) -> tuple[bool, str]:
        result = await self.executor.execute(skill, params, timeout=60.0)
        if result.state is not ExecutionState.SUCCEEDED:
            return False, result.error or result.execution_summary
        screenshot = await self.screenshots.get_screenshot()
        if screenshot is None:
            return False, "final screenshot unavailable"
        final_valid = await _verify_final_page(self.llm, skill.description, screenshot)
        return final_valid, result.execution_summary

    async def shutdown(self) -> None:
        await self.backend.shutdown()


def _build_validator(
    args: argparse.Namespace,
    llm: OpenAICompatibleLLMProvider,
) -> _DeviceSkillValidator:
    return _DeviceSkillValidator(args, llm)


async def _verify_final_page(
    llm: OpenAICompatibleLLMProvider,
    task: str,
    screenshot: Path,
) -> bool:
    image = base64.b64encode(scale_image(screenshot.read_bytes(), scale_ratio=0.5)).decode()
    try:
        response = await llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Judge whether this Android screenshot proves the following skill "
                                f"goal is complete: {task}\n"
                                'Return JSON only: {"valid":true|false}'
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image}"},
                        },
                    ],
                }
            ],
            max_tokens=64,
            reasoning_effort="none",
        )
        payload = json.loads(str(response.content or ""))
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("valid") is True


def _write_report(path: Path, report: dict[str, Any]) -> None:
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.promote and not args.validate:
            raise ValueError(
                "--promote requires --validate; use --apply for unvalidated persistence"
            )
        report = asyncio.run(_run_optimize(args))
        if args.report is not None:
            _write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"skills optimize: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
