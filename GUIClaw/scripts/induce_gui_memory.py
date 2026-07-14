#!/usr/bin/env python3
"""Induce lightweight GUI memory from one task directory or a trace root.

The reusable formatter, parser, filters, and bank writer live in
``guiclaw.memory.induction`` so installed GUIClaw can run the same pipeline
automatically after a configured ``gui_task``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

from guiclaw.interfaces import LLMResponse
from guiclaw.memory.gui_memory_item import GuiMemoryItem
from guiclaw.memory.induction import (
    DEFAULT_DEDUP_THRESHOLD as _DEFAULT_DEDUP_THRESHOLD,
)
from guiclaw.memory.induction import (
    DEFAULT_MEMORY_BANK_PATH,
    append_to_memory_bank,
    format_trajectory_compact,
    get_trace_outcome,
    induce_memory_items,
    load_memory_bank,
    parse_memory_items,
    trace_is_abnormal,
)
from guiclaw.memory.induction import (
    find_gui_task_traces as _find_gui_task_traces,
)
from guiclaw.memory.induction import (
    guess_app as _guess_app,
)
from guiclaw.memory.induction import (
    is_abnormal_termination as _is_abnormal_termination,
)
from guiclaw.memory.induction import (
    resolve_trace_app as _resolve_trace_app,
)
from guiclaw.memory.induction import (
    trace_step_count as _trace_step_count,
)
from guiclaw.trajectory.recorder import trajectory_subtask_indices


class _OpenAICompatMemoryLLM:
    """Small OpenAI-compatible adapter used only by this offline CLI."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del tools, tool_choice
        response = await self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            max_tokens=max_tokens or self._max_tokens,
            temperature=self._temperature,
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            tool_calls=None,
            raw=response,
        )


def get_task_outcome(task_dir: Path) -> tuple[str, str]:
    """Return a MobileWorld task directory's score-derived outcome."""
    result_txt = task_dir / "result.txt"
    if not result_txt.exists():
        return "failure", "no result.txt found"
    text = result_txt.read_text(encoding="utf-8", errors="ignore")
    score_match = re.search(r"(?m)^score:\s*([0-9.]+)", text)
    if not score_match:
        return "failure", "no score field in result.txt"
    try:
        score = float(score_match.group(1))
    except ValueError:
        return "failure", f"unparseable score: {score_match.group(1)!r}"
    return ("success", "") if score >= 1.0 else ("failure", f"score={score}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format-only",
        type=Path,
        help="Format one trace and print it without calling an LLM",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--trace-dir", type=Path, help="Single task trace directory")
    source.add_argument("--trace-root", type=Path, help="Root with multiple task directories")
    parser.add_argument(
        "--memory-bank",
        type=Path,
        default=DEFAULT_MEMORY_BANK_PATH,
        help=f"JSONL memory bank path (default: {DEFAULT_MEMORY_BANK_PATH})",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=_DEFAULT_DEDUP_THRESHOLD,
        help="Same-app token Jaccard threshold used for duplicate suppression",
    )
    parser.add_argument(
        "--max-items-per-task",
        type=int,
        default=5,
        help="Maximum items contributed by one task directory (0 = unlimited)",
    )
    parser.add_argument("--model", default=os.getenv("MEMORY_INDUCE_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--base-url", default=os.getenv("MEMORY_INDUCE_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _select_task_items(items: list[GuiMemoryItem], budget: int) -> list[GuiMemoryItem]:
    """Prefer failure lessons and round-robin apps within a task budget."""
    if budget <= 0 or len(items) <= budget:
        return items
    groups: dict[str, list[GuiMemoryItem]] = {}
    for item in items:
        groups.setdefault(item.app or "", []).append(item)
    for app_items in groups.values():
        app_items.sort(key=lambda item: item.status == "success")

    order = list(groups)
    selected: list[GuiMemoryItem] = []
    index = 0
    while len(selected) < budget and any(groups[app] for app in order):
        app = order[index % len(order)]
        if groups[app]:
            selected.append(groups[app].pop(0))
        index += 1
    return selected


async def main_async(args: argparse.Namespace) -> int:
    if args.format_only:
        formatted = [
            text
            for index in trajectory_subtask_indices(args.format_only)
            if (text := format_trajectory_compact(args.format_only, subtask_index=index))
        ]
        if not formatted:
            print("No usable steps found in trace.", file=sys.stderr)
            return 1
        print("\n\n".join(formatted))
        return 0

    if not args.trace_dir and not args.trace_root:
        print("Error: --trace-dir or --trace-root is required.", file=sys.stderr)
        return 1

    if args.trace_dir:
        task_dirs = [args.trace_dir]
    else:
        root = args.trace_root.expanduser()
        task_dirs = sorted(
            directory
            for directory in root.iterdir()
            if directory.is_dir() and not directory.name.startswith(".")
        )
        if args.tasks:
            selected_tasks = set(args.tasks)
            task_dirs = [directory for directory in task_dirs if directory.name in selected_tasks]
    if args.limit > 0:
        task_dirs = task_dirs[: args.limit]
    if not task_dirs:
        print("No task directories found.", file=sys.stderr)
        return 1

    api_key = os.getenv(args.api_key_env, "")
    if not api_key and not args.dry_run:
        print(f"Error: {args.api_key_env} not set. Use --dry-run to skip LLM calls.", file=sys.stderr)
        return 1
    llm = None
    if not args.dry_run:
        llm = _OpenAICompatMemoryLLM(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    all_items: list[GuiMemoryItem] = []
    for task_dir in task_dirs:
        task_name = task_dir.name
        trace_paths = _find_gui_task_traces(task_dir)
        if not trace_paths:
            print(f"[SKIP] {task_name}: no trace found")
            continue

        task_outcome, task_error = get_task_outcome(task_dir)
        jobs: list[tuple[Path, int, str, int, str, str | None]] = []
        skipped_short = 0
        skipped_empty = 0
        skipped_abnormal = 0
        for trace_path in trace_paths:
            for subtask_index in trajectory_subtask_indices(trace_path):
                step_count = _trace_step_count(trace_path, subtask_index=subtask_index)
                if step_count <= 2:
                    skipped_short += 1
                    continue
                if trace_is_abnormal(trace_path, subtask_index=subtask_index):
                    skipped_abnormal += 1
                    continue
                trajectory_text = format_trajectory_compact(
                    trace_path,
                    subtask_index=subtask_index,
                )
                if trajectory_text is None:
                    skipped_empty += 1
                    continue
                trace_outcome = get_trace_outcome(trace_path, subtask_index=subtask_index)
                outcome = trace_outcome[0] if trace_outcome is not None else task_outcome
                app = _resolve_trace_app(
                    trace_path,
                    subtask_index=subtask_index,
                ) or _guess_app(task_name)
                jobs.append(
                    (trace_path, subtask_index, trajectory_text, step_count, outcome, app)
                )

        if not jobs:
            detail = []
            if skipped_short:
                detail.append(f"{skipped_short} short trace(s)")
            if skipped_abnormal:
                detail.append(f"{skipped_abnormal} abnormal trace(s)")
            if skipped_empty:
                detail.append(f"{skipped_empty} empty trace(s)")
            suffix = f" ({', '.join(detail)})" if detail else ""
            print(f"[SKIP] {task_name}: no memory-worthy GUI task traces{suffix}")
            continue

        if args.dry_run:
            print(f"\n{'=' * 60}")
            print(f"Task: {task_name}  (task-level outcome: {task_outcome})")
            if task_error:
                print(f"Task-level note: {task_error}")
            for trace_path, subtask_index, trajectory_text, step_count, outcome, app in jobs:
                print(
                    f"  - {trace_path}#subtask-{subtask_index} ({step_count} steps) "
                    f"outcome={outcome} app={app or '-'}"
                )
                print(trajectory_text[:1500])
            continue

        assert llm is not None
        task_items: list[GuiMemoryItem] = []
        print(f"[GM] {task_name} ... ", end="", flush=True)
        for trace_path, _subtask_index, trajectory_text, _, outcome, app in jobs:
            try:
                items = await induce_memory_items(
                    llm=llm,
                    trajectory_text=trajectory_text,
                    task_outcome=outcome,
                    app=app,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - keep offline batch processing
                print(f"\n  {trace_path}: LLM error: {exc}")
                continue
            task_items.extend(items)
        selected = _select_task_items(task_items, args.max_items_per_task)
        dropped = len(task_items) - len(selected)
        suffix = f" (+{dropped} over per-task budget dropped)" if dropped else ""
        print(f"{len(selected)} item(s){suffix}")
        all_items.extend(selected)

    if all_items:
        bank_path = args.memory_bank.expanduser()
        added = append_to_memory_bank(
            all_items,
            bank_path,
            similarity_threshold=args.dedup_threshold,
        )
        skipped = len(all_items) - added
        suffix = f" ({skipped} near-duplicate(s) skipped)" if skipped else ""
        print(f"\nTotal: {added} new memory item(s) in {bank_path}{suffix}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async(parse_args())))


__all__ = [
    "DEFAULT_MEMORY_BANK_PATH",
    "GuiMemoryItem",
    "_find_gui_task_traces",
    "_guess_app",
    "_is_abnormal_termination",
    "_resolve_trace_app",
    "_select_task_items",
    "_trace_step_count",
    "append_to_memory_bank",
    "format_trajectory_compact",
    "get_task_outcome",
    "get_trace_outcome",
    "load_memory_bank",
    "parse_memory_items",
    "trace_is_abnormal",
]


if __name__ == "__main__":
    main()
