#!/usr/bin/env python3
"""Induce compact, parameterized GUI skills from successful MobileWorld traces.

This script no longer reimplements its own extraction prompt/filter.  Instead it
reuses the mature :class:`guiclaw.skills.extractor.SkillExtractor` pipeline — the
same codegen-driven structured evidence, fixed/parameter handling, ``state_contract``
guards, quality loop, and contract alignment that the online extractor uses — and
then applies a thin *compact* post-process on top:

* keep only skills that fit the compact envelope (>=2 and <= ``--max-steps`` steps,
  ``open_app`` only as the first step, at most ``--max-scroll-steps`` scrolls);
* re-tag as ``["compact", "compact_extracted"]`` with ``compact:<app>:<name>`` ids
  (step ``valid_state`` / ``state_contract`` guards are preserved verbatim — skipping
  small-model NL validation for compact skills is a runtime tier policy, not baked
  into the definition, so ``optional`` popup steps keep their conditional guard);
* cluster across trajectories by ``(app, action sequence, contract signature,
  parameter slots)`` and record ``success_count`` so high-frequency, repeatedly
  successful skills are preferred at retrieval time.

By default only *successful* tasks are mined. ``--include-failures`` additionally
mines the reusable succeeded *prefix* of FAILED trajectories (the extractor's
failure mode forbids fixing terminal actions like send/pay/delete). Failure
sources never count toward ``success_count`` and a skill is produced only when it
also appears in a success trace; clusters that drew on a failure source are tagged
``from_failure``.

The target app is taken from the trajectory itself (``codegen`` app resolution),
not guessed from the task name, so generated skills actually match at retrieval.

Usage::

    # Inspect the structured evidence that would be sent to the extractor
    python scripts/induce_compact_skills.py --trace-dir /path/to/TaskDir --dry-run

    # Batch-extract from all successful tasks under a trace root
    python scripts/induce_compact_skills.py \\
        --trace-root ~/Project/MobileWorld_fork/traj_logs/v2 \\
        --output ~/.guiclaw/skill/skills.py \\
        --model deepseek-v4-pro --base-url https://...

Note: the extractor sends screenshots alongside the structured trajectory, so the
``--model`` must be vision-capable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from guiclaw.action import normalize_action_type
from guiclaw.interfaces import LLMResponse
from guiclaw.memory.induction import (
    find_gui_task_traces as _find_gui_task_traces,
)
from guiclaw.memory.induction import (
    get_trace_outcome,
    trace_is_abnormal,
)
from guiclaw.memory.induction import (
    trace_step_count as _trace_step_count,
)
from guiclaw.skills.data import Skill, collect_placeholder_names
from guiclaw.skills.extractor import SkillExtractor
from guiclaw.skills.flat import (
    DEFAULT_SKILLS_SOURCE_PATH,
    compile_flat_skills,
    export_skills_to_source,
)
from guiclaw.skills.induction import (
    induce_compact_skills_from_trace,
)
from guiclaw.skills.state_contract import state_contract_fingerprint
from guiclaw.skills.trajectory_codegen import codegen_to_extraction_text, codegen_trajectory
from guiclaw.trajectory.recorder import trajectory_subtask_indices
from scripts.induce_gui_memory import get_task_outcome

_FROM_FAILURE_TAG = "from_failure"
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------


class _OpenAICompatLLM:
    """Minimal :class:`guiclaw.interfaces.LLMProvider` over an OpenAI-compatible API.

    The extractor only ever calls ``chat(messages)`` with a single user message
    that already contains the rendered prompt and screenshots, so this adapter
    only has to forward that payload and surface token usage.
    """

    def __init__(self, *, api_key: str, base_url: str, model: str, max_tokens: int,
                 temperature: float = 0.1) -> None:
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
        response = await self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            max_tokens=max_tokens or self._max_tokens,
            temperature=self._temperature,
        )
        usage = getattr(response, "usage", None)
        usage_dict: dict[str, int] = {}
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, key, None)
                if value is not None:
                    usage_dict[key] = int(value)
        return LLMResponse(
            content=response.choices[0].message.content or "",
            tool_calls=None,
            raw=response,
            usage=usage_dict,
        )


def _literal_target(text: str) -> str:
    """Normalized target text with ``{{placeholders}}`` removed.

    Used as a cluster discriminator so that two same-app workflows sharing an
    action sequence but acting on different controls (``tap "To"`` vs
    ``tap "Search"``) do not collapse into one cluster.  Placeholders are
    stripped first so a parameterized target still matches across trajectories.
    """
    stripped = _PLACEHOLDER_RE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _cluster_key(skill: Skill) -> tuple[Any, ...]:
    """Structural identity of a skill for cross-trajectory clustering.

    Two skills cluster together when they target the same app and share the same
    sequence of (normalized action type, ``state_contract`` signature, literal
    target text, parameter slots).  Placeholder-only differences in a target are
    ignored so the same parameterized workflow still clusters, but distinct
    literal controls keep skills apart to avoid inflating ``success_count``.
    """
    step_sig = tuple(
        (
            normalize_action_type(step.action_type),
            state_contract_fingerprint(step.state_contract),
            _literal_target(step.target),
            tuple(sorted(collect_placeholder_names(
                (step.target, step.parameters, step.fixed_values)
            ))),
        )
        for step in skill.steps
    )
    return (skill.app, step_sig)


def cluster_compact_skills(
    items: list[tuple[Skill, bool]],
    *,
    min_support: int,
) -> list[Skill]:
    """Collapse structurally-identical skills, recording observed support.

    ``items`` pairs each induced skill with whether its source trajectory
    *succeeded*.  A cluster is produced only when it has at least one
    success-derived member (so a workflow only ever seen failing is never
    promoted) and its total member count reaches ``min_support`` (failure-derived
    members may help reach the threshold). ``success_count`` counts
    *success-derived* members only, so failures never inflate confidence;
    clusters that include any failure-derived member are tagged ``from_failure``.
    The richest success member (most ``state_contract`` guards) represents the
    cluster.
    """
    groups: dict[tuple[Any, ...], list[tuple[Skill, bool]]] = {}
    for skill, is_success in items:
        groups.setdefault(_cluster_key(skill), []).append((skill, is_success))

    selected: list[Skill] = []
    for members in groups.values():
        success_members = [skill for skill, ok in members if ok]
        if not success_members or len(members) < min_support:
            continue
        representative = max(success_members, key=_contract_richness)
        has_failure = len(success_members) < len(members)
        tags = representative.tags
        if has_failure and _FROM_FAILURE_TAG not in tags:
            tags = (*tags, _FROM_FAILURE_TAG)
        selected.append(
            replace(
                representative,
                success_count=len(success_members),
                tags=tags,
            )
        )

    selected.sort(key=lambda s: (-s.success_count, s.app, s.name))
    return _disambiguate_skill_ids(selected)


def _contract_richness(skill: Skill) -> int:
    return sum(1 for step in skill.steps if step.state_contract)


def _disambiguate_skill_ids(skills: list[Skill]) -> list[Skill]:
    """Ensure skill ids are unique by suffixing structural collisions."""
    seen: dict[str, int] = {}
    out: list[Skill] = []
    for skill in skills:
        base_id = skill.skill_id
        count = seen.get(base_id, 0)
        seen[base_id] = count + 1
        if count == 0:
            out.append(skill)
            continue
        suffix = count + 1
        out.append(replace(skill, skill_id=f"{base_id}_{suffix}", name=f"{skill.name}_{suffix}"))
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def merge_into_output(skills: list[Skill], path: Path) -> int:
    """Merge *skills* into the compact skills file by ``skill_id``.

    Newly induced definitions win; ``success_count`` is kept monotonic (max of
    existing and new) so re-running on a subset never silently lowers support.
    Returns the number of newly added skill ids.
    """
    existing: list[Skill] = []
    if path.exists():
        compiled = compile_flat_skills(path.read_text(encoding="utf-8"))
        if compiled.errors:
            raise ValueError(f"existing {path} does not compile: {compiled.errors}")
        existing = list(compiled.skills)

    by_id: dict[str, Skill] = {s.skill_id: s for s in existing}
    new_ids = 0
    for skill in skills:
        prior = by_id.get(skill.skill_id)
        if prior is None:
            new_ids += 1
            by_id[skill.skill_id] = skill
        else:
            by_id[skill.skill_id] = replace(
                skill,
                success_count=max(skill.success_count, prior.success_count),
            )

    ordered = sorted(by_id.values(), key=lambda s: (-s.success_count, s.app, s.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export_skills_to_source(ordered), encoding="utf-8")
    return new_ids


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def induce_from_trace(
    extractor: SkillExtractor,
    trace_path: Path,
    *,
    max_steps: int,
    max_scroll_steps: int,
    is_success: bool = True,
    subtask_index: int = 1,
) -> list[Skill]:
    """Extract and compactify all skills from a single trace.

    When ``is_success`` is False the extractor mines only the reusable succeeded
    *prefix* of a failed trajectory (its ``_FAILURE_NOTE`` forbids fixing terminal
    actions such as send/pay/delete).
    """
    return await induce_compact_skills_from_trace(
        extractor,
        trace_path,
        max_steps=max_steps,
        max_scroll_steps=max_scroll_steps,
        is_success=is_success,
        subtask_index=subtask_index,
    )


def _discover_task_dirs(args: argparse.Namespace) -> list[Path]:
    if args.trace_dir:
        return [args.trace_dir]
    root = args.trace_root.expanduser()
    task_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if args.tasks:
        task_dirs = [d for d in task_dirs if d.name in set(args.tasks)]
    if args.limit > 0:
        task_dirs = task_dirs[: args.limit]
    return task_dirs


async def main_async(args: argparse.Namespace) -> int:
    if not args.trace_dir and not args.trace_root:
        print("Error: --trace-dir or --trace-root required", file=sys.stderr)
        return 1

    task_dirs = _discover_task_dirs(args)
    if not task_dirs:
        print("No task directories found.", file=sys.stderr)
        return 1

    api_key = os.getenv(args.api_key_env, "")
    if not api_key and not args.dry_run:
        print(f"Error: {args.api_key_env} not set", file=sys.stderr)
        return 1

    extractor: SkillExtractor | None = None
    if not args.dry_run:
        extractor = SkillExtractor(
            _OpenAICompatLLM(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        )

    all_items: list[tuple[Skill, bool]] = []
    processed = 0
    skipped_failure = 0
    skipped_no_trace = 0
    skipped_short = 0
    skipped_abnormal = 0

    for task_dir in task_dirs:
        task_outcome, _ = get_task_outcome(task_dir)

        trace_paths = _find_gui_task_traces(task_dir)
        if not trace_paths:
            skipped_no_trace += 1
            continue

        usable: list[tuple[Path, int, bool]] = []
        for tp in trace_paths:
            for subtask_index in trajectory_subtask_indices(tp):
                trace_outcome = get_trace_outcome(tp, subtask_index=subtask_index)
                outcome = trace_outcome[0] if trace_outcome is not None else task_outcome
                is_success = outcome == "success"
                if not is_success and not args.include_failures:
                    skipped_failure += 1
                    continue
                if _trace_step_count(tp, subtask_index=subtask_index) <= 2:
                    skipped_short += 1
                    continue
                if trace_is_abnormal(tp, subtask_index=subtask_index):
                    skipped_abnormal += 1
                    continue
                usable.append((tp, subtask_index, is_success))
        if not usable:
            continue

        if args.dry_run:
            print(f"\n{'=' * 60}\nTask dir: {task_dir.name}")
            for tp, subtask_index, is_success in usable:
                result = codegen_trajectory(tp, subtask_index=subtask_index)
                if result is None:
                    continue
                tag = "success" if is_success else "FAILURE(prefix)"
                print(
                    f"--- {tp.name} subtask {subtask_index} "
                    f"(app={result.app}, {tag}) ---"
                )
                print(codegen_to_extraction_text(result)[:1800])
            continue

        assert extractor is not None
        print(f"[SKL] {task_dir.name} ... ", end="", flush=True)
        task_items: list[tuple[Skill, bool]] = []
        for tp, subtask_index, is_success in usable:
            try:
                skills = await induce_from_trace(
                    extractor,
                    tp,
                    max_steps=args.max_steps,
                    max_scroll_steps=args.max_scroll_steps,
                    is_success=is_success,
                    subtask_index=subtask_index,
                )
                task_items.extend((skill, is_success) for skill in skills)
            except Exception as exc:  # noqa: BLE001 - keep batch going
                print(f"\n  {tp.name} subtask {subtask_index}: extraction error: {exc}")
        print(f"{len(task_items)} compact skill(s)")
        all_items.extend(task_items)
        processed += 1

    if args.dry_run:
        return 0

    clustered = cluster_compact_skills(all_items, min_support=args.min_support)
    print(
        f"\nInduced {len(all_items)} compact skill(s) -> "
        f"{len(clustered)} after clustering (min_support={args.min_support})"
    )
    if clustered:
        new_count = merge_into_output(clustered, args.output.expanduser())
        print(f"Wrote {len(clustered)} skill(s) ({new_count} new) to {args.output}")

    print(
        f"Processed: {processed} task(s); skipped "
        f"{skipped_failure} failure, {skipped_no_trace} no-trace, "
        f"{skipped_short} short, {skipped_abnormal} abnormal"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--trace-dir", type=Path, help="Single task trace directory")
    src.add_argument("--trace-root", type=Path, help="Root directory with task subdirs")

    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SKILLS_SOURCE_PATH,
        help=f"Output skills file (default: {DEFAULT_SKILLS_SOURCE_PATH})",
    )
    p.add_argument("--model", default=os.getenv("SKILL_INDUCE_MODEL", "deepseek-v4-pro"))
    p.add_argument("--base-url", default=os.getenv("SKILL_INDUCE_BASE_URL", ""))
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--task", action="append", dest="tasks",
                   help="Only process specific task(s). Repeatable.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="Print codegen evidence per trace without LLM calls")
    p.add_argument("--max-steps", type=int, default=7,
                   help="Reject skills longer than this many steps (default: 7)")
    p.add_argument("--max-scroll-steps", type=int, default=1,
                   help="Reject skills with more than this many scroll/swipe/drag steps")
    p.add_argument("--min-support", type=int, default=1,
                   help="Keep only skills whose cluster reaches this many trajectories "
                        "(success + failure sources)")
    p.add_argument("--include-failures", action="store_true",
                   help="Also mine the reusable succeeded prefix of FAILED trajectories. "
                        "Failure-derived skills never count toward success_count and are "
                        "kept only when the same skill also appears in a success trace; "
                        "such clusters are tagged 'from_failure'.")
    return p.parse_args(argv)


def main() -> None:
    raise SystemExit(asyncio.run(main_async(parse_args())))


if __name__ == "__main__":
    main()
