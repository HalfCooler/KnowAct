"""
guiclaw.postprocessing
======================
Post-run background processor for GUI task trajectories.

This module intentionally keeps the learning path flat:

    summarize/evaluate trace -> evolve failed reused Skill or extract Skill -> write skills.py

There is no skill graph, transition evidence, deeplink discovery, or
JSON-backed skill store in this path.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guiclaw.memory.induction import (
    DEFAULT_MEMORY_BANK_PATH,
    append_to_memory_bank,
    format_trajectory_compact,
    guess_app,
    induce_memory_items,
    resolve_trace_app,
    trace_is_abnormal,
    trace_step_count,
)
from guiclaw.skills.induction import induce_compact_skills_from_trace
from guiclaw.trajectory.recorder import load_trajectory_events, update_result_section

logger = logging.getLogger(__name__)

_FLAT_SKILL_LOCKS: dict[Path, asyncio.Lock] = {}
_RESULT_LOCKS: dict[Path, asyncio.Lock] = {}

_ABNORMAL_TERMINATION_PREFIXES: tuple[str, ...] = (
    "stagnation_detected",
    "step_timeout",
    "intervention_cancelled",
)


def _load_trajectory_result(
    trace_path: Path,
    *,
    subtask_index: int = 1,
) -> dict[str, Any] | None:
    """Return the selected subtask's compact result event."""
    events = load_trajectory_events(trace_path, subtask_index=subtask_index)
    return next((event for event in reversed(events) if event.get("type") == "result"), None)


def _load_completed_reuse(
    trace_path: Path,
    *,
    subtask_index: int = 1,
) -> dict[str, Any] | None:
    """Return reuse metadata when a reused flat skill completed the task."""
    full_reuse: dict[str, Any] | None = None
    agent_work_after_full_reuse = False
    for event in load_trajectory_events(trace_path, subtask_index=subtask_index):
        event_type = event.get("type")
        if event_type == "step" and full_reuse is not None:
            action = event.get("action")
            action_type = action.get("action_type") if isinstance(action, dict) else None
            if action_type != "done":
                agent_work_after_full_reuse = True
            continue
        if event_type == "skill_execution_result" and event.get("state") == "succeeded":
            full_reuse = {
                "reuse_source": "skill",
                "skill_id": event.get("skill_id"),
                "skill_name": event.get("skill_name"),
            }
            agent_work_after_full_reuse = False
    if agent_work_after_full_reuse:
        return None
    return full_reuse


def _is_abnormal_termination(result_event: dict[str, Any]) -> bool:
    """True if the trajectory was cut short by detector or infrastructure noise."""
    error = result_event.get("error")
    total_steps = result_event.get("total_steps") or 0
    if total_steps == 0 and error:
        return True
    if not isinstance(error, str):
        return False
    if total_steps > 0 and (
        error == "stagnation_detected"
        or error.startswith("stagnation_detected:")
        or error == "step_timeout"
        or error.startswith("step_timeout:")
    ):
        return False
    return any(
        error == prefix or error.startswith(prefix + ":")
        for prefix in _ABNORMAL_TERMINATION_PREFIXES
    )


def _learning_mode(is_success: bool) -> str:
    return "success_full" if is_success else "failure_prefix"


@dataclass
class EvaluationConfig:
    """Mirrors the evaluation subset of the host config."""

    enabled: bool = False
    judge_model: str = "qwen3-vl-plus"
    api_key: str = ""
    api_base: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class PostRunProcessor:
    """Schedule and manage post-run background tasks for GUI trajectories."""

    def __init__(
        self,
        *,
        llm: Any,
        merge_llm: Any | None = None,
        embedding_provider: Any | None = None,
        embedding_signature: str | None = None,
        skill_store_root: Path | None = None,
        enable_skill_extraction: bool = False,
        enable_memory_extraction: bool = False,
        memory_bank_path: Path | None = None,
        evaluation: EvaluationConfig | None = None,
    ) -> None:
        self._llm = llm
        self._merge_llm = merge_llm
        self._embedding_provider = embedding_provider
        self._embedding_signature = embedding_signature
        self._skill_store_root = skill_store_root
        self._enable_skill_extraction = enable_skill_extraction
        self._enable_memory_extraction = enable_memory_extraction
        self._memory_bank_path = memory_bank_path or DEFAULT_MEMORY_BANK_PATH
        self._evaluation = evaluation or EvaluationConfig()
        self._pending: set[asyncio.Task[None]] = set()

    def schedule(
        self,
        trace_path: Path | None,
        *,
        is_success: bool,
        platform: str,
        task: str,
        subtask_index: int = 1,
    ) -> None:
        if trace_path is None or not trace_path.exists():
            return
        bg = asyncio.create_task(
            self._run_all(
                trace_path,
                is_success=is_success,
                platform=platform,
                task=task,
                subtask_index=subtask_index,
            ),
            name=f"postprocess-{trace_path.stem}-{subtask_index}",
        )
        self._pending.add(bg)
        bg.add_done_callback(self._on_done)

    async def drain(self) -> None:
        if not self._pending:
            return
        await asyncio.gather(*list(self._pending), return_exceptions=True)

    async def _run_all(
        self,
        trace_path: Path,
        *,
        is_success: bool,
        platform: str,
        task: str,
        subtask_index: int = 1,
    ) -> None:
        summary, evaluation_result, _ = await asyncio.gather(
            self._summarize_trajectory(trace_path, subtask_index=subtask_index),
            self._run_evaluation(
                trace_path=trace_path,
                is_success=is_success,
                task=task,
                subtask_index=subtask_index,
            ),
            self._extract_memory(
                trace_path,
                is_success=is_success,
                task=task,
                subtask_index=subtask_index,
            ),
        )
        if evaluation_result is not None:
            await self._write_result_section(
                trace_path,
                "evaluation",
                subtask_index,
                evaluation_result,
            )
        effective_success = is_success
        if isinstance(evaluation_result, dict) and evaluation_result.get("success") is False:
            effective_success = False
        evolution_result = await self._evolve_failed_skill(
            trace_path,
            effective_success,
            platform,
            task=task,
            evaluation_result=evaluation_result,
            agent_success=is_success,
            subtask_index=subtask_index,
        )
        if evolution_result is not None and evolution_result.get("status") != "no_failure_case":
            if summary:
                logger.info("Trajectory state note: %s", summary.replace("\n", " | ")[:200])
            return
        await self._extract_skill(
            trace_path,
            effective_success,
            platform,
            task=task,
            evaluation_result=evaluation_result,
            agent_success=is_success,
            subtask_index=subtask_index,
        )
        if summary:
            logger.info("Trajectory state note: %s", summary.replace("\n", " | ")[:200])

    async def _extract_memory(
        self,
        trace_path: Path,
        *,
        is_success: bool,
        task: str,
        subtask_index: int = 1,
    ) -> dict[str, Any] | None:
        if not self._enable_memory_extraction:
            return None
        if not trace_path.exists():
            return None

        step_count = trace_step_count(trace_path, subtask_index=subtask_index)
        if step_count <= 2:
            result = {"status": "skipped", "reason": "short_trajectory"}
            await self._write_result_section(
                trace_path,
                "memory_extraction",
                subtask_index,
                result,
            )
            return result
        if trace_is_abnormal(trace_path, subtask_index=subtask_index):
            result = {"status": "skipped", "reason": "abnormal_termination"}
            await self._write_result_section(
                trace_path,
                "memory_extraction",
                subtask_index,
                result,
            )
            return result

        trajectory_text = format_trajectory_compact(
            trace_path,
            subtask_index=subtask_index,
        )
        if not trajectory_text:
            result = {"status": "skipped", "reason": "no_usable_steps"}
            await self._write_result_section(
                trace_path,
                "memory_extraction",
                subtask_index,
                result,
            )
            return result

        try:
            app = resolve_trace_app(
                trace_path,
                subtask_index=subtask_index,
            ) or guess_app(task)
            items = await induce_memory_items(
                llm=self._llm,
                trajectory_text=trajectory_text,
                task_outcome="success" if is_success else "failure",
                app=app,
            )
            if not items:
                result = {
                    "status": "no_candidate",
                    "added_count": 0,
                    "duplicate_count": 0,
                }
            else:
                added = append_to_memory_bank(items, self._memory_bank_path)
                result = {
                    "status": "processed",
                    "added_count": added,
                    "duplicate_count": len(items) - added,
                }
        except Exception as exc:
            logger.warning("Memory extraction failed for %s", trace_path, exc_info=True)
            result = {
                "status": "error",
                "reason": str(exc) or type(exc).__name__,
            }

        await self._write_result_section(
            trace_path,
            "memory_extraction",
            subtask_index,
            result,
        )
        return result

    async def _summarize_trajectory(self, trace_path: Path, *, subtask_index: int = 1) -> str:
        if not trace_path.exists():
            return ""
        try:
            from guiclaw.trajectory.summarizer import TrajectorySummarizer

            summarizer = TrajectorySummarizer(llm=self._llm)
            return await summarizer.summarize_file(trace_path, subtask_index=subtask_index)
        except Exception:
            logger.warning("Trajectory summarization failed for %s", trace_path, exc_info=True)
            return ""

    async def _evolve_failed_skill(
        self,
        trace_path: Path,
        is_success: bool,
        platform: str,
        *,
        task: str | None = None,
        evaluation_result: dict[str, Any] | None = None,
        agent_success: bool | None = None,
        subtask_index: int = 1,
    ) -> dict[str, Any] | None:
        if not self._enable_skill_extraction:
            return None
        if not trace_path.exists():
            return None

        store_root = self._skill_store_root or trace_path.parent
        lock = _FLAT_SKILL_LOCKS.setdefault(
            store_root.expanduser().resolve(strict=False),
            asyncio.Lock(),
        )
        try:
            from guiclaw.skills.evolution import evolve_failed_skill_from_trace

            async with lock:
                result = await evolve_failed_skill_from_trace(
                    llm=self._llm,
                    trace_path=trace_path,
                    store_root=store_root,
                    task=task,
                    platform=platform,
                    embedding_provider=self._embedding_provider,
                    embedding_signature=self._embedding_signature,
                    subtask_index=subtask_index,
                )
        except Exception as exc:
            logger.warning("Skill evolution failed for %s", trace_path, exc_info=True)
            result = {
                "status": "error",
                "trace": str(trace_path),
                "reason": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            }

        await self._write_result_section(
            trace_path,
            "evolution",
            subtask_index,
            result,
        )
        if result.get("status") == "no_failure_case":
            return result

        status = (
            "processed_evolution"
            if result.get("status") == "processed_evolution"
            else "evolution_error"
        )
        await self._write_result_section(
            trace_path,
            "extraction",
            subtask_index,
            {
                "status": status,
                "trace": str(trace_path),
                "is_success": is_success,
                "agent_success": agent_success,
                "evaluation_success": _evaluation_success(evaluation_result),
                "platform": platform,
                "task": task,
                "learning_mode": _learning_mode(is_success),
                "updated_functions": list(result.get("updated_functions") or []),
                "compiled_skill_ids": list(result.get("compiled_skill_ids") or []),
                "reuse_failure_trace": True,
                "ordinary_code_extraction_skipped": True,
            },
        )
        return result

    async def _extract_skill(
        self,
        trace_path: Path,
        is_success: bool,
        platform: str,
        *,
        task: str | None = None,
        evaluation_result: dict[str, Any] | None = None,
        agent_success: bool | None = None,
        subtask_index: int = 1,
    ) -> str | None:
        if not self._enable_skill_extraction:
            logger.info("Skipping skill extraction: disabled")
            return None
        if not trace_path.exists():
            return None

        result_event = _load_trajectory_result(trace_path, subtask_index=subtask_index)
        if result_event is not None and _is_abnormal_termination(result_event):
            await self._write_result_section(
                trace_path,
                "extraction",
                subtask_index,
                {
                    "status": "skipped",
                    "reason": "abnormal_termination",
                    "trace": str(trace_path),
                    "is_success": is_success,
                    "agent_success": agent_success,
                    "evaluation_success": _evaluation_success(evaluation_result),
                    "platform": platform,
                    "learning_mode": _learning_mode(is_success),
                    "updated_functions": [],
                    "compiled_skill_ids": [],
                },
            )
            return None

        completed_reuse = _load_completed_reuse(trace_path, subtask_index=subtask_index)
        if completed_reuse is not None:
            await self._write_result_section(
                trace_path,
                "extraction",
                subtask_index,
                {
                    "status": "skipped_reused_skill_complete",
                    "trace": str(trace_path),
                    "is_success": is_success,
                    "agent_success": agent_success,
                    "evaluation_success": _evaluation_success(evaluation_result),
                    "platform": platform,
                    "learning_mode": _learning_mode(is_success),
                    "updated_functions": [],
                    "compiled_skill_ids": [],
                    **completed_reuse,
                },
            )
            return None

        try:
            from guiclaw.skills.extractor import SkillExtractor

            extractor = SkillExtractor(llm=self._llm)
            skills = await induce_compact_skills_from_trace(
                extractor,
                trace_path,
                is_success=is_success,
                subtask_index=subtask_index,
            )
            if not skills:
                await self._write_result_section(
                    trace_path,
                    "extraction",
                    subtask_index,
                    {
                        "status": "no_candidate",
                        "reason": "extractor_returned_none",
                        "trace": str(trace_path),
                        "is_success": is_success,
                        "agent_success": agent_success,
                        "evaluation_success": _evaluation_success(evaluation_result),
                        "platform": platform,
                        "learning_mode": _learning_mode(is_success),
                        "updated_functions": [],
                        "compiled_skill_ids": [],
                        "extractor_diagnostics": extractor.last_diagnostics,
                    },
                )
                return None

            store_root = self._skill_store_root or trace_path.parent
            lock = _FLAT_SKILL_LOCKS.setdefault(
                store_root.expanduser().resolve(strict=False),
                asyncio.Lock(),
            )
            skill_infos = []
            compiled_skill_ids: list[str] = []

            async with lock:
                from guiclaw.skills.flat import FlatSkillLibrary

                library = FlatSkillLibrary(
                    store_dir=store_root,
                    embedding_provider=self._embedding_provider,
                    merge_llm=self._merge_llm,
                    embedding_signature=self._embedding_signature,
                )
                for skill in skills:
                    decision, skill_id = await library.add_or_merge(skill)
                    skill_infos.append(
                        {
                            "skill_id": skill_id,
                            "decision": decision,
                            "name": skill.name,
                            "description": skill.description,
                            "app": skill.app,
                            "platform": skill.platform,
                            "step_count": len(skill.steps),
                        },
                    )
                    if skill_id is not None:
                        compiled_skill_ids.append(skill_id)

            await self._write_result_section(
                trace_path,
                "extraction",
                subtask_index,
                {
                    "status": "processed_code",
                    "trace": str(trace_path),
                    "is_success": is_success,
                    "agent_success": agent_success,
                    "evaluation_success": _evaluation_success(evaluation_result),
                    "platform": platform,
                    "task": task,
                    "learning_mode": _learning_mode(is_success),
                    "updated_functions": [skill.name for skill in skills],
                    "compiled_skill_ids": compiled_skill_ids,
                    "source_path": str((store_root / "skills.py").expanduser()),
                    "extractor_diagnostics": extractor.last_diagnostics,
                    "skills": skill_infos,
                },
            )
            return compiled_skill_ids[0]
        except Exception as exc:
            logger.warning("Skill extraction failed for %s", trace_path, exc_info=True)
            await self._write_result_section(
                trace_path,
                "extraction",
                subtask_index,
                {
                    "status": "error",
                    "trace": str(trace_path),
                    "reason": str(exc) or type(exc).__name__,
                    "error_type": type(exc).__name__,
                    "is_success": is_success,
                    "agent_success": agent_success,
                    "evaluation_success": _evaluation_success(evaluation_result),
                    "platform": platform,
                    "learning_mode": _learning_mode(is_success),
                    "updated_functions": [],
                    "compiled_skill_ids": [],
                },
            )
            return None

    async def _run_evaluation(
        self,
        *,
        trace_path: Path,
        is_success: bool,
        task: str,
        subtask_index: int = 1,
    ) -> dict[str, Any] | None:
        if not self._evaluation.enabled:
            return None
        if not is_success:
            logger.info("Skipping GUI evaluation for unsuccessful task: %s", task)
            return None
        if not trace_path.exists():
            logger.info("Skipping GUI evaluation: trace missing %s", trace_path)
            return None

        api_key = self._evaluation.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.info("Skipping GUI evaluation: no api_key configured")
            return None

        try:
            from guiclaw.evaluation import evaluate_gui_trajectory

            result = await evaluate_gui_trajectory(
                instruction=task,
                trace_path=trace_path,
                model=self._evaluation.judge_model,
                api_key=api_key,
                api_base=self._evaluation.api_base,
                task_id=trace_path.parent.name,
                output_path=None,
                subtask_index=subtask_index,
            )
            logger.info(
                "GUI evaluation completed: success=%s reason=%s",
                result.get("success"),
                result.get("reason"),
            )
            return result
        except Exception:
            logger.warning("GUI evaluation failed for %s", trace_path, exc_info=True)
            return None

    def _on_done(self, task: asyncio.Task[None]) -> None:
        self._pending.discard(task)
        if task.cancelled():
            logger.debug("Post-processing task was cancelled: %s", task.get_name())
            return
        try:
            exc = task.exception()
        except Exception:
            logger.warning("Post-processing task failed", exc_info=True)
            return
        if exc is not None:
            logger.warning(
                "Post-processing task failed: %s",
                task.get_name(),
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    @staticmethod
    async def _write_result_section(
        trace_path: Path,
        section: str,
        subtask_index: int,
        result: dict[str, Any],
    ) -> None:
        result_path = trace_path.parent / "result.json"
        lock = _RESULT_LOCKS.setdefault(
            result_path.expanduser().resolve(strict=False),
            asyncio.Lock(),
        )
        async with lock:
            try:
                update_result_section(trace_path.parent, section, subtask_index, result)
            except OSError as exc:
                logger.warning("Could not update %s in %s: %s", section, result_path, exc)


def _evaluation_success(evaluation_result: dict[str, Any] | None) -> bool | None:
    if not isinstance(evaluation_result, dict):
        return None
    value = evaluation_result.get("success")
    return value if isinstance(value, bool) else None


__all__ = [
    "EvaluationConfig",
    "PostRunProcessor",
    "_is_abnormal_termination",
    "_learning_mode",
    "_load_completed_reuse",
    "_load_trajectory_result",
]
