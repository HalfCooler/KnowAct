"""Task-difficulty routing for choosing the GUI actor and prompt profile.

A larger model classifies the incoming task as easy, medium, or hard. That
verdict selects both the actor model and the agent profile for the whole run:

* easy   -> small GUI model + ``general_compact``
* medium -> large model + ``general_compact``
* hard   -> large model + ``general_e2e``

Repeat-plan escalation is unchanged: when the actor is the small model it still
hands a single step to the large model; when the actor is already the large
model the same path replans with that large model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from guiclaw.interfaces import LLMResponse

TaskDifficulty = Literal["easy", "medium", "hard"]
TASK_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
DEFAULT_TASK_DIFFICULTY: TaskDifficulty = "medium"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_DIFFICULTY_TOKEN_RE = re.compile(
    r"\b(easy|medium|hard)\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DifficultyVerdict:
    """Model judgment for how hard a GUI task is."""

    difficulty: TaskDifficulty
    reason: str = ""
    raw_content: str = ""
    fallback: bool = False


@dataclass(frozen=True)
class DifficultyRoute:
    """Actor and profile selected from a difficulty verdict."""

    difficulty: TaskDifficulty
    use_large_model: bool
    agent_profile: str
    reason: str = ""
    raw_content: str = ""
    fallback: bool = False

    @property
    def actor(self) -> str:
        return "large" if self.use_large_model else "small"

    def snapshot(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "actor": self.actor,
            "agent_profile": self.agent_profile,
            "reason": self.reason,
            "fallback": self.fallback,
        }


def format_difficulty_progress(snapshot: dict[str, Any] | DifficultyRoute | None) -> str:
    """One-line CLI summary of a difficulty route or trajectory snapshot."""
    if snapshot is None:
        return ""
    payload = snapshot.snapshot() if isinstance(snapshot, DifficultyRoute) else dict(snapshot)
    difficulty = str(payload.get("difficulty") or DEFAULT_TASK_DIFFICULTY).strip() or (
        DEFAULT_TASK_DIFFICULTY
    )
    actor = str(payload.get("actor") or "").strip() or (
        "large" if payload.get("use_large_model") else "small"
    )
    profile = str(payload.get("agent_profile") or "").strip()
    reason = " ".join(str(payload.get("reason") or "").split()).strip()
    details = [f"actor={actor}"]
    if profile:
        details.append(f"profile={profile}")
    if payload.get("fallback"):
        details.append("fallback")
    line = f"GUI difficulty: {difficulty} ({', '.join(details)})"
    if reason:
        line = f"{line}: {reason}"
    return line


_DIFFICULTY_ROUTES: dict[TaskDifficulty, tuple[bool, str]] = {
    "easy": (False, "general_compact"),
    "medium": (True, "general_compact"),
    "hard": (True, "general_e2e"),
}


def canonicalize_task_difficulty(value: str | None) -> TaskDifficulty | None:
    key = str(value or "").strip().lower()
    if key in TASK_DIFFICULTIES:
        return key  # type: ignore[return-value]
    return None


def route_for_difficulty(
    difficulty: str | None,
    *,
    reason: str = "",
    raw_content: str = "",
    fallback: bool = False,
) -> DifficultyRoute:
    """Map a difficulty label onto the actor model and agent profile."""
    resolved = canonicalize_task_difficulty(difficulty) or DEFAULT_TASK_DIFFICULTY
    use_large_model, agent_profile = _DIFFICULTY_ROUTES[resolved]
    return DifficultyRoute(
        difficulty=resolved,
        use_large_model=use_large_model,
        agent_profile=agent_profile,
        reason=reason,
        raw_content=raw_content,
        fallback=fallback or canonicalize_task_difficulty(difficulty) is None,
    )


def build_difficulty_judge_text(task: str) -> str:
    return (
        "Classify the difficulty of this GUI automation task.\n\n"
        f"Task:\n{task.strip() or '(none)'}\n\n"
        "Choose exactly one label:\n"
        "- easy: a short, single-app task with a clear few-step path "
        "(open an app, tap one control, type a short query).\n"
        "- medium: a multi-step task in one app, or a simple cross-app flow "
        "with a known path.\n"
        "- hard: long-horizon, ambiguous, multi-app, or likely to need "
        "planning, recovery, or careful reading of changing UI state.\n\n"
        'Return JSON only: {"difficulty": "easy"|"medium"|"hard", '
        '"reason": "short reason"}.'
    )


def parse_difficulty_verdict(content: str) -> DifficultyVerdict:
    """Parse a difficulty response. Unknown or empty output falls back to medium."""
    raw = str(content or "").strip()
    if not raw:
        return DifficultyVerdict(
            difficulty=DEFAULT_TASK_DIFFICULTY,
            reason="empty_judge_response",
            raw_content=raw,
            fallback=True,
        )

    payload = _extract_json_object(raw)
    if isinstance(payload, dict):
        difficulty = canonicalize_task_difficulty(
            payload.get("difficulty") or payload.get("level") or payload.get("label")
        )
        if difficulty is not None:
            return DifficultyVerdict(
                difficulty=difficulty,
                reason=str(payload.get("reason") or "").strip(),
                raw_content=raw,
            )

    match = _DIFFICULTY_TOKEN_RE.search(raw)
    if match:
        difficulty = canonicalize_task_difficulty(match.group(1))
        if difficulty is not None:
            return DifficultyVerdict(
                difficulty=difficulty,
                reason=raw[:200],
                raw_content=raw,
            )
    return DifficultyVerdict(
        difficulty=DEFAULT_TASK_DIFFICULTY,
        reason="unparsed_judge_response",
        raw_content=raw,
        fallback=True,
    )


async def judge_task_difficulty(llm: Any, task: str) -> DifficultyVerdict:
    """Ask ``llm`` (the large model) to classify ``task``."""
    prompt = build_difficulty_judge_text(task)
    try:
        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            max_tokens=256,
        )
    except Exception as exc:
        logger.warning("Difficulty judge failed; treating as medium: %s", exc)
        return DifficultyVerdict(
            difficulty=DEFAULT_TASK_DIFFICULTY,
            reason=f"judge_error: {exc}",
            fallback=True,
        )
    content = response.content if isinstance(response, LLMResponse) else str(
        getattr(response, "content", "") or ""
    )
    return parse_difficulty_verdict(content)


async def resolve_difficulty_route(
    *,
    task: str,
    enabled: bool,
    large_llm: Any | None,
    explicit_profile: str | None = None,
) -> DifficultyRoute | None:
    """Return a route when difficulty routing should run, else ``None``.

    Routing is skipped when disabled, when no large model is available, or when
    the caller already chose an agent profile (for example ``--agent-profile``).
    """
    if not enabled or large_llm is None or str(explicit_profile or "").strip():
        return None
    verdict = await judge_task_difficulty(large_llm, task)
    route = route_for_difficulty(
        verdict.difficulty,
        reason=verdict.reason,
        raw_content=verdict.raw_content,
        fallback=verdict.fallback,
    )
    logger.info(
        "Difficulty route: %s -> actor=%s profile=%s (%s)",
        route.difficulty,
        route.actor,
        route.agent_profile,
        route.reason or "no reason",
    )
    return route


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
