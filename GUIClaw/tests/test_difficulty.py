from __future__ import annotations

from typing import Any

import pytest

from guiclaw.difficulty import (
    DifficultyRoute,
    canonicalize_task_difficulty,
    judge_task_difficulty,
    parse_difficulty_verdict,
    resolve_difficulty_route,
    route_for_difficulty,
)
from guiclaw.interfaces import LLMResponse


class _FakeLLM:
    def __init__(self, content: str = "", *, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content)


def test_route_for_difficulty_maps_actor_and_profile() -> None:
    easy = route_for_difficulty("easy")
    medium = route_for_difficulty("MEDIUM")
    hard = route_for_difficulty("hard")

    assert easy.use_large_model is False
    assert easy.agent_profile == "general_compact"
    assert easy.actor == "small"
    assert medium.use_large_model is True
    assert medium.agent_profile == "general_compact"
    assert medium.actor == "large"
    assert hard.use_large_model is True
    assert hard.agent_profile == "general_e2e"
    assert route_for_difficulty("unknown").difficulty == "medium"
    assert route_for_difficulty("unknown").fallback is True


def test_canonicalize_task_difficulty() -> None:
    assert canonicalize_task_difficulty("Easy") == "easy"
    assert canonicalize_task_difficulty("HARD") == "hard"
    assert canonicalize_task_difficulty("nope") is None


@pytest.mark.parametrize(
    ("content", "difficulty", "fallback"),
    [
        ('{"difficulty": "easy", "reason": "one tap"}', "easy", False),
        ('{"difficulty": "Medium"}', "medium", False),
        ('{"level": "hard", "reason": "multi-app"}', "hard", False),
        ("```json\n{\"difficulty\": \"easy\"}\n```", "easy", False),
        ("This looks hard overall", "hard", False),
        ("easy", "easy", False),
        ("", "medium", True),
        ("I am not sure", "medium", True),
    ],
)
def test_parse_difficulty_verdict(content: str, difficulty: str, fallback: bool) -> None:
    verdict = parse_difficulty_verdict(content)
    assert verdict.difficulty == difficulty
    assert verdict.fallback is fallback


@pytest.mark.asyncio
async def test_judge_task_difficulty_uses_large_model_prompt() -> None:
    llm = _FakeLLM('{"difficulty": "hard", "reason": "long horizon"}')

    verdict = await judge_task_difficulty(llm, "Book a multi-city itinerary")

    assert verdict.difficulty == "hard"
    assert verdict.reason == "long horizon"
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is None
    assert "Book a multi-city itinerary" in llm.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_judge_task_difficulty_falls_back_to_medium_on_error() -> None:
    llm = _FakeLLM(error=RuntimeError("boom"))

    verdict = await judge_task_difficulty(llm, "Open Settings")

    assert verdict.difficulty == "medium"
    assert verdict.fallback is True
    assert "judge_error" in verdict.reason


@pytest.mark.asyncio
async def test_resolve_difficulty_route_skips_when_disabled() -> None:
    llm = _FakeLLM('{"difficulty": "hard"}')

    route = await resolve_difficulty_route(
        task="Open Settings",
        enabled=False,
        large_llm=llm,
    )

    assert route is None
    assert llm.calls == []


@pytest.mark.asyncio
async def test_resolve_difficulty_route_skips_without_large_model() -> None:
    route = await resolve_difficulty_route(
        task="Open Settings",
        enabled=True,
        large_llm=None,
    )
    assert route is None


@pytest.mark.asyncio
async def test_resolve_difficulty_route_skips_explicit_profile() -> None:
    llm = _FakeLLM('{"difficulty": "hard"}')

    route = await resolve_difficulty_route(
        task="Open Settings",
        enabled=True,
        large_llm=llm,
        explicit_profile="seed",
    )

    assert route is None
    assert llm.calls == []


@pytest.mark.asyncio
async def test_resolve_difficulty_route_returns_easy_actor() -> None:
    llm = _FakeLLM('{"difficulty": "easy", "reason": "one tap"}')

    route = await resolve_difficulty_route(
        task="Open Settings",
        enabled=True,
        large_llm=llm,
    )

    assert isinstance(route, DifficultyRoute)
    assert route.difficulty == "easy"
    assert route.use_large_model is False
    assert route.agent_profile == "general_compact"
    assert route.snapshot()["actor"] == "small"
