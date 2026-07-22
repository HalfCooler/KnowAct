"""Adapter bridge from nanobot providers to guiclaw protocols."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from guiclaw.interfaces import LLMResponse as GUIClawLLMResponse
from guiclaw.interfaces import ToolCall
from nanobot.providers.base import LLMProvider as NanobotLLMProvider


class NanobotLLMAdapter:
    """Wrap a nanobot LLM provider with guiclaw's chat interface.

    When ``capture_ttft`` is True, the adapter routes through
    ``chat_stream_with_retry`` so the first text delta timestamp can be
    recorded as ``ttft_s``.  If the model emits no text content (pure
    tool_calls), ``ttft_s`` is left as ``None``.  ``latency_s`` is always
    captured from wall-clock around the LLM call.
    """

    def __init__(
        self,
        provider: NanobotLLMProvider,
        model: str,
        *,
        capture_ttft: bool = False,
        max_tokens: int | None = 2048,
    ) -> None:
        self._provider = provider
        self._model = model
        self._capture_ttft = capture_ttft
        # Default per-call output cap for GUI decision steps. The nanobot
        # provider otherwise falls back to ``generation.max_tokens`` (which can
        # be very large, e.g. 8192), letting a single step run away — e.g. a
        # non-thinking model rolling out dozens of repeated ``Thought:/Action:``
        # pairs until the server cap. Callers may override per-call.
        self._max_tokens = max_tokens

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> GUIClawLLMResponse:
        effective_max_tokens = max_tokens if max_tokens is not None else self._max_tokens
        kwargs: dict[str, Any] = dict(
            messages=messages,
            tools=tools,
            model=model or self._model,
            tool_choice=tool_choice,
        )
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        ttft_s: float | None = None
        start = time.perf_counter()

        if self._capture_ttft:
            first_delta_at: list[float] = []

            async def _on_delta(_: str) -> None:
                if not first_delta_at:
                    first_delta_at.append(time.perf_counter())

            nano_resp = await self._provider.chat_stream_with_retry(
                **kwargs, on_content_delta=_on_delta,
            )
            if first_delta_at:
                ttft_s = first_delta_at[0] - start
        else:
            nano_resp = await self._provider.chat_with_retry(**kwargs)

        latency_s = time.perf_counter() - start

        tool_calls = [
            ToolCall(id=call.id, name=call.name, arguments=call.arguments)
            for call in (nano_resp.tool_calls or [])
        ] or None
        return GUIClawLLMResponse(
            content=nano_resp.content or "",
            tool_calls=tool_calls,
            raw=nano_resp,
            usage=nano_resp.usage or {},
            ttft_s=ttft_s,
            latency_s=latency_s,
        )


class NanobotEmbeddingAdapter:
    """Wrap an async embedding callable with guiclaw's embed interface."""

    def __init__(self, embed_fn: Callable[[list[str]], Awaitable[np.ndarray]]) -> None:
        self._embed_fn = embed_fn

    async def embed(self, texts: list[str]) -> np.ndarray:
        return await self._embed_fn(texts)
