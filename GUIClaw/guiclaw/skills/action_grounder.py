"""
guiclaw.skills.action_grounder
==============================
Vision-LLM action grounding for non-fixed ``SkillStep`` objects.

Consumed by ``SkillExecutor`` to resolve parameterised steps into concrete
``Action`` instances at run time.  This was previously an inner class in
``guiclaw.agent``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from guiclaw.action import parse_action
from guiclaw.agent_profiles import (
    canonicalize_agent_profile,
    normalize_profile_response_for_screen,
    profile_uses_native_tools,
    prompt_contract_for_profile,
)
from guiclaw.image_utils import scale_image
from guiclaw.interfaces import LLMProvider
from guiclaw.skills.executor import _ground_text
from guiclaw.tool_schemas import image_dimensions, minimal_tool_schema


class ActionGrounder:
    """Ground a non-fixed SkillStep into a concrete Action via vision LLM.

    Sends the current screenshot and step description to the LLM with the
    ``computer_use`` tool and parses the returned tool call into an Action.
    """

    _MAX_RETRIES = 2

    def __init__(
        self,
        llm: LLMProvider,
        model: str,
        agent_profile: str | None = None,
        image_scale_ratio: float = 0.5,
    ) -> None:
        self._llm = llm
        self._model = model
        self._agent_profile = canonicalize_agent_profile(agent_profile)
        self._image_scale_ratio = image_scale_ratio
        self._usage_accum: dict[str, int] = {}
        self._ttft_samples: list[float] = []
        self._latency_samples: list[float] = []

    def drain_usage(self) -> dict[str, int]:
        usage = dict(self._usage_accum)
        self._usage_accum.clear()
        return usage

    def drain_timings(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._ttft_samples:
            out["ttft_s"] = sum(self._ttft_samples) / len(self._ttft_samples)
        if self._latency_samples:
            out["chat_latency_s"] = sum(self._latency_samples) / len(self._latency_samples)
        self._ttft_samples.clear()
        self._latency_samples.clear()
        return out

    async def ground(
        self,
        step: Any,  # SkillStep — avoid circular import
        screenshot: Path | bytes,
        params: dict[str, str],
    ) -> Any:  # Action — avoid circular import
        target = _ground_text(step.target, params)
        extra_ctx = ""
        if step.parameters:
            resolved_params = {
                k: _ground_text(str(v), params) if isinstance(v, str) else v
                for k, v in step.parameters.items()
            }
            extra_ctx = f"\nContext: {resolved_params}"

        prompt = (
            f"Locate the target UI element on screen and execute the action.\n"
            f"  action_type: {step.action_type}\n"
            f"  target: {target}{extra_ctx}\n\n"
            f"{self._profile_response_instruction(target_action=step.action_type)}"
        )
        raw = screenshot.read_bytes() if isinstance(screenshot, Path) else screenshot
        screen_width, screen_height = image_dimensions(raw)
        image_data = base64.b64encode(
            scale_image(raw, scale_ratio=self._image_scale_ratio)
        ).decode()

        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        }]

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                native_tools_enabled = profile_uses_native_tools(self._agent_profile)
                response = await self._llm.chat(
                    messages=messages,
                    tools=[minimal_tool_schema(step.action_type)] if native_tools_enabled else None,
                    tool_choice="required" if native_tools_enabled else None,
                    model=self._model or None,
                    max_tokens=256,
                )
            except Exception as exc:
                raise RuntimeError(f"ActionGrounder LLM call failed: {exc}") from exc

            for k, v in (response.usage or {}).items():
                self._usage_accum[k] = self._usage_accum.get(k, 0) + v
            if getattr(response, "ttft_s", None) is not None:
                self._ttft_samples.append(response.ttft_s)
            if getattr(response, "latency_s", None) is not None:
                self._latency_samples.append(response.latency_s)

            try:
                response = normalize_profile_response_for_screen(
                    self._agent_profile,
                    response,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    model_name=self._model,
                )
            except Exception as exc:
                if attempt < self._MAX_RETRIES:
                    messages.append({"role": "assistant", "content": response.content or ""})
                    messages.append({
                        "role": "user",
                        "content": f"Format error: {exc}. Follow the configured profile format exactly.",
                    })
                    continue
                raise RuntimeError(f"ActionGrounder profile parse failed: {exc}") from exc

            if response.tool_calls:
                tc = response.tool_calls[0]
                if tc.name == "computer_use" or not native_tools_enabled:
                    try:
                        return parse_action(tc.arguments)
                    except Exception as exc:
                        if attempt < self._MAX_RETRIES:
                            messages.append({"role": "assistant", "content": response.content or ""})
                            if native_tools_enabled:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": f"Error parsing action: {exc}. Please fix.",
                                })
                            else:
                                messages.append({
                                    "role": "user",
                                    "content": f"Format error: {exc}. Please fix and retry.",
                                })
                            continue
                        raise RuntimeError(f"ActionGrounder parse failed: {exc}") from exc

            if attempt < self._MAX_RETRIES:
                messages.append({"role": "assistant", "content": response.content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Error: no computer_use tool call. You must use it."
                        if native_tools_enabled
                        else "Error: no Action block found. Use configured profile format (Thought:/Action:)."
                    ),
                })
            else:
                if native_tools_enabled:
                    raise RuntimeError("ActionGrounder: LLM did not return a computer_use call after retries.")
                raise RuntimeError("ActionGrounder: LLM did not return a valid profile action after retries.")

        raise RuntimeError("ActionGrounder: unexpected exit from retry loop.")

    def _profile_response_instruction(self, *, target_action: str) -> str:
        if self._agent_profile == "default":
            return f"Respond with ONLY a computer_use tool call. You MUST use action_type='{target_action}'."
        contract = prompt_contract_for_profile(self._agent_profile)
        format_lines = " ".join(contract["format"])
        rules = " ".join(contract["rules"][:2])
        return (
            f"Respond using the configured `{self._agent_profile}` profile format. "
            f"Choose the profile-native action that corresponds to canonical action_type='{target_action}'. "
            f"{format_lines} {rules}"
        )
