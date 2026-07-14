# GUIClaw Adapter Patterns

`guiclaw` stays host-agnostic by depending on two small protocols from
`guiclaw/interfaces.py`: `LLMProvider` and `DeviceBackend`. A host agent keeps
its own runtime, config, and SDK choices on its side of the boundary, then
adapts into these protocols before constructing `GuiAgent`.

The production reference is `NanobotLLMAdapter` in
`nanobot/agent/gui_adapter.py`. Treat that file as a reference example for
adapter authors, not as a runtime dependency for `guiclaw`.

## LLMProvider

`LLMProvider` is the model-facing side of the contract. Your adapter takes the
host runtime's chat client, translates GUIClaw messages and tool definitions
into the host format, calls the host model, then maps the host response back to
`LLMResponse` and `ToolCall`.

The important rule is shape conversion, not inheritance. A host adapter can be
any class with an async `chat(...) -> LLMResponse` method that matches the
protocol.

```python
from __future__ import annotations

from typing import Any

from guiclaw.interfaces import LLMResponse, ToolCall


class ExampleHostLLMAdapter:
    """Wrap a host agent's model client with GUIClaw's chat protocol."""

    def __init__(self, host_client: Any, model: str) -> None:
        self._host_client = host_client
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        host_response = await self._host_client.chat(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        tool_calls = [
            ToolCall(
                id=call["id"],
                name=call["name"],
                arguments=call["arguments"],
            )
            for call in host_response.get("tool_calls", [])
        ] or None
        return LLMResponse(
            content=host_response.get("content", ""),
            tool_calls=tool_calls,
            raw=host_response,
        )
```

Adapter checklist:

- Preserve the final assistant text in `LLMResponse.content`.
- Convert each host tool call into `ToolCall(id, name, arguments)`.
- Normalize "no tool calls" to `None` instead of an empty list.
- Preserve the original provider payload in `raw` when useful for debugging.

If you want a real implementation instead of a starter skeleton, read
`NanobotLLMAdapter` in `nanobot/agent/gui_adapter.py`.

## DeviceBackend

`DeviceBackend` is the execution-facing side of the contract. It must provide:

- `observe(...)` to capture a screenshot plus any metadata GUIClaw needs.
- `execute(...)` to dispatch a single UI action and return a short status
  string.
- `preflight()` to fail early when the target device or desktop is unavailable.
- `platform` to identify the runtime target (`android`, `macos`, `linux`,
  `windows`, and so on).

If your host already runs on Android devices or local desktops, you may be able
to reuse `AdbBackend` or `LocalDesktopBackend` directly. Otherwise, implement
your own `DeviceBackend` around the host's automation APIs.

## Wiring Pattern

Keep the dependency direction one-way:

```text
your-host-runtime -> guiclaw
guiclaw -/-> your-host-runtime
```

That means:

- The host runtime owns adapter modules such as `ExampleHostLLMAdapter`.
- `guiclaw` runtime code should only import protocol types and shared GUIClaw
  modules.
- Reference adapters like `NanobotLLMAdapter` and
  `nanobot/agent/gui_adapter.py` are documentation inputs, not imports for
  `guiclaw`.

Typical wiring looks like this:

1. Build or locate the host's model client.
2. Wrap it in an adapter that satisfies `LLMProvider`.
3. Choose `AdbBackend`, `LocalDesktopBackend`, or a custom `DeviceBackend`.
4. Call `guiclaw.memory.policy.load_policy_context(memory_dir)` and pass the
   result as `GuiAgent(policy_context=...)` if the host constructs the Python API
   directly.
5. Construct `GuiAgent` with the adapted LLM and backend.

`load_policy_context()` creates a conservative default `POLICY` entry only when
the selected memory store has no policies; it does not replace user-defined
entries. The returned policy is advisory prompt context, not a hard enforcement
mechanism. Mandatory restrictions belong in the host, `DeviceBackend`, OS
permissions, intervention handler, or sandbox.

This keeps GUIClaw reusable across claw hosts without coupling the core runtime
to nanobot or any other host-specific framework.
