from __future__ import annotations

from typing import Any

from .tool_registry import ToolRegistry


class ToolExecutionNode:
    """Executes one or more tool calls from state using ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def __call__(self, state: Any) -> dict[str, list[dict[str, Any]]]:
        if isinstance(state, dict):
            tool_calls = state.get("tool_calls", [])
        else:
            tool_calls = getattr(state, "tool_calls", [])

        outputs: list[dict[str, Any]] = []
        for call in tool_calls:
            name = str(call.get("name", ""))
            args = call.get("args", {})
            result = self._registry.invoke(name, args)
            outputs.append(
                {
                    "name": name,
                    "status": result.status,
                    "content": result.content,
                    "error": result.error,
                    "raw": result.raw,
                }
            )

        return {"tool_results": outputs, "tool_calls": []}
