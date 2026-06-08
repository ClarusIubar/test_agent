from __future__ import annotations

from typing import Any

from .tool_types import ToolAdapter, ToolResult


class ToolRegistry:
    """Name-based registry for reusable tool adapters."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolAdapter] = {}

    def register(self, tool: ToolAdapter) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolAdapter:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        return tool.invoke(args)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())
