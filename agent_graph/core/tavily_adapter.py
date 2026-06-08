from __future__ import annotations

import json
from typing import Any

from .tool_types import ToolResult


class TavilyAdapter:
    """Tool adapter for Tavily web search."""

    name = "tavily_search"

    def __init__(self, max_results: int = 3) -> None:
        self._tool: Any | None = None
        self._import_error: str | None = None
        try:
            from langchain_tavily import TavilySearch

            self._tool = TavilySearch(max_results=max_results)
        except Exception as exc:  # pragma: no cover - env dependent
            self._import_error = str(exc)

    def invoke(self, args: dict[str, Any]) -> ToolResult:
        if self._import_error:
            return ToolResult(
                status="error",
                content="",
                raw=None,
                error=f"Tavily dependency is unavailable: {self._import_error}",
            )

        if "query" not in args:
            return ToolResult(
                status="error",
                content="",
                raw=None,
                error="Missing required argument: query",
            )

        try:
            raw = self._tool.invoke(args)
        except Exception as exc:  # pragma: no cover - network/env dependent
            return ToolResult(
                status="error",
                content="",
                raw=None,
                error=str(exc),
            )

        return ToolResult(
            status="ok",
            content=json.dumps(raw, ensure_ascii=False),
            raw=raw,
            error=None,
        )
