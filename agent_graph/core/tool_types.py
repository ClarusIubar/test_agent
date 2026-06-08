from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ToolResult:
    """Normalized output for tool calls across providers."""

    status: str
    content: str
    raw: Any | None = None
    error: str | None = None


class ToolAdapter(Protocol):
    """Contract for reusable tool adapters."""

    name: str

    def invoke(self, args: dict[str, Any]) -> ToolResult:
        ...
