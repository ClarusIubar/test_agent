from __future__ import annotations

from typing import Any

from langgraph.graph import END


def has_tool_calls(state: Any, tool_calls_key: str = "tool_calls") -> bool:
    if isinstance(state, dict):
        calls = state.get(tool_calls_key, [])
    else:
        calls = getattr(state, tool_calls_key, [])
    return bool(calls)


def route_tools_or_end(
    state: Any,
    tools_node_name: str = "tools",
    tool_calls_key: str = "tool_calls",
) -> str:
    if has_tool_calls(state, tool_calls_key=tool_calls_key):
        return tools_node_name
    return END
