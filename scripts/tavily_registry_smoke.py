from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_graph.core import TavilyAdapter, ToolRegistry


def main() -> None:
    registry = ToolRegistry()
    registry.register(TavilyAdapter(max_results=2))

    print("registered:", registry.names())

    # This requires TAVILY_API_KEY in environment.
    result = registry.invoke("tavily_search", {"query": "What is LangGraph?"})
    print("status:", result.status)
    print("has_error:", bool(result.error))
    print("content_preview:", result.content[:200])


if __name__ == "__main__":
    main()
