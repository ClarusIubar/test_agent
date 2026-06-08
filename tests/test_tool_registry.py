import unittest

from agent_graph.core.tool_registry import ToolRegistry
from agent_graph.core.tool_types import ToolResult


class DummyTool:
    name = "dummy"

    def invoke(self, args: dict[str, object]) -> ToolResult:
        return ToolResult(status="ok", content=str(args.get("query", "")), raw=args, error=None)


class ToolRegistryTest(unittest.TestCase):
    def test_register_and_invoke(self) -> None:
        registry = ToolRegistry()
        registry.register(DummyTool())

        result = registry.invoke("dummy", {"query": "hello"})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.content, "hello")
        self.assertIsNone(result.error)

    def test_unknown_tool_raises_key_error(self) -> None:
        registry = ToolRegistry()

        with self.assertRaises(KeyError):
            registry.get("missing")


if __name__ == "__main__":
    unittest.main()
