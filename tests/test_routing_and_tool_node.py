import unittest

from langgraph.graph import END

from agent_graph.core.routing import has_tool_calls, route_tools_or_end
from agent_graph.core.tool_node import ToolExecutionNode
from agent_graph.core.tool_registry import ToolRegistry
from agent_graph.core.tool_types import ToolResult


class EchoTool:
    name = "echo"

    def invoke(self, args):
        return ToolResult(status="ok", content=str(args.get("query", "")), raw=args)


class RoutingAndNodeTest(unittest.TestCase):
    def test_has_tool_calls(self):
        self.assertTrue(has_tool_calls({"tool_calls": [{"name": "echo", "args": {}}]}))
        self.assertFalse(has_tool_calls({"tool_calls": []}))

    def test_route_tools_or_end(self):
        self.assertEqual(route_tools_or_end({"tool_calls": [{"name": "echo", "args": {}}]}), "tools")
        self.assertEqual(route_tools_or_end({"tool_calls": []}), END)

    def test_tool_execution_node(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        node = ToolExecutionNode(registry)

        out = node({"tool_calls": [{"name": "echo", "args": {"query": "hello"}}]})

        self.assertEqual(out["tool_calls"], [])
        self.assertEqual(len(out["tool_results"]), 1)
        self.assertEqual(out["tool_results"][0]["status"], "ok")
        self.assertEqual(out["tool_results"][0]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
