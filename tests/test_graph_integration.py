import unittest

from agent_graph.core.graph_definition import build_agent_graph
from agent_graph.core.tool_registry import ToolRegistry
from agent_graph.core.tool_types import ToolResult
from state import InputState, OutputState, OverallState


class EchoTool:
    name = "echo"

    def invoke(self, args):
        return ToolResult(status="ok", content=str(args.get("query", "")), raw=args)


def make_chatbot_node():
    def chatbot(state: OverallState) -> OverallState:
        # Emit one tool call first, then finish after tool result is produced.
        if not state.tool_results:
            return OverallState(
                question=state.question,
                answer="tool requested",
                message=[state.question, "tool requested"],
                tool_calls=[{"name": "echo", "args": {"query": state.question}}],
                tool_results=state.tool_results,
            )

        return OverallState(
            question=state.question,
            answer=f"done:{state.tool_results[-1]['content']}",
            message=[state.question, f"done:{state.tool_results[-1]['content']}"],
            tool_calls=[],
            tool_results=state.tool_results,
        )

    return chatbot


class GraphIntegrationTest(unittest.TestCase):
    def test_tool_loop_routing_and_results(self):
        registry = ToolRegistry()
        registry.register(EchoTool())

        graph = build_agent_graph(
            state_schema=OverallState,
            input_schema=InputState,
            output_schema=OutputState,
            chatbot_node=make_chatbot_node(),
            tool_registry=registry,
        )

        result = graph.invoke(InputState(question="hello"))

        self.assertIn("answer", result)
        self.assertEqual(result["answer"], "done:hello")


if __name__ == "__main__":
    unittest.main()
