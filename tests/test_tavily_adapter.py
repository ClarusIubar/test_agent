import unittest

from agent_graph.core.tavily_adapter import TavilyAdapter


class TavilyAdapterTest(unittest.TestCase):
    def test_invoke_returns_normalized_result_shape(self) -> None:
        adapter = TavilyAdapter(max_results=1)
        result = adapter.invoke({"query": "langgraph"})

        self.assertIn(result.status, ("ok", "error"))
        self.assertTrue(hasattr(result, "content"))
        self.assertTrue(hasattr(result, "raw"))
        self.assertTrue(hasattr(result, "error"))


if __name__ == "__main__":
    unittest.main()
