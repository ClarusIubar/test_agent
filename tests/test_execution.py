import asyncio
import unittest

from agent_graph.core.execution import run_ainvoke, run_astream, run_invoke, run_stream


class FakeGraph:
    def invoke(self, payload):
        return {"payload": payload, "mode": "invoke"}

    async def ainvoke(self, payload):
        return {"payload": payload, "mode": "ainvoke"}

    def stream(self, payload, stream_mode=None):
        yield {"payload": payload, "mode": "stream", "stream_mode": stream_mode}

    async def astream(self, payload, stream_mode=None):
        yield {"payload": payload, "mode": "astream", "stream_mode": stream_mode}


class ExecutionTest(unittest.TestCase):
    def test_run_invoke(self):
        result = run_invoke(FakeGraph(), {"q": "x"})
        self.assertEqual(result["mode"], "invoke")

    def test_run_stream(self):
        events = list(run_stream(FakeGraph(), {"q": "x"}, stream_mode="values"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["stream_mode"], "values")

    def test_run_ainvoke(self):
        result = asyncio.run(run_ainvoke(FakeGraph(), {"q": "x"}))
        self.assertEqual(result["mode"], "ainvoke")

    def test_run_astream(self):
        async def collect():
            items = []
            async for event in run_astream(FakeGraph(), {"q": "x"}, stream_mode="messages"):
                items.append(event)
            return items

        items = asyncio.run(collect())
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["mode"], "astream")


if __name__ == "__main__":
    unittest.main()
