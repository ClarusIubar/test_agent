import sys
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from agent_graph.core import (
    TavilyAdapter,
    ToolRegistry,
    build_agent_graph,
    run_ainvoke,
    run_astream,
    run_invoke,
    run_stream,
)
from chatbot import chatbot
from state import InputState, OutputState, OverallState


def build_graph(
    chatbot_node: Any = None,
    tool_registry: ToolRegistry | None = None,
) -> CompiledStateGraph:
    active_chatbot = chatbot_node or chatbot
    return build_agent_graph(
        state_schema=OverallState,
        input_schema=InputState,
        output_schema=OutputState,
        chatbot_node=active_chatbot,
        tool_registry=tool_registry,
    )


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(TavilyAdapter(max_results=3))
    return registry


graph = build_graph(tool_registry=build_default_tool_registry())


def invoke_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
) -> OutputState:
    active_graph = compiled_graph or graph
    result = run_invoke(active_graph, InputState(question=question))
    return OutputState.model_validate(result)


async def ainvoke_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
) -> OutputState:
    active_graph = compiled_graph or graph
    result = await run_ainvoke(active_graph, InputState(question=question))
    return OutputState.model_validate(result)


def stream_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
    stream_mode: str | None = None,
) -> Iterator[Any]:
    active_graph = compiled_graph or graph
    return run_stream(active_graph, InputState(question=question), stream_mode=stream_mode)


async def astream_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
    stream_mode: str | None = None,
) -> AsyncIterator[Any]:
    active_graph = compiled_graph or graph
    async for event in run_astream(active_graph, InputState(question=question), stream_mode=stream_mode):
        yield event


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python graph.py \"your question\"")

    output = invoke_graph(" ".join(sys.argv[1:]))
    print(output.answer)