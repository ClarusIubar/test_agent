import sys
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from chatbot import chatbot
from state import InputState, OutputState, OverallState


def build_graph(
    chatbot_node: Any = None,
) -> CompiledStateGraph:
    active_chatbot = chatbot_node or chatbot
    graph_builder = StateGraph(
        OverallState,
        input_schema=InputState,
        output_schema=OutputState,
    )
    graph_builder.add_node("chatbot", cast(Any, active_chatbot))
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


graph = build_graph()


def invoke_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
) -> OutputState:
    active_graph = compiled_graph or graph
    result = active_graph.invoke(InputState(question=question))
    return OutputState.model_validate(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python graph.py \"your question\"")

    output = invoke_graph(" ".join(sys.argv[1:]))
    print(output.answer)