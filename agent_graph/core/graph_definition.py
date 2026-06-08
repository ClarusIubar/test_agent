from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .routing import route_tools_or_end
from .tool_node import ToolExecutionNode
from .tool_registry import ToolRegistry


def build_agent_graph(
    *,
    state_schema: Any,
    input_schema: Any,
    output_schema: Any,
    chatbot_node: Any,
    tool_registry: ToolRegistry | None = None,
    chatbot_node_name: str = "chatbot",
    tools_node_name: str = "tools",
) -> CompiledStateGraph:
    graph_builder = StateGraph(
        state_schema,
        input_schema=input_schema,
        output_schema=output_schema,
    )

    graph_builder.add_node(chatbot_node_name, cast(Any, chatbot_node))
    graph_builder.add_edge(START, chatbot_node_name)

    if tool_registry is None:
        graph_builder.add_edge(chatbot_node_name, END)
        return graph_builder.compile()

    tool_node = ToolExecutionNode(tool_registry)
    graph_builder.add_node(tools_node_name, cast(Any, tool_node))
    graph_builder.add_conditional_edges(
        chatbot_node_name,
        lambda state: route_tools_or_end(
            state,
            tools_node_name=tools_node_name,
            tool_calls_key="tool_calls",
        ),
        {
            tools_node_name: tools_node_name,
            END: END,
        },
    )
    graph_builder.add_edge(tools_node_name, chatbot_node_name)
    return graph_builder.compile()
