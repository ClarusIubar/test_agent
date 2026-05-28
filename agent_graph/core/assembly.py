from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph

from agent_graph.core.graph import create_graph_builder

StateNode = Any
GraphEdge = tuple[str, str]


def build_graph(
    *,
    nodes: Mapping[str, StateNode],
    entry_point: str,
    finish_point: str,
    edges: Sequence[GraphEdge] = (),
) -> CompiledStateGraph:
    builder = create_graph_builder()

    for node_name, node_handler in nodes.items():
        builder.add_node(node_name, node_handler)

    builder.add_edge(START, entry_point)

    for start_node, end_node in edges:
        builder.add_edge(start_node, end_node)

    builder.add_edge(finish_point, END)
    return builder.compile()