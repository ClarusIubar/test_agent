from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph

from agent_graph.core.graph import create_graph_builder

StateNode = Any
GraphEdge = tuple[str, str]


class GraphSpecError(ValueError):
    pass


@dataclass(frozen=True)
class GraphSpec:
    nodes: Mapping[str, StateNode]
    entry_point: str
    finish_point: str | None = None
    edges: Sequence[GraphEdge] = field(default_factory=tuple)

    @property
    def resolved_finish_point(self) -> str:
        return self.finish_point or self.entry_point

    @classmethod
    def single_node(cls, *, node_name: str, node_handler: StateNode) -> "GraphSpec":
        return cls(nodes={node_name: node_handler}, entry_point=node_name)


def _validate_graph_spec(spec: GraphSpec) -> None:
    if not spec.nodes:
        raise GraphSpecError("nodes must not be empty")

    node_names = set(spec.nodes)

    if spec.entry_point not in node_names:
        raise GraphSpecError(f"entry_point '{spec.entry_point}' must exist in nodes")

    if spec.resolved_finish_point not in node_names:
        raise GraphSpecError(
            f"finish_point '{spec.resolved_finish_point}' must exist in nodes"
        )

    for start_node, end_node in spec.edges:
        if start_node not in node_names:
            raise GraphSpecError(f"edge start '{start_node}' must exist in nodes")
        if end_node not in node_names:
            raise GraphSpecError(f"edge end '{end_node}' must exist in nodes")


def build_graph(spec: GraphSpec) -> CompiledStateGraph:
    _validate_graph_spec(spec)

    builder = create_graph_builder()

    for node_name, node_handler in spec.nodes.items():
        builder.add_node(node_name, node_handler)

    builder.add_edge(START, spec.entry_point)

    for start_node, end_node in spec.edges:
        builder.add_edge(start_node, end_node)

    builder.add_edge(spec.resolved_finish_point, END)
    return builder.compile()
