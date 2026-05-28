from langgraph.graph import StateGraph

from agent_graph.core.state import InputState, OutputState, OverallState


def create_graph_builder() -> StateGraph:
    return StateGraph(
        OverallState,
        input_schema=InputState,
        output_schema=OutputState,
    )
