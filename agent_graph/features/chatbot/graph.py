from agent_graph.core.assembly import build_graph
from agent_graph.features.chatbot.node import chatbot


graph = build_graph(
    nodes={"chatbot": chatbot},
    entry_point="chatbot",
    finish_point="chatbot",
)