from agent_graph.core.assembly import GraphSpec, build_graph
from agent_graph.features.chatbot.node import chatbot


graph = build_graph(GraphSpec.single_node(node_name="chatbot", node_handler=chatbot))