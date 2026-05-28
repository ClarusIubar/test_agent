from agent_graph.core.llm import get_llm
from agent_graph.core.state import OverallState


def chatbot(state: OverallState) -> OverallState:
    question = state.question
    response = get_llm().invoke(question)
    answer = response.text
    return OverallState(question=question, answer=answer, message=[question, answer])