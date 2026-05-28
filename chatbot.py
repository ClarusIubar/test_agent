from collections.abc import Callable
from typing import Protocol

from llm import get_llm
from state import OverallState


class SupportsTextResponse(Protocol):
    text: str


class SupportsInvoke(Protocol):
    def invoke(self, input: str) -> SupportsTextResponse: ...


def create_chatbot_node(llm: SupportsInvoke | None = None) -> Callable[[OverallState], OverallState]:
    active_llm = llm or get_llm()

    def chatbot(state: OverallState) -> OverallState:
        question = state.question
        response = active_llm.invoke(question)
        answer = response.text
        return OverallState(question=question, answer=answer, message=[question, answer])

    return chatbot


chatbot = create_chatbot_node()