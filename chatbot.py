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
        # Simple convention for routing tests and CLI demos:
        # questions prefixed by "search:" request one tavily tool call.
        next_tool_calls = []
        if question.lower().startswith("search:") and not state.tool_results:
            next_tool_calls = [{"name": "tavily_search", "args": {"query": question[7:].strip()}}]

        return OverallState(
            question=question,
            answer=answer,
            message=[question, answer],
            tool_calls=next_tool_calls,
            tool_results=state.tool_results,
        )

    return chatbot


chatbot = create_chatbot_node()