from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from state import OverallState


ROUTING_POLICY = """당신은 라우팅 담당 에이전트입니다.
질문을 받으면 반드시 도구를 먼저 호출한 뒤 답변 생성 단계로 넘기세요.
도구 선택 규칙:
1) 한국어 맞춤법/표준어/음운변동/발음 규칙 관련 질문은 pdf_search 사용
2) 최신 정보, 일반 상식, 기술/뉴스/동향 질문은 tavily_search 사용
3) 애매하면 tavily_search 우선
직접 답변만 하고 도구 호출을 생략하지 마세요."""


def create_chatbot_node(
    llm: Any | None = None,
    tools: list[Any] | None = None,
) -> Callable[[OverallState], dict]:
    """LLM에 도구를 바인딩한 chatbot 노드 함수를 반환한다.

    Args:
        llm: ChatOpenAI 인스턴스. None이면 get_llm()으로 초기화.
        tools: 바인딩할 도구 목록. None이거나 빈 리스트면 도구 없이 실행.

    Returns:
        LangGraph 노드 함수 — OverallState를 받아 dict를 반환.
    """
    active_llm = llm or get_llm()
    bound_llm = active_llm.bind_tools(tools) if tools else active_llm

    def chatbot(state: OverallState) -> dict:
        print("----- [CHATBOT] -----")
        messages = list(state.messages)
        if not messages:
            messages = [HumanMessage(content=state.question)]

        response = bound_llm.invoke([SystemMessage(content=ROUTING_POLICY), *messages])
        tool_calls = getattr(response, "tool_calls", None) or []
        return {
            "messages": [response],
            "question": state.question,
            "answer": "" if tool_calls else getattr(response, "content", ""),
            "routing_trace": ["chatbot"],
        }

    return chatbot
