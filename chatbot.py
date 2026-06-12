from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm import get_llm
from state import OverallState


ROUTING_POLICY = """당신은 라우팅 담당 에이전트입니다.
질문을 받으면 반드시 도구를 먼저 호출한 뒤 답변 생성 단계로 넘기세요.
도구 선택 규칙:
1) 한국어 맞춤법/표준어/음운변동/발음 규칙 관련 질문은 pdf_search 사용
2) 최신 정보, 일반 상식, 기술/뉴스/동향 질문은 tavily_search 사용
3) 애매하면 tavily_search 우선
직접 답변만 하고 도구 호출을 생략하지 마세요."""


def _pick_tool_name(tools: list[Any], prefix: str) -> str | None:
    for tool in tools:
        name = getattr(tool, "name", "")
        if isinstance(name, str) and name.startswith(prefix):
            return name
    return None


def _parse_forced_route(question: str) -> tuple[str | None, str]:
    lowered = question.lower()
    # Preferred explicit prefixes for UI integration.
    web_prefixes = (
        "search/web:",
        "search/web ",
        "search/web,",
        # Backward-compatible prefixes
        "search:",
        "web:",
    )
    docs_prefixes = (
        "search/docs:",
        "search/docs ",
        "search/docs,",
        # Backward-compatible prefixes
        "docs:",
        "doc:",
        "rag:",
    )

    for prefix in web_prefixes:
        if lowered.startswith(prefix):
            return "web", question[len(prefix) :].strip()

    for prefix in docs_prefixes:
        if lowered.startswith(prefix):
            return "docs", question[len(prefix) :].strip()

    return None, question


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
    tool_list = tools or []
    web_tool_name = _pick_tool_name(tool_list, "tavily_search")
    docs_tool_name = _pick_tool_name(tool_list, "pdf_search")

    def chatbot(state: OverallState) -> dict:
        print("----- [CHATBOT] -----")
        forced_route, cleaned_question = _parse_forced_route(state.question)
        messages = list(state.messages)
        if not messages:
            messages = [HumanMessage(content=cleaned_question)]

        # Prefix 기반 명시적 라우팅은 LLM 자동 라우팅보다 항상 우선한다.
        if forced_route == "web" and web_tool_name:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": web_tool_name,
                        "args": {"query": cleaned_question},
                        "id": "forced_web_route",
                        "type": "tool_call",
                    }
                ],
            )
            return {
                "messages": [response],
                "question": cleaned_question,
                "answer": "",
                "routing_trace": ["chatbot", "forced_prefix:web"],
            }

        if forced_route == "docs" and docs_tool_name:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": docs_tool_name,
                        "args": {"query": cleaned_question},
                        "id": "forced_docs_route",
                        "type": "tool_call",
                    }
                ],
            )
            return {
                "messages": [response],
                "question": cleaned_question,
                "answer": "",
                "routing_trace": ["chatbot", "forced_prefix:docs"],
            }

        response = bound_llm.invoke([SystemMessage(content=ROUTING_POLICY), *messages])
        tool_calls = getattr(response, "tool_calls", None) or []
        return {
            "messages": [response],
            "question": cleaned_question,
            "answer": "" if tool_calls else getattr(response, "content", ""),
            "routing_trace": ["chatbot"],
        }

    return chatbot
