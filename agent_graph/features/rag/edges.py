"""RAG 파이프라인 조건부 엣지 함수 모음."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# route_to_search: chatbot → web_search | rag_search | END
# ---------------------------------------------------------------------------

def route_to_search(state: Any) -> str:
    """chatbot 노드 이후 어느 검색 노드로 분기할지 결정한다."""
    messages = state.messages if hasattr(state, "messages") else state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        return END

    tool_name = tool_calls[0].get("name", "") if isinstance(tool_calls[0], dict) else tool_calls[0].name
    if tool_name == "tavily_search":
        return "web_search"
    if tool_name.startswith("pdf_search"):
        return "rag_search"
    return END


# ---------------------------------------------------------------------------
# decide_to_generate: context_organizer → transform_query | generate
# ---------------------------------------------------------------------------

class _Grade(BaseModel):
    """관련성 확인을 위한 점수 스키마."""

    binary_score: str = Field(description="문서가 질문과 관련이 있는지 여부, 'yes' 또는 'no'")


def create_decide_to_generate(llm: Any):
    """관련성 평가 엣지 함수를 반환한다."""

    grader = llm.with_structured_output(_Grade)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 평가자입니다.
문서가 사용자 질문과 관련된 키워드나 의미를 포함하고 있다면 관련성이 있다고 평가하세요.
엄격한 테스트일 필요는 없습니다. 목표는 잘못된 검색 결과를 필터링하는 것입니다.
문서가 질문과 관련이 있는지를 나타내는 'yes' 또는 'no'의 이진 점수를 제공하세요.""",
            ),
            ("user", "검색된 문서: {context} \n\n 사용자 질문: {question} \n\n 관련성 점수:"),
        ]
    )
    chain = prompt | grader

    def decide_to_generate(state: Any) -> str:
        print("----- ASSESS GRADED DOCUMENTS -----")
        retry_num = state.retry_num if hasattr(state, "retry_num") else state.get("retry_num", 0)
        if retry_num >= 3:
            return "generate"

        question = state.question if hasattr(state, "question") else state.get("question", "")
        context = state.context if hasattr(state, "context") else state.get("context", "")

        if not context or not question:
            print("---ERROR: Missing context or question, defaulting to generate---")
            return "generate"

        score = chain.invoke({"question": question, "context": context})
        if score.binary_score == "no":
            print("---DECISION: RETRIEVED DOCUMENT NOT RELEVANT, TRANSFORM QUERY---")
            return "transform_query"
        print("---DECISION: GENERATE---")
        return "generate"

    return decide_to_generate


# ---------------------------------------------------------------------------
# check_hallucinations: generate → support | not supported
# ---------------------------------------------------------------------------

class _GradeHallucinations(BaseModel):
    """생성된 답변의 환각 여부를 판단하기 위한 점수 스키마."""

    binary_score: str = Field(description="답변이 사실에 근거하고 있는지 여부, 'yes' 또는 'no'")


def create_check_hallucinations(llm: Any):
    """환각 평가 엣지 함수를 반환한다."""

    structured_llm = llm.with_structured_output(_GradeHallucinations)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 LLM이 생성한 답변이 검색된 사실들에 근거하고 있는지 평가하는 평가자입니다.
'yes' 또는 'no'의 이진 점수를 제공하세요. 'yes'는 답변이 사실들에 근거하고 있음을 의미합니다.""",
            ),
            (
                "user",
                "질문: {question} \n\n 사실 집합: \n\n {context} \n\n LLM 생성 답변: {generation}",
            ),
        ]
    )
    chain = prompt | structured_llm

    def check_hallucinations(state: Any) -> str:
        print("----- CHECK HALLUCINATIONS -----")
        question = state.question if hasattr(state, "question") else state.get("question", "")
        context = state.context if hasattr(state, "context") else state.get("context", "")
        answer = state.answer if hasattr(state, "answer") else state.get("answer", "")

        score = chain.invoke({"question": question, "context": context, "generation": answer})
        if score.binary_score == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            return "support"
        print("---DECISION: GENERATION NOT GROUNDED, RE-TRY---")
        return "not supported"

    return check_hallucinations
