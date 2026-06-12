"""RAG 파이프라인 노드 팩토리 함수 모음."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate


def create_web_tool_node(tavily_tool: Any):
    """Tavily 웹검색 노드를 반환한다."""

    def web_search(state: Any) -> dict:
        print("----- [WEB SEARCH] -----")
        messages = state.messages if hasattr(state, "messages") else state.get("messages", [])
        last_message = messages[-1]
        tool_call = last_message.tool_calls[0]
        query = tool_call["args"].get("query", "")
        tool_call_id = tool_call["id"]

        raw = tavily_tool.invoke({"query": query})
        if isinstance(raw, list):
            context = "\n".join(
                f"[{i+1}] {item.get('content', '')}" for i, item in enumerate(raw)
            )
        elif isinstance(raw, str):
            context = raw
        else:
            context = json.dumps(raw, ensure_ascii=False)

        tool_message = ToolMessage(
            content=context,
            name="tavily_search",
            tool_call_id=tool_call_id,
        )
        return {
            "context": context,
            "messages": [tool_message],
            "routing_trace": ["web_search"],
        }

    return web_search


def create_rag_tool_node(retriever: Any):
    """Chroma 문서 검색 노드를 반환한다."""

    def rag_search(state: Any) -> dict:
        print("----- [RAG SEARCH] -----")
        messages = state.messages if hasattr(state, "messages") else state.get("messages", [])
        last_message = messages[-1]
        tool_call = last_message.tool_calls[0]
        query = tool_call["args"].get("query", "")
        tool_call_id = tool_call["id"]

        docs = retriever.invoke(query)
        context = ""
        for doc in docs:
            page_num = doc.metadata.get("page", 0) + 1
            context += f"Page {page_num}: {doc.page_content}\n"

        tool_message = ToolMessage(
            content=context,
            name="pdf_search",
            tool_call_id=tool_call_id,
        )
        return {
            "context": context,
            "messages": [tool_message],
            "routing_trace": ["rag_search"],
        }

    return rag_search


def create_context_organizer(llm: Any):
    """검색 결과 텍스트 정리 노드를 반환한다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 검색증강생성(RAG)을 위한 검색문서를 정리하는 전문가입니다.
아래의 검색된 결과 문서를 확인하고, LLM이 해당 문서를 정리된 형태로 참고할 수 있도록
문서의 불필요한 공백 등을 삭제하거나 정렬을 다시하여 정리된 형태로 반환해주세요.
내용을 삭제하는 것을 최소로 합니다. 페이지 번호 정보를 절대 삭제하지 마세요.""",
            ),
            ("user", "검색 결과: {context}"),
        ]
    )
    chain = prompt | llm

    def context_organizer(state: Any) -> dict:
        print("----- [CONTEXT ORGANIZER] -----")
        context = state.context if hasattr(state, "context") else state.get("context", "")
        result = chain.invoke({"context": context})
        organized = result.content
        return {
            "context": organized,
            "messages": [AIMessage(content=organized)],
            "routing_trace": ["context_organizer"],
        }

    return context_organizer


def create_transform_query(llm: Any):
    """질문 재작성 노드를 반환한다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 질문을 다시 작성하는 전문가입니다. 입력된 질문을 벡터 저장소 검색에 최적화된 더 나은 버전으로 변환하세요.
입력을 살펴보고 질문의 핵심적인 의미와 의도를 파악하여 개선된 질문을 만들어주세요.""",
            ),
            (
                "user",
                "다음은 초기 질문입니다: \n\n {question} \n 한국어로 개선된 질문을 작성해주세요.",
            ),
        ]
    )
    chain = prompt | llm

    def transform_query(state: Any) -> dict:
        print("----- [TRANSFORM QUERY] -----")
        question = state.question if hasattr(state, "question") else state.get("question", "")
        retry_num = state.retry_num if hasattr(state, "retry_num") else state.get("retry_num", 0)
        result = chain.invoke({"question": question})
        better_question = result.content
        return {
            "question": better_question,
            "messages": [HumanMessage(content=better_question)],
            "retry_num": retry_num + 1,
            "routing_trace": ["transform_query"],
        }

    return transform_query


def create_generate(llm: Any):
    """답변 생성 노드를 반환한다."""

    normal_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 질문-답변 업무를 수행하는 어시스턴트입니다. 검색된 컨텍스트를 사용하여 질문에 답변하세요.
답변을 모르는 경우, 모른다고 말하세요.
답변은 간결하게 작성하고, 반드시 답변의 출처(페이지 번호)를 함께 명시해주세요.""",
            ),
            ("user", "질문: {question} \n\n검색 결과: {context} \n\n답변:"),
        ]
    )
    fallback_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 검색된 문서를 통해 해결할 수 있는 질문을 추출하는 어시스턴트입니다.
사용자가 해결하고자 한 질문이 있었으나 검색 컨텍스트가 충분하지 않은 상황이므로,
주어진 검색 결과 내에서 답변할 수 있는 질문을 새롭게 작성해 나열하세요.
사용자에게 질문에 대한 답변을 하지 못함에 양해를 구하고, 다른 질문의 기회와 선택지를 제공하는 친절한 가이드를 하세요.""",
            ),
            ("user", "질문: {question} \n\n검색 결과: {context} \n\n답변:"),
        ]
    )

    def generate(state: Any) -> dict:
        print("----- [GENERATE] -----")
        question = state.question if hasattr(state, "question") else state.get("question", "")
        context = state.context if hasattr(state, "context") else state.get("context", "")
        retry_num = state.retry_num if hasattr(state, "retry_num") else state.get("retry_num", 0)

        prompt = fallback_prompt if retry_num >= 3 else normal_prompt
        chain = prompt | llm
        result = chain.invoke({"question": question, "context": context})
        return {
            "answer": result.content,
            "messages": [AIMessage(content=result.content)],
            "routing_trace": ["generate"],
        }

    return generate
