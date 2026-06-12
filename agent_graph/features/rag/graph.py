"""RAG 에이전트 그래프 빌더."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .edges import route_to_search


def build_rag_agent_graph(
    *,
    state_schema: Any,
    input_schema: Any,
    output_schema: Any,
    chatbot_node: Any,
    web_tool_node: Any,
    rag_tool_node: Any,
    context_organizer_node: Any,
    transform_query_node: Any,
    generate_node: Any,
    decide_to_generate_fn: Any,
    check_hallucinations_fn: Any,
) -> CompiledStateGraph:
    """웹검색 + RAG 조건부 엣지가 포함된 에이전트 그래프를 빌드한다.

    그래프 흐름:
        START → chatbot
        chatbot --route_to_search--> web_search | rag_search | END
        web_search  → context_organizer
        rag_search  → context_organizer
        context_organizer --decide_to_generate--> transform_query | generate
        transform_query → chatbot
        generate --check_hallucinations--> generate(재시도) | END
    """
    builder = StateGraph(state_schema, input_schema=input_schema, output_schema=output_schema)

    # 노드 등록
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("web_search", web_tool_node)
    builder.add_node("rag_search", rag_tool_node)
    builder.add_node("context_organizer", context_organizer_node)
    builder.add_node("transform_query", transform_query_node)
    builder.add_node("generate", generate_node)

    # 진입점
    builder.add_edge(START, "chatbot")

    # chatbot → 분기
    builder.add_conditional_edges(
        "chatbot",
        route_to_search,
        {
            "web_search": "web_search",
            "rag_search": "rag_search",
            END: END,
        },
    )

    # 검색 노드 → context_organizer
    builder.add_edge("web_search", "context_organizer")
    builder.add_edge("rag_search", "context_organizer")

    # context_organizer → 관련성 평가 분기
    builder.add_conditional_edges(
        "context_organizer",
        decide_to_generate_fn,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    # transform_query → chatbot (질문 재작성 후 재검색)
    builder.add_edge("transform_query", "chatbot")

    # generate → 환각 체크 분기
    builder.add_conditional_edges(
        "generate",
        check_hallucinations_fn,
        {
            "not supported": "generate",
            "support": END,
        },
    )

    return builder.compile()
