import os
import sys
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.graph.state import CompiledStateGraph

from agent_graph.core import (
    run_ainvoke,
    run_astream,
    run_invoke,
    run_stream,
)
from agent_graph.features.rag import (
    build_rag_agent_graph,
    create_check_hallucinations,
    create_context_organizer,
    create_decide_to_generate,
    create_generate,
    create_rag_tool_node,
    create_retriever_tool,
    create_retriever_tools_from_config,
    create_transform_query,
    create_web_tool_node,
)
from chatbot import create_chatbot_node
from llm import get_llm
from state import InputState, OutputState, OverallState


def build_default_graph() -> CompiledStateGraph:
    load_dotenv()
    llm = get_llm()

    # Tavily 웹검색 도구
    tavily_tool = TavilySearch(max_results=3)

    # Chroma 문서 검색 도구
    # 1) DOC_SOURCES_JSON 이 있으면 다중 소스 구성 사용
    # 2) 없으면 CHROMA_DB_PATH/CHROMA_COLLECTION_NAME 단일 소스 fallback
    doc_sources_json = os.environ.get("DOC_SOURCES_JSON", "").strip()
    if doc_sources_json:
        sources = json.loads(doc_sources_json)
        bundles_by_tool_name = create_retriever_tools_from_config(sources)
    else:
        db_path = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
        collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "korean_pdf")
        bundle = create_retriever_tool(
            db_path=db_path,
            collection_name=collection_name,
            tool_name="pdf_search_default",
        )
        bundles_by_tool_name = {bundle.tool_name: bundle}

    retrievers_by_tool_name = {
        tool_name: bundle.retriever for tool_name, bundle in bundles_by_tool_name.items()
    }
    rag_tools = [bundle.tool for bundle in bundles_by_tool_name.values()]

    # chatbot 노드: 웹검색 + (1개 이상) 문서검색 도구를 바인딩
    chatbot_node = create_chatbot_node(llm=llm, tools=[tavily_tool, *rag_tools])

    return build_rag_agent_graph(
        state_schema=OverallState,
        input_schema=InputState,
        output_schema=OutputState,
        chatbot_node=chatbot_node,
        web_tool_node=create_web_tool_node(tavily_tool),
        rag_tool_node=create_rag_tool_node(retrievers_by_tool_name),
        context_organizer_node=create_context_organizer(llm),
        transform_query_node=create_transform_query(llm),
        generate_node=create_generate(llm),
        decide_to_generate_fn=create_decide_to_generate(llm),
        check_hallucinations_fn=create_check_hallucinations(llm),
    )


graph = build_default_graph()


def invoke_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
) -> OutputState:
    active_graph = compiled_graph or graph
    result = run_invoke(active_graph, InputState(question=question))
    return OutputState.model_validate(result)


async def ainvoke_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
) -> OutputState:
    active_graph = compiled_graph or graph
    result = await run_ainvoke(active_graph, InputState(question=question))
    return OutputState.model_validate(result)


def stream_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
    stream_mode: str | None = None,
) -> Iterator[Any]:
    active_graph = compiled_graph or graph
    return run_stream(active_graph, InputState(question=question), stream_mode=stream_mode)


async def astream_graph(
    question: str,
    compiled_graph: CompiledStateGraph | None = None,
    stream_mode: str | None = None,
) -> AsyncIterator[Any]:
    active_graph = compiled_graph or graph
    async for event in run_astream(active_graph, InputState(question=question), stream_mode=stream_mode):
        yield event


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python graph.py \"your question\"")

    output = invoke_graph(" ".join(sys.argv[1:]))
    print(output.answer)