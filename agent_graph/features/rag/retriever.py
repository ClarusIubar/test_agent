"""Chroma 벡터 DB 기반 retriever 팩토리."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, create_retriever_tool as _create_retriever_tool
from langchain_openai import OpenAIEmbeddings


@dataclass
class RetrieverBundle:
    """retriever 원본 객체와 LangChain 도구를 함께 보관한다."""

    retriever: Any
    tool: BaseTool


def create_retriever_tool(
    db_path: str,
    collection_name: str,
    embedding_model: str = "text-embedding-3-small",
    k: int = 3,
    tool_description: str = (
        "Use this tool to search information from the document store. "
        "Input should be a search query."
    ),
) -> RetrieverBundle:
    """Chroma 벡터 DB로부터 retriever 도구를 생성한다.

    Args:
        db_path: Chroma 영속화 디렉터리 경로.
        collection_name: Chroma 컬렉션 이름.
        embedding_model: OpenAI 임베딩 모델명.
        k: 검색 결과 상위 k개.
        tool_description: 도구 설명 (LLM이 도구 선택 시 참조).

    Returns:
        RetrieverBundle — .retriever(원본)과 .tool(LangChain BaseTool) 포함.
    """
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(model=embedding_model),
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    tool = _create_retriever_tool(
        retriever,
        name="pdf_search",
        description=tool_description,
    )
    return RetrieverBundle(retriever=retriever, tool=tool)
