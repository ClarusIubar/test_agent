from operator import add
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class InputState(BaseModel):
    question: str


class OutputState(BaseModel):
    answer: str
    routing_trace: list[str] = Field(default_factory=list)


class OverallState(BaseModel):
    question: str
    answer: str = ""
    message: Annotated[list[str], add] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: Annotated[list[dict[str, Any]], add] = Field(default_factory=list)
    # RAG 통합 필드
    messages: Annotated[list[BaseMessage], add] = Field(default_factory=list)
    context: str = ""
    retry_num: int = 0
    routing_trace: Annotated[list[str], add] = Field(default_factory=list)