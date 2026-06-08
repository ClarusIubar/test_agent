from operator import add
from typing import Annotated, Any

from pydantic import BaseModel, Field


class InputState(BaseModel):
    question: str


class OutputState(BaseModel):
    answer: str


class OverallState(BaseModel):
    question: str
    answer: str = ""
    message: Annotated[list[str], add] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: Annotated[list[dict[str, Any]], add] = Field(default_factory=list)