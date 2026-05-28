from operator import add
from typing import Annotated

from pydantic import BaseModel, Field


class InputState(BaseModel):
    question: str


class OutputState(BaseModel):
    answer: str


class OverallState(BaseModel):
    question: str
    answer: str = ""
    message: Annotated[list[str], add] = Field(default_factory=list)