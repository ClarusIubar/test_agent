from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    INVOKE = "invoke"
    AINVOKE = "ainvoke"
    STREAM = "stream"
    ASTREAM = "astream"


def run_invoke(compiled_graph: Any, payload: Any) -> Any:
    return compiled_graph.invoke(payload)


async def run_ainvoke(compiled_graph: Any, payload: Any) -> Any:
    return await compiled_graph.ainvoke(payload)


def run_stream(
    compiled_graph: Any,
    payload: Any,
    stream_mode: str | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> Iterator[Any]:
    if stream_mode is None:
        iterator = compiled_graph.stream(payload)
    else:
        iterator = compiled_graph.stream(payload, stream_mode=stream_mode)

    for event in iterator:
        if on_event is not None:
            on_event(event)
        yield event


async def run_astream(
    compiled_graph: Any,
    payload: Any,
    stream_mode: str | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> AsyncIterator[Any]:
    if stream_mode is None:
        iterator = compiled_graph.astream(payload)
    else:
        iterator = compiled_graph.astream(payload, stream_mode=stream_mode)

    async for event in iterator:
        if on_event is not None:
            on_event(event)
        yield event
