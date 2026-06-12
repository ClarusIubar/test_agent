from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable when running as a script.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graph import build_default_graph
from state import InputState, OutputState


def run_eval(question: str, expected_route: str) -> dict[str, Any]:
    graph = build_default_graph()

    node_path: list[str] = []
    for event in graph.stream(InputState(question=question), stream_mode="updates"):
        for node_name in event.keys():
            node_path.append(node_name)

    output_raw = graph.invoke(InputState(question=question))
    output = OutputState.model_validate(output_raw)

    selected_route = "none"
    if "web_search" in node_path:
        selected_route = "web_search"
    if "rag_search" in node_path:
        selected_route = "rag_search"

    is_route_ok = selected_route == expected_route
    has_answer = bool((output.answer or "").strip())

    answer = output.answer or ""
    quality_ok = True
    quality_reason = "ok"
    if expected_route == "rag_search":
        # RAG 질문은 fallback 사과문 대신 핵심 용어를 직접 설명해야 한다.
        if "죄송" in answer or "다른 질문" in answer:
            quality_ok = False
            quality_reason = "fallback_answer_detected"
        elif "구개음화" not in answer:
            quality_ok = False
            quality_reason = "missing_key_term"
    elif expected_route == "web_search":
        if "langgraph" not in answer.lower() and "랭그래프" not in answer:
            quality_ok = False
            quality_reason = "missing_topic_term"

    return {
        "question": question,
        "expected_route": expected_route,
        "selected_route": selected_route,
        "route_ok": is_route_ok,
        "has_answer": has_answer,
        "quality_ok": quality_ok,
        "quality_reason": quality_reason,
        "routing_trace": output.routing_trace,
        "node_path": node_path,
        "answer": answer,
    }


def main() -> int:
    eval_cases = [
        ("랭그래프가 뭐야?", "web_search"),
        ("구개음화가 뭐야?", "rag_search"),
    ]

    results: list[dict[str, Any]] = []
    for question, expected_route in eval_cases:
        results.append(run_eval(question=question, expected_route=expected_route))

    print(json.dumps(results, ensure_ascii=False, indent=2))

    all_ok = all(item["route_ok"] and item["has_answer"] and item["quality_ok"] for item in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
