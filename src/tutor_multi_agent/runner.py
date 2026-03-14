from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage

from .graph import build_graph


@lru_cache(maxsize=4)
def _cached_graph(model_name: str):
    return build_graph(model_name=model_name)


def run_turn(
    user_input: str,
    session_id: str,
    model_name: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Run one user turn in a persistent session."""
    graph = _cached_graph(model_name)
    result = graph.invoke(
        {
            "session_id": session_id,
            "messages": [HumanMessage(content=user_input)],
            "route": "",
            "pending_routes": [],
            "completed_routes": [],
            "tool_events": [],
            "turn_count": 0,
            "final_answer": "",
        },
        config={"configurable": {"thread_id": session_id}},
    )
    return {
        "final_answer": result.get("final_answer", ""),
        "route": result.get("route", "finish"),
        "tool_events": result.get("tool_events", []),
        "turn_count": result.get("turn_count", 0),
    }
