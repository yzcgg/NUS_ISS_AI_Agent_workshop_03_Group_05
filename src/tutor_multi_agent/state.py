from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class TutorState(TypedDict, total=False):
    """State container shared across LangGraph nodes."""

    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    pending_routes: list[str]
    completed_routes: list[str]
    tool_events: list[str]
    turn_count: int
    final_answer: str
