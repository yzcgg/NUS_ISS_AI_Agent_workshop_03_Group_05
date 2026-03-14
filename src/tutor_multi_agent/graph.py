from __future__ import annotations

import json
from typing import Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from .state import TutorState
from .tools import get_date_tool, get_poem_tool

RouteName = Literal["chinese", "math", "planner", "finish"]

ORCHESTRATOR_PROMPT = """
You are the Orchestrator of a multi-agent primary school tutoring system.

Available routes:
- chinese: Chinese language tutoring
- math: Math tutoring
- planner: Weekly study planning
- finish: End this turn and return collected agent answers

Decision rules:
- Infer which routes are explicitly needed by the latest user request.
- Never add unrelated routes that the user did not ask for.
- Then choose the next route from that inferred route list.
- If all inferred routes are completed, choose finish.
- Do not call tools directly.

Output format:
Return strict JSON only, with keys:
{"routes": ["chinese|math|planner"], "reason": "short reason"}
"""

CHINESE_AGENT_PROMPT = """
You are Agent-01, a private Chinese tutor for primary school students.
Goals:
- Explain concepts in simple language.
- Encourage and guide the student.
- If suitable, call get_poem_tool to provide one random Chinese poem line.
Rules:
- Keep answer concise and practical for children.
- Use English in your final answer to the student.
"""

MATH_AGENT_PROMPT = """
You are Agent-02, a private Math tutor for primary school students.
Goals:
- Explain math step-by-step with simple examples.
- Keep a friendly and educational style.
Rules:
- Do not use any external tool.
- Use English in your final answer to the student.
"""

PLANNER_AGENT_PROMPT = """
You are Agent-03, a weekly study planner for primary school students.
Goals:
- Build a practical and age-appropriate weekly study schedule.
- Call get_date_tool to anchor the plan with the current date/time.
Rules:
- Give clear daily tasks and short durations.
- Use English in your final answer to the student.
"""


def _stringify_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _latest_user_message(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _stringify_content(msg.content).strip()
    return ""


def _collect_current_turn_answers(messages: Sequence[BaseMessage]) -> list[str]:
    last_user_index = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_user_index = i
    answers: list[str] = []
    for msg in messages[last_user_index + 1 :]:
        if isinstance(msg, AIMessage):
            text = _stringify_content(msg.content).strip()
            if text:
                answers.append(text)
    return answers


def _parse_orchestrator_routes(decision_text: str) -> list[str]:
    candidates = ("chinese", "math", "planner")
    lowered = decision_text.lower().strip()
    routes: list[str] = []

    try:
        parsed = json.loads(decision_text)
        if isinstance(parsed, dict):
            raw_routes = parsed.get("routes", [])
            if isinstance(raw_routes, list):
                for item in raw_routes:
                    route = str(item).lower().strip()
                    if route in candidates and route not in routes:
                        routes.append(route)
            if routes:
                return routes
    except json.JSONDecodeError:
        pass

    for route in candidates:
        if f'"{route}"' in lowered or route in lowered:
            routes.append(route)
    return routes or ["math"]


def _extract_delta_messages(
    input_messages: Sequence[BaseMessage],
    output_messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    if len(output_messages) <= len(input_messages):
        return list(output_messages)
    return list(output_messages[len(input_messages) :])


def build_graph(model_name: str = "gpt-4o-mini"):
    """Build and compile the LangGraph workflow."""
    llm = ChatOpenAI(model=model_name, temperature=0)

    chinese_agent = create_react_agent(
        model=llm,
        tools=[get_poem_tool],
        prompt=CHINESE_AGENT_PROMPT,
    )
    planner_agent = create_react_agent(
        model=llm,
        tools=[get_date_tool],
        prompt=PLANNER_AGENT_PROMPT,
    )

    max_turns = 6

    def orchestrator_node(state: TutorState) -> dict:
        turn_count = state.get("turn_count", 0)
        pending_routes = list(state.get("pending_routes", []))
        completed_routes = list(state.get("completed_routes", []))
        messages = state.get("messages", [])
        tool_events = list(state.get("tool_events", []))

        if turn_count >= max_turns:
            tool_events.append(f"orchestrator -> finish (max_turns={max_turns})")
            return {"route": "finish", "tool_events": tool_events}

        latest_user_text = _latest_user_message(messages)
        if not pending_routes:
            decision = llm.invoke(
                [
                    ("system", ORCHESTRATOR_PROMPT),
                    (
                        "human",
                        (
                            "Infer the route list needed for this user request.\\n"
                            f"latest_user_message: {latest_user_text}\\n"
                            "Return JSON only."
                        ),
                    ),
                ]
            )
            decision_text = _stringify_content(decision.content).strip()
            pending_routes = _parse_orchestrator_routes(decision_text)
            tool_events.append(f"orchestrator llm routes: {decision_text}")
            tool_events.append(f"orchestrator intent queue: {pending_routes}")

        next_route: RouteName = "finish"
        for route in pending_routes:
            if route not in completed_routes:
                next_route = route  # type: ignore[assignment]
                break

        if next_route in completed_routes and next_route != "finish":
            next_route = "finish"
            tool_events.append("orchestrator override: repeated route -> finish")

        if next_route == "finish":
            tool_events.append("orchestrator -> finish (all inferred intents completed)")
            return {
                "route": "finish",
                "pending_routes": pending_routes,
                "tool_events": tool_events,
            }

        tool_events.append(f"orchestrator -> {next_route}")
        return {
            "route": next_route,
            "pending_routes": pending_routes,
            "turn_count": turn_count + 1,
            "tool_events": tool_events,
        }

    def chinese_node(state: TutorState) -> dict:
        input_messages = state.get("messages", [])
        completed_routes = state.get("completed_routes", [])
        tool_events = list(state.get("tool_events", []))
        result = chinese_agent.invoke({"messages": input_messages})
        output_messages = result.get("messages", [])
        delta_messages = _extract_delta_messages(input_messages, output_messages)
        used_tools = [
            msg.name for msg in delta_messages if isinstance(msg, ToolMessage) and msg.name
        ]

        tool_events.append("chinese_agent completed response")
        if used_tools:
            tool_events.append(f"chinese_agent tools: {used_tools}")
        return {
            "messages": delta_messages,
            "completed_routes": completed_routes + ["chinese"],
            "tool_events": tool_events,
        }

    def math_node(state: TutorState) -> dict:
        messages = state.get("messages", [])
        completed_routes = state.get("completed_routes", [])
        tool_events = list(state.get("tool_events", []))
        response = llm.invoke(
            [
                ("system", MATH_AGENT_PROMPT),
                *messages,
            ]
        )
        tool_events.extend(["math_agent completed response", "math_agent tools: []"])
        return {
            "messages": [response],
            "completed_routes": completed_routes + ["math"],
            "tool_events": tool_events,
        }

    def planner_node(state: TutorState) -> dict:
        input_messages = state.get("messages", [])
        completed_routes = state.get("completed_routes", [])
        tool_events = list(state.get("tool_events", []))
        result = planner_agent.invoke({"messages": input_messages})
        output_messages = result.get("messages", [])
        delta_messages = _extract_delta_messages(input_messages, output_messages)
        used_tools = [
            msg.name for msg in delta_messages if isinstance(msg, ToolMessage) and msg.name
        ]

        tool_events.append("planner_agent completed response")
        if used_tools:
            tool_events.append(f"planner_agent tools: {used_tools}")
        return {
            "messages": delta_messages,
            "completed_routes": completed_routes + ["planner"],
            "tool_events": tool_events,
        }

    def finalize_node(state: TutorState) -> dict:
        tool_events = list(state.get("tool_events", []))
        answers = _collect_current_turn_answers(state.get("messages", []))
        if answers:
            final_answer = "\n\n".join(answers)
        else:
            final_answer = "I can help with Chinese, Math, and weekly study planning."
        tool_events.append("orchestrator finalized output")
        return {
            "route": "finish",
            "final_answer": final_answer,
            "tool_events": tool_events,
        }

    graph = StateGraph(TutorState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("chinese", chinese_node)
    graph.add_node("math", math_node)
    graph.add_node("planner", planner_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state.get("route", "finish"),
        {
            "chinese": "chinese",
            "math": "math",
            "planner": "planner",
            "finish": "finalize",
        },
    )
    graph.add_edge("chinese", "orchestrator")
    graph.add_edge("math", "orchestrator")
    graph.add_edge("planner", "orchestrator")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())
