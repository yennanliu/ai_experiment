"""LangGraph node functions — one per processing stage"""

import json
from datetime import datetime
from typing import Literal

from graph.state import TicketState
from utils.llm import chat

SLA_MAP = {"critical": 1, "high": 4, "medium": 12, "low": 24}


def _log(state: TicketState, node: str, detail: str) -> None:
    state.history.append({"node": node, "time": datetime.now().isoformat(), "detail": detail})


def classify(state: TicketState) -> TicketState:
    system = """You classify customer support tickets.
Reply with JSON only: {"category": "<category>", "confidence": <0-1>}
Categories: technical, billing, account, feature_request, bug_report, other"""
    result = json.loads(chat(system, state.user_message))
    state.category = result["category"]
    state.confidence = result["confidence"]
    _log(state, "classify", f"category={state.category} confidence={state.confidence:.2f}")
    return state


def prioritize(state: TicketState) -> TicketState:
    system = """You assess the urgency of a support ticket.
Reply with JSON only: {"priority": "<priority>"}
Priorities: critical, high, medium, low"""
    result = json.loads(chat(system, f"Category: {state.category}\nMessage: {state.user_message}"))
    state.priority = result["priority"]
    state.sla_hours = SLA_MAP.get(state.priority, 24)
    _log(state, "prioritize", f"priority={state.priority} sla={state.sla_hours}h")
    return state


def route(state: TicketState) -> TicketState:
    mapping = {
        "technical": "Engineering",
        "billing": "Finance",
        "account": "Customer Success",
        "feature_request": "Product",
        "bug_report": "Engineering",
        "other": "General Support",
    }
    state.department = mapping.get(state.category, "General Support")
    _log(state, "route", f"department={state.department}")
    return state


def generate_response(state: TicketState) -> TicketState:
    system = """You write concise, professional customer support replies (150-300 chars).
Acknowledge the issue, provide next steps, and mention the SLA."""
    prompt = (
        f"Ticket: {state.user_message}\n"
        f"Category: {state.category} | Priority: {state.priority} | "
        f"Department: {state.department} | SLA: {state.sla_hours}h"
    )
    state.response = chat(system, prompt)
    _log(state, "generate_response", f"len={len(state.response)}")
    return state


def quality_check(state: TicketState) -> TicketState:
    system = """You score a support response on: relevance, clarity, professionalism, completeness, tone.
Reply with JSON only: {"score": <0-1>}"""
    result = json.loads(chat(system, f"Ticket: {state.user_message}\nResponse: {state.response}"))
    state.quality_score = result["score"]
    state.retry_count += 1
    _log(state, "quality_check", f"score={state.quality_score:.2f} retry={state.retry_count}")
    return state


def should_retry(state: TicketState) -> Literal["generate_response", "__end__"]:
    if state.quality_score < 0.8 and state.retry_count < 3:
        return "generate_response"
    return "__end__"
