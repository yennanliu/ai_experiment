"""LangGraph node functions — one per processing stage"""

import json
from datetime import datetime
from typing import Literal

from graph.state import TicketState
from tools.web_search import web_search
from utils.llm import chat
from utils.logger import log

SLA_MAP = {"critical": 1, "high": 4, "medium": 12, "low": 24}


def _log(state: TicketState, node: str, detail: str, prompt: str = "", response: str = "") -> None:
    """Append a history entry and emit a colored terminal log line."""
    state.history.append({
        "node": node,
        "time": datetime.now().isoformat(),
        "detail": detail,
        "prompt": prompt,
        "response": response,
    })
    log.info(
        detail,
        extra={"node": node, "ticket_id": state.ticket_id,
               "llm_in": prompt or None, "llm_out": response or None},
    )


def classify(state: TicketState) -> TicketState:
    log.info("starting", extra={"node": "classify", "ticket_id": state.ticket_id})
    system = """You classify customer support tickets.
Reply with JSON only: {"category": "<category>", "confidence": <0-1>}
Categories: technical, billing, account, feature_request, bug_report, other"""
    prompt = state.user_message
    raw = chat(system, prompt)
    result = json.loads(raw)
    state.category = result["category"]
    state.confidence = result["confidence"]
    _log(state, "classify",
         f"category={state.category}  confidence={state.confidence:.2f}",
         prompt=prompt, response=raw)
    return state


RESEARCH_CATEGORIES = {"technical", "bug_report"}


def research(state: TicketState) -> TicketState:
    """Web-search for context on technical / bug tickets; skip others."""
    log.info("starting", extra={"node": "research", "ticket_id": state.ticket_id})
    if state.category not in RESEARCH_CATEGORIES:
        _log(state, "research", f"skipped (category={state.category})")
        return state

    query = f"{state.category} issue: {state.user_message[:200]}"
    notes = web_search(query)
    state.research_notes = notes
    _log(state, "research", f"fetched {len(notes)} chars of context", prompt=query, response=notes[:300])
    return state


def prioritize(state: TicketState) -> TicketState:
    log.info("starting", extra={"node": "prioritize", "ticket_id": state.ticket_id})
    system = """You assess the urgency of a support ticket.
Reply with JSON only: {"priority": "<priority>"}
Priorities: critical, high, medium, low"""
    prompt = f"Category: {state.category}\nMessage: {state.user_message}"
    raw = chat(system, prompt)
    result = json.loads(raw)
    state.priority = result["priority"]
    state.sla_hours = SLA_MAP.get(state.priority, 24)
    _log(state, "prioritize",
         f"priority={state.priority}  sla={state.sla_hours}h",
         prompt=prompt, response=raw)
    return state


def route(state: TicketState) -> TicketState:
    log.info("starting", extra={"node": "route", "ticket_id": state.ticket_id})
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
    log.info("starting", extra={"node": "generate_response", "ticket_id": state.ticket_id})
    system = """You write concise, professional customer support replies (150-300 chars).
Acknowledge the issue, provide next steps, and mention the SLA."""
    research_section = f"\nResearch context:\n{state.research_notes[:500]}" if state.research_notes else ""
    prompt = (
        f"Ticket: {state.user_message}\n"
        f"Category: {state.category} | Priority: {state.priority} | "
        f"Department: {state.department} | SLA: {state.sla_hours}h"
        f"{research_section}"
    )
    raw = chat(system, prompt)
    state.response = raw
    _log(state, "generate_response",
         f"len={len(state.response)}",
         prompt=prompt, response=raw)
    return state


def quality_check(state: TicketState) -> TicketState:
    log.info("starting", extra={"node": "quality_check", "ticket_id": state.ticket_id})
    system = """You score a support response on: relevance, clarity, professionalism, completeness, tone.
Reply with JSON only: {"score": <0-1>}"""
    prompt = f"Ticket: {state.user_message}\nResponse: {state.response}"
    raw = chat(system, prompt)
    result = json.loads(raw)
    state.quality_score = result["score"]
    state.retry_count += 1
    _log(state, "quality_check",
         f"score={state.quality_score:.2f}  retry={state.retry_count}",
         prompt=prompt, response=raw)
    return state


def should_retry(state: TicketState) -> Literal["generate_response", "__end__"]:
    if state.quality_score < 0.8 and state.retry_count < 3:
        log.warning(
            f"quality below threshold ({state.quality_score:.2f}), retrying…",
            extra={"node": "quality_check", "ticket_id": state.ticket_id},
        )
        return "generate_response"
    log.info(
        f"pipeline complete  score={state.quality_score:.2f}",
        extra={"node": "quality_check", "ticket_id": state.ticket_id},
    )
    return "__end__"
