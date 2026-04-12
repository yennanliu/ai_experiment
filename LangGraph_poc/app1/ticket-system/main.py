"""Ticket Processing System using LangGraph + OpenAI"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"

# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class TicketState:
    # Input
    ticket_id: str = ""
    user_message: str = ""

    # Classification
    category: str = ""          # technical | billing | account | feature_request | bug_report | other
    confidence: float = 0.0

    # Priority
    priority: str = ""          # critical | high | medium | low
    sla_hours: int = 24

    # Routing
    department: str = ""

    # Response
    response: str = ""
    quality_score: float = 0.0
    retry_count: int = 0

    # Audit
    history: list = field(default_factory=list)


SLA_MAP = {"critical": 1, "high": 4, "medium": 12, "low": 24}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _chat(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def _log(state: TicketState, node: str, detail: str) -> None:
    state.history.append({"node": node, "time": datetime.now().isoformat(), "detail": detail})


# ── Nodes ─────────────────────────────────────────────────────────────────────

def classify(state: TicketState) -> TicketState:
    system = """You classify customer support tickets.
Reply with JSON only: {"category": "<category>", "confidence": <0-1>}
Categories: technical, billing, account, feature_request, bug_report, other"""

    raw = _chat(system, state.user_message)
    result = json.loads(raw)
    state.category = result["category"]
    state.confidence = result["confidence"]
    _log(state, "classify", f"category={state.category} confidence={state.confidence:.2f}")
    return state


def prioritize(state: TicketState) -> TicketState:
    system = """You assess the urgency of a support ticket.
Reply with JSON only: {"priority": "<priority>"}
Priorities: critical, high, medium, low"""

    raw = _chat(system, f"Category: {state.category}\nMessage: {state.user_message}")
    result = json.loads(raw)
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
    state.response = _chat(system, prompt)
    _log(state, "generate_response", f"len={len(state.response)}")
    return state


def quality_check(state: TicketState) -> TicketState:
    system = """You score a support response on: relevance, clarity, professionalism, completeness, tone.
Reply with JSON only: {"score": <0-1>}"""

    raw = _chat(system, f"Ticket: {state.user_message}\nResponse: {state.response}")
    result = json.loads(raw)
    state.quality_score = result["score"]
    state.retry_count += 1
    _log(state, "quality_check", f"score={state.quality_score:.2f} retry={state.retry_count}")
    return state


# ── Conditional edge ──────────────────────────────────────────────────────────

def should_retry(state: TicketState) -> Literal["generate_response", "__end__"]:
    if state.quality_score < 0.8 and state.retry_count < 3:
        return "generate_response"
    return "__end__"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("prioritize", prioritize)
    g.add_node("route", route)
    g.add_node("generate_response", generate_response)
    g.add_node("quality_check", quality_check)

    g.add_edge(START, "classify")
    g.add_edge("classify", "prioritize")
    g.add_edge("prioritize", "route")
    g.add_edge("route", "generate_response")
    g.add_edge("generate_response", "quality_check")
    g.add_conditional_edges("quality_check", should_retry)

    return g.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

def process_ticket(ticket_id: str, message: str) -> TicketState:
    graph = build_graph()
    initial = TicketState(ticket_id=ticket_id, user_message=message)
    return graph.invoke(initial)


def print_result(state: TicketState) -> None:
    print(f"\n{'='*60}")
    print(f"Ticket ID : {state.ticket_id}")
    print(f"Category  : {state.category} (confidence: {state.confidence:.0%})")
    print(f"Priority  : {state.priority} | SLA: {state.sla_hours}h")
    print(f"Department: {state.department}")
    print(f"Quality   : {state.quality_score:.0%} (retries: {state.retry_count})")
    print(f"\nResponse:\n  {state.response}")
    print(f"\nAudit trail:")
    for h in state.history:
        print(f"  [{h['time'][11:19]}] {h['node']:20s} {h['detail']}")
    print("="*60)


if __name__ == "__main__":
    tickets = [
        ("TKT-001", "My payment was charged twice this month and I need a refund immediately!"),
        ("TKT-002", "The app crashes when I try to upload a file larger than 10MB."),
        ("TKT-003", "Can you add dark mode to the dashboard?"),
    ]

    for ticket_id, message in tickets:
        print(f"\nProcessing {ticket_id}: {message[:60]}...")
        result = process_ticket(ticket_id, message)
        print_result(result)
