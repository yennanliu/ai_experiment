"""Builds and exposes the compiled LangGraph pipeline"""

from langgraph.graph import START, StateGraph

from graph.nodes import classify, generate_response, prioritize, quality_check, research, route, should_retry
from graph.state import TicketState


def _build() -> object:
    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("research", research)
    g.add_node("prioritize", prioritize)
    g.add_node("route", route)
    g.add_node("generate_response", generate_response)
    g.add_node("quality_check", quality_check)

    g.add_edge(START, "classify")
    g.add_edge("classify", "research")
    g.add_edge("research", "prioritize")
    g.add_edge("prioritize", "route")
    g.add_edge("route", "generate_response")
    g.add_edge("generate_response", "quality_check")
    g.add_conditional_edges("quality_check", should_retry)

    return g.compile()


_graph = _build()


def process_ticket(ticket_id: str, message: str) -> TicketState:
    initial = TicketState(ticket_id=ticket_id, user_message=message)
    result = _graph.invoke(initial)
    return TicketState(**result) if isinstance(result, dict) else result
