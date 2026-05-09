"""
Project 12: Multi-Project Orchestrator — LangGraph StateGraph.

Graph topology:
  START → decompose → [fan-out via Send] → process_project → aggregate → END

The fan-out spawns one async process_project invocation per sub-project.
All invocations run concurrently; their results accumulate via operator.add
before aggregate runs.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .nodes import decompose, process_project, aggregate


class OrchestratorState(TypedDict):
    goal: str
    gen_id: str                                        # timestamp run ID, e.g. "20260509_175030"
    projects: list[dict]                              # set by decompose
    results: Annotated[list[dict], operator.add]      # accumulated by fan-out
    final_report: str                                 # set by aggregate


def _fan_out(state: OrchestratorState) -> list[Send]:
    """Route each sub-project to its own process_project invocation."""
    return [
        Send("process_project", {"goal": state["goal"], "gen_id": state["gen_id"], "project": p})
        for p in state["projects"]
    ]


def build() -> object:
    g = StateGraph(OrchestratorState)

    g.add_node("decompose", decompose)
    g.add_node("process_project", process_project)
    g.add_node("aggregate", aggregate)

    g.add_edge(START, "decompose")
    g.add_conditional_edges("decompose", _fan_out, ["process_project"])
    g.add_edge("process_project", "aggregate")
    g.add_edge("aggregate", END)

    return g.compile()


graph = build()
