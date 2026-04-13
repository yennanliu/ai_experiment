from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import classify, select_template, generate_draft, build_checklist


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify)
    graph.add_node("select_template", select_template)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("build_checklist", build_checklist)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "select_template")
    graph.add_edge("select_template", "generate_draft")
    graph.add_edge("generate_draft", "build_checklist")
    graph.add_edge("build_checklist", END)

    return graph.compile()


email_agent = build_graph()
