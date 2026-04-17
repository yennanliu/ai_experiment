from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import retrieve_node, generate_node


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


rag_graph = build_graph()
