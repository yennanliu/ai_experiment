from langgraph.graph import StateGraph, END
from .state import ResumeState
from .nodes import (
    parse_inputs,
    ats_simulate,
    rewrite_resume,
    write_cover_letter,
    recruiter_review,
    hiring_manager_review,
    score_output,
)


def build_graph():
    graph = StateGraph(ResumeState)

    graph.add_node("parse_inputs", parse_inputs)
    graph.add_node("ats_simulate", ats_simulate)
    graph.add_node("rewrite_resume", rewrite_resume)
    graph.add_node("write_cover_letter", write_cover_letter)
    graph.add_node("recruiter_review", recruiter_review)
    graph.add_node("hiring_manager_review", hiring_manager_review)
    graph.add_node("score_output", score_output)

    graph.set_entry_point("parse_inputs")
    graph.add_edge("parse_inputs", "ats_simulate")
    graph.add_edge("ats_simulate", "rewrite_resume")
    graph.add_edge("rewrite_resume", "write_cover_letter")
    graph.add_edge("write_cover_letter", "recruiter_review")
    graph.add_edge("recruiter_review", "hiring_manager_review")
    graph.add_edge("hiring_manager_review", "score_output")
    graph.add_edge("score_output", END)

    return graph.compile()


resume_agent = build_graph()
