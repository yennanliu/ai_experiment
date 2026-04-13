"""Graph router — exposes the LangGraph structure as JSON"""

from fastapi import APIRouter

router = APIRouter(prefix="/graph", tags=["graph"])

GRAPH_DEFINITION = {
    "nodes": [
        {"id": "START",             "label": "START",             "type": "terminal",    "description": ""},
        {"id": "classify",          "label": "Classify",          "type": "llm",         "description": "Tags category & confidence score"},
        {"id": "research",          "label": "Research",          "type": "tool",        "description": "Web search for technical/bug tickets"},
        {"id": "prioritize",        "label": "Prioritize",        "type": "llm",         "description": "Sets priority & SLA hours"},
        {"id": "route",             "label": "Route",             "type": "deterministic","description": "Assigns department"},
        {"id": "generate_response", "label": "Generate Response", "type": "llm",         "description": "Writes customer reply"},
        {"id": "quality_check",     "label": "Quality Check",     "type": "llm",         "description": "Scores response 0–1"},
        {"id": "END",               "label": "END",               "type": "terminal",    "description": ""},
    ],
    "edges": [
        {"from": "START",             "to": "classify",          "label": ""},
        {"from": "classify",          "to": "research",          "label": ""},
        {"from": "research",          "to": "prioritize",        "label": ""},
        {"from": "prioritize",        "to": "route",             "label": ""},
        {"from": "route",             "to": "generate_response", "label": ""},
        {"from": "generate_response", "to": "quality_check",     "label": ""},
        {"from": "quality_check",     "to": "generate_response", "label": "score < 0.8 & retries < 3", "conditional": True},
        {"from": "quality_check",     "to": "END",               "label": "score ≥ 0.8", "conditional": True},
    ],
}


@router.get("")
async def get_graph():
    return GRAPH_DEFINITION
