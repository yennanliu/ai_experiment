"""Exercise 4 — the wrapper publishes the schemas opacity was hiding.

    Design an A2A agent that wraps an MCP server. Map each MCP tool to an A2A
    skill. Note the trade-offs -- what opacity is lost?

Reading of the exercise: "what opacity is lost" is answerable by diffing the
two descriptors rather than by argument, so the mapping is built and then the
fields are counted on each side. The loss is not that the wrapper leaks
something extra; it is that an A2A skill has no field corresponding to an
MCP `inputSchema`, so the mapping has to put it somewhere, and every
somewhere is public.

**ANSWER: one skill per tool, with the schema's parameter names surfaced.**
Three MCP tools become **3** skills carrying `id`, `name`, `description`,
`inputModes` and `outputModes` -- the Agent Card's own shape. The caller
invokes `draft_report`-style skills and never names an MCP method.

**WHAT OPACITY IS LOST: the argument surface.** An MCP tool publishes an
`inputSchema` with named, typed properties; an A2A skill publishes
`inputModes` -- `["text", "file", "data"]` -- which says what *kinds* of part
it takes and nothing about their contents. Mapping faithfully means writing
the parameter names into the skill's prose, so **4** property names that were
a machine-readable contract become documentation. Mapping *without* them is
not opacity either: the tools' own descriptions already name `query` and
`destination`, **2** of the **4**, so the honest choice is between a contract
in prose and half a contract by accident.

**FINDING: the lesson's own agent already pays this price.** `draft_report`
declares `inputModes: ["text", "file", "data"]` and the writer requires a
`targetLength` key inside the data part -- a requirement stated in **0**
machine-readable fields of the card and discoverable only by being paused.
Exercise 1's whole failure mode is this gap.

**FINDING: the reverse direction loses the task lifecycle.** An MCP
`tools/call` is one request and one result; an A2A task has **5** states and
can pause for input. Wrapping MCP in A2A means the skill can never reach
`input_required`, so a skill backed by a tool is a strictly smaller thing
than one backed by an agent -- and the card cannot say which it is.

Structure: `as_skill` is the mapping, applied with and without the schema
names so the two readings of "faithful" can be compared.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "19-a2a-protocol"
MCP_TOOLS = [
    {"name": "notes_search", "description": "Search notes by query.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                     "required": ["query"]}},
    {"name": "notes_create", "description": "Create a note.",
     "inputSchema": {"type": "object",
                     "properties": {"title": {"type": "string"},
                                    "body": {"type": "string"}},
                     "required": ["title"]}},
    {"name": "notes_export", "description": "Export notes to a destination.",
     "inputSchema": {"type": "object",
                     "properties": {"destination": {"type": "string"}},
                     "required": ["destination"]}},
]


def as_skill(tool, *, surface_schema=True):
    """One MCP tool as one A2A skill, with or without the parameter names."""
    properties = sorted(tool["inputSchema"].get("properties", {}))
    description = tool["description"]
    if surface_schema and properties:
        description = f"{description} Data part keys: {', '.join(properties)}."
    return {"id": tool["name"], "name": tool["name"].replace("_", " ").title(),
            "description": description,
            "inputModes": ["text", "data"], "outputModes": ["text", "artifact"]}


def property_names(tools):
    return sorted({name for tool in tools
                   for name in tool["inputSchema"].get("properties", {})})


def discoverable(names, skills):
    """Which property names a reader could recover from the skill descriptions."""
    return sorted(n for n in names if any(n in skill["description"] for skill in skills))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    surfaced = [as_skill(tool) for tool in MCP_TOOLS]
    opaque = [as_skill(tool, surface_schema=False) for tool in MCP_TOOLS]
    card_skill = ref.WRITER_AGENT_CARD["skills"][0]
    names = property_names(MCP_TOOLS)
    return {
        "skills": len(surfaced), "ids": [skill["id"] for skill in surfaced],
        "skill_fields": sorted(surfaced[0]),
        "card_fields": sorted(card_skill),
        "same_shape": sorted(surfaced[0]) == sorted(card_skill),
        "properties": names,
        "surfaced_found": len(discoverable(names, surfaced)),
        "opaque_found": discoverable(names, opaque),
        "input_modes": card_skill["inputModes"],
        "card_schema_fields": [f for f in card_skill if "schema" in f.lower()],
        "states": ["submitted", "working", "input_required", "completed", "failed"],
        "mcp_round_trips": 1,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one skill per tool, in the Agent Card's own shape",
            all([result["skills"] == 3, result["same_shape"],
                 result["ids"] == ["notes_search", "notes_create", "notes_export"],
                 result["skill_fields"] == ["description", "id", "inputModes", "name",
                                            "outputModes"]]),
            f"three MCP tools become {result['skills']} skills carrying "
            f"{result['skill_fields']}, the same fields the lesson's own card declares "
            f"({result['same_shape']}). The caller invokes {result['ids']} and never names "
            "an MCP method",
        ),
        practice.Check(
            "WHAT IS LOST: the argument surface, which has no field to go in",
            all([result["card_schema_fields"] == [],
                 result["input_modes"] == ["text", "file", "data"],
                 result["surfaced_found"] == 4, result["opaque_found"] ==
                 ["destination", "query"], len(result["properties"]) == 4]),
            f"an A2A skill has {len(result['card_schema_fields'])} schema fields and declares "
            f"{result['input_modes']} -- what kinds of part it takes, not what goes in them. "
            f"Mapping faithfully writes the {len(result['properties'])} property names "
            f"{result['properties']} into prose, turning a machine-readable contract into "
            f"documentation. Mapping without them is not opacity either: the tools' own "
            f"descriptions already leak {result['opaque_found']}, "
            f"{len(result['opaque_found'])} of {len(result['properties'])}, by accident -- "
            "half a contract, which is worse than either choice made deliberately",
        ),
        practice.Check(
            "FINDING: the lesson's own agent already pays this price",
            all([result["input_modes"] == ["text", "file", "data"],
                 result["card_schema_fields"] == []]),
            f"draft_report declares {result['input_modes']} while the writer requires a "
            f"targetLength key inside the data part -- stated in "
            f"{len(result['card_schema_fields'])} machine-readable fields of the card and "
            "discoverable only by being paused. Exercise 1's whole failure mode is this gap",
        ),
        practice.Check(
            "FINDING: the reverse direction loses the task lifecycle",
            all([len(result["states"]) == 5, result["mcp_round_trips"] == 1]),
            f"an MCP tools/call is {result['mcp_round_trips']} request and one result, while "
            f"an A2A task has {len(result['states'])} states and can pause for input. A "
            "skill backed by a tool can never reach input_required, so it is a strictly "
            "smaller thing than one backed by an agent -- and the card cannot say which it is",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
