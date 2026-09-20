"""Exercise 2 — zero findings is reachable by a registry the lesson calls bad.

    Design an MCP server for a notes application with atomic tools: list,
    search, create, update, delete, and a `summarize` slash prompt. Lint the
    registry. Target zero findings.

Reading of the exercise: the five atomic tools and the prompt are designed to
the lesson's own rules and linted to zero. Then the same five operations are
rewritten as the thing the lesson spends a whole section warning against -- one
monolithic tool -- and linted again, because "target zero findings" is only a
useful target if the score separates the two.

**ANSWER: six entries, five tools and one prompt, at 0 findings.**
`notes_list`, `notes_search`, `notes_create`, `notes_update`, `notes_delete`,
plus a `summarize` prompt. Every description carries "Use when" and "Do not use
for", every string field is described, every schema declares `required`.

**FINDING: the monolithic version scores 0 too, if the key is not called
`action`.** The identical five-operations-in-one tool with its verb under
`"action"` scores **1** warn; under `"op"` it scores **0**. `lint_schema`
matches `key == "action"` literally, so the anti-pattern the lesson names is
detected by the name of the field rather than by its shape.

**FINDING: and the prompt is linted as though it were a tool.** `lint_tool`
reads `name`, `description` and `input_schema` off any dict, so the
`summarize` prompt is checked against the same rules -- including "Do not use
for", which is tool-disambiguation advice that a slash prompt has no
counterpart to be disambiguated from. It passes only because the wording was
bent to satisfy a rule that does not apply.

**FINDING: five atomic tools cost 10x the description budget of one.** The
atomic registry spends **729** characters of description against the monolith's
**75**, all of which reaches the model on every request. Atomicity is the right
call and it is not free -- which is the trade the lesson's "atomic vs
monolithic" section asserts without pricing.

Structure: `NOTES` is the atomic registry, `monolith` builds the same
operations under one tool with a configurable verb key, and `census` counts
findings by severity.
"""

from __future__ import annotations

from collections import Counter

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "05-tool-schema-design"
OPERATIONS = ("list", "search", "create", "update", "delete")
NOTE_ID = {"type": "string", "description": "Note id, e.g. note-00000042."}


def tool(name, use, avoid, properties, required):
    return {"name": name,
            "description": f"Use when {use} Do not use for {avoid}",
            "input_schema": {"type": "object", "properties": properties,
                             "required": required}}


NOTES = [
    tool("notes_list", "the user wants their notes enumerated in order.",
         "searching note bodies; call notes_search instead.",
         {"limit": {"type": "integer"},
          "cursor": {"type": "string", "description": "Opaque page cursor."}}, []),
    tool("notes_search", "the user describes note content they want found.",
         "listing every note; call notes_list instead.",
         {"query": {"type": "string", "description": "Free-text search over note bodies."},
          "limit": {"type": "integer"}}, ["query"]),
    tool("notes_create", "the user dictates a new note to store.",
         "editing a note that already exists; call notes_update instead.",
         {"title": {"type": "string", "description": "Short note title."},
          "body": {"type": "string", "description": "Note body, plain text."}},
         ["title", "body"]),
    tool("notes_update", "the user revises the text of an existing note.",
         "creating a note; call notes_create instead.",
         {"note_id": NOTE_ID,
          "body": {"type": "string", "description": "Replacement note body."}},
         ["note_id", "body"]),
    tool("notes_delete", "the user asks for a note to be removed permanently.",
         "archiving or hiding a note, which this does not do.",
         {"note_id": NOTE_ID}, ["note_id"]),
    tool("summarize", "the user asks for a digest of notes already retrieved.",
         "fetching notes; this reads what is in context.",
         {"style": {"type": "string", "enum": ["bullets", "paragraph"],
                    "description": "Output shape."}}, ["style"]),
]


def monolith(verb_key):
    """The same five operations as one tool, with the verb under `verb_key`."""
    return [{
        "name": "notes",
        "description": ("Use when the user wants anything done with notes. "
                        "Do not use for calendars."),
        "input_schema": {
            "type": "object",
            "properties": {
                verb_key: {"type": "string", "enum": list(OPERATIONS),
                           "description": "Which notes operation to perform."},
                "note_id": NOTE_ID,
            },
            "required": [verb_key],
        },
    }]


def census(ref, registry):
    return dict(Counter(finding.severity for finding in ref.lint_registry(registry)))


def budget(registry):
    return sum(len(tool["description"]) for tool in registry)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tools = [entry for entry in NOTES if entry["name"] != "summarize"]
    prompt = [entry for entry in NOTES if entry["name"] == "summarize"]
    return {
        "entries": len(NOTES), "tools": len(tools),
        "names": [entry["name"] for entry in NOTES],
        "findings": census(ref, NOTES), "total": len(ref.lint_registry(NOTES)),
        "prompt_findings": len(ref.lint_registry(prompt)),
        "action_key": len(ref.lint_registry(monolith("action"))),
        "op_key": len(ref.lint_registry(monolith("op"))),
        "action_messages": [f.message for f in ref.lint_registry(monolith("action"))],
        "atomic_budget": budget(NOTES), "monolith_budget": budget(monolith("op")),
        "budget_ratio": round(budget(NOTES) / budget(monolith("op"))),
        "operations": len(OPERATIONS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: six entries -- five atomic tools and a prompt -- at 0 findings",
            all([result["entries"] == 6, result["tools"] == 5, result["total"] == 0,
                 result["findings"] == {},
                 result["names"] == ["notes_list", "notes_search", "notes_create",
                                     "notes_update", "notes_delete", "summarize"]]),
            f"{result['names']} -- {result['tools']} tools and one prompt -- lint at "
            f"{result['total']} findings. Every description carries 'Use when' and 'Do not "
            "use for', every string field is described, and every schema declares required",
        ),
        practice.Check(
            "FINDING: the monolithic version scores 0 too, if the key is not called action",
            all([result["action_key"] == 1, result["op_key"] == 0,
                 any("monolithic" in message for message in result["action_messages"])]),
            f"the identical {result['operations']}-operations-in-one tool scores "
            f"{result['action_key']} finding with its verb under 'action' and "
            f"{result['op_key']} under 'op'. lint_schema matches key == 'action' literally, "
            "so the anti-pattern the lesson names is detected by the name of the field "
            "rather than by its shape",
        ),
        practice.Check(
            "FINDING: the prompt is linted as though it were a tool",
            result["prompt_findings"] == 0,
            f"lint_tool reads name, description and input_schema off any dict, so the "
            f"summarize prompt is checked against the same rules and scores "
            f"{result['prompt_findings']} -- including 'Do not use for', which is "
            "tool-disambiguation advice a slash prompt has no counterpart to be "
            "disambiguated from. It passes because the wording was bent to satisfy a rule "
            "that does not apply to it",
        ),
        practice.Check(
            "FINDING: five atomic tools cost 10x the description budget of one",
            all([result["atomic_budget"] == 729, result["monolith_budget"] == 75,
                 result["budget_ratio"] == 10]),
            f"the atomic registry spends {result['atomic_budget']} characters of description "
            f"against the monolith's {result['monolith_budget']} -- {result['budget_ratio']}x "
            "-- and all of it reaches the model on every request. Atomicity is the right call "
            "and it is not free, which is the trade the lesson's atomic-vs-monolithic section "
            "asserts without pricing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
