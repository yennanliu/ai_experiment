"""
Extended tool set for app2.

Adds artifact I/O (write_artifact / read_artifact / list_artifacts) and
search_memory on top of app1's calculate / remember / recall primitives.
"""

import json
import math
from pathlib import Path

_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

SCHEMAS = [
    # ── Core (inherited from app1) ────────────────────────────────────────────
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python math expression, e.g. '15 / 100 * 847'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "remember",
        "description": "Store a named fact in session memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall",
        "description": "Retrieve a fact from session memory by name",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    # ── Advanced: memory search ───────────────────────────────────────────────
    {
        "name": "search_memory",
        "description": "Search session memory for all keys containing a substring",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Substring to match against memory keys (case-insensitive)",
                }
            },
            "required": ["query"],
        },
    },
    # ── Advanced: artifact I/O ────────────────────────────────────────────────
    {
        "name": "write_artifact",
        "description": (
            "Write a design artifact to disk. "
            "Use for every plan deliverable (specs, designs, implementation plans)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Artifact slug — no extension, snake_case (e.g. 'auth_design')",
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown content to save",
                },
            },
            "required": ["name", "content"],
        },
    },
    {
        "name": "read_artifact",
        "description": "Read a previously written artifact from disk",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Artifact slug without extension",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_artifacts",
        "description": "List all artifacts currently saved to disk",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def run(name: str, inputs: dict, memory: dict) -> str:
    if name == "calculate":
        try:
            safe_globals = {"__builtins__": {}} | vars(math)
            result = eval(inputs["expression"], safe_globals)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    if name == "remember":
        memory[inputs["key"]] = inputs["value"]
        return f"Stored '{inputs['key']}'"

    if name == "recall":
        return memory.get(inputs["key"], "Not found")

    if name == "search_memory":
        query = inputs["query"].lower()
        matches = {k: v for k, v in memory.items() if query in k.lower()}
        return json.dumps(matches) if matches else "No matches found"

    if name == "write_artifact":
        _ARTIFACTS_DIR.mkdir(exist_ok=True)
        path = _ARTIFACTS_DIR / f"{inputs['name']}.md"
        path.write_text(inputs["content"])
        return f"Written: {path.name} ({len(inputs['content'])} chars)"

    if name == "read_artifact":
        path = _ARTIFACTS_DIR / f"{inputs['name']}.md"
        return path.read_text() if path.exists() else f"Artifact '{inputs['name']}' not found"

    if name == "list_artifacts":
        _ARTIFACTS_DIR.mkdir(exist_ok=True)
        files = sorted(f.stem for f in _ARTIFACTS_DIR.glob("*.md"))
        return ", ".join(files) if files else "No artifacts yet"

    return f"Unknown tool: {name}"
