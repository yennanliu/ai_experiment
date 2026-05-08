"""
Tool registry for app3 — design-artifact tools only (no code workspace).
"""

import json
import math
from pathlib import Path

_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_CALCULATE = {
    "name": "calculate",
    "description": "Evaluate a mathematical expression",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Python math expression"},
        },
        "required": ["expression"],
    },
}

_REMEMBER = {
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
}

_RECALL = {
    "name": "recall",
    "description": "Retrieve a fact from session memory by name",
    "input_schema": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
}

_SEARCH_MEMORY = {
    "name": "search_memory",
    "description": "Search session memory for all keys containing a substring",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Substring to match (case-insensitive)"},
        },
        "required": ["query"],
    },
}

_WRITE_ARTIFACT = {
    "name": "write_artifact",
    "description": "Write a design artifact to disk (specs, designs, implementation plans)",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Artifact slug — snake_case, no extension"},
            "content": {"type": "string", "description": "Full markdown content"},
        },
        "required": ["name", "content"],
    },
}

_READ_ARTIFACT = {
    "name": "read_artifact",
    "description": "Read a previously written artifact from disk",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Artifact slug without extension"},
        },
        "required": ["name"],
    },
}

_LIST_ARTIFACTS = {
    "name": "list_artifacts",
    "description": "List all artifacts currently saved to disk",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

SCHEMAS = [_CALCULATE, _REMEMBER, _RECALL, _SEARCH_MEMORY, _WRITE_ARTIFACT, _READ_ARTIFACT, _LIST_ARTIFACTS]


def run(name: str, inputs: dict, memory: dict) -> str:
    if name == "calculate":
        try:
            safe_globals = {"__builtins__": {}} | vars(math)
            return str(eval(inputs["expression"], safe_globals))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    if name == "remember":
        memory[inputs["key"]] = inputs["value"]
        return f"Stored '{inputs['key']}'"

    if name == "recall":
        return memory.get(inputs["key"], "Not found")

    if name == "search_memory":
        q = inputs["query"].lower()
        matches = {k: v for k, v in memory.items() if q in k.lower()}
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
