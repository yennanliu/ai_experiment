"""
Tool registry for app2.

Two exported schema groups keep each agent's tool surface focused:
  SCHEMAS      — memory + design-artifact tools  (design Generator)
  CODE_SCHEMAS — memory + workspace code tools   (Software Eng Agent)

Both groups share the same run() dispatcher so no logic is duplicated.
"""

import json
import math
from pathlib import Path

_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
_WORKSPACE_DIR = Path(__file__).parent.parent / "workspace"

# ── Schema definitions ─────────────────────────────────────────────────────────

_CALCULATE = {
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
            "query": {
                "type": "string",
                "description": "Substring to match against memory keys (case-insensitive)",
            }
        },
        "required": ["query"],
    },
}

_WRITE_ARTIFACT = {
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
}

_READ_ARTIFACT = {
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
}

_LIST_ARTIFACTS = {
    "name": "list_artifacts",
    "description": "List all artifacts currently saved to disk",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

_WRITE_CODE = {
    "name": "write_code",
    "description": (
        "Write a Python source file to the workspace. "
        "filename must include the .py extension (e.g. 'rate_limiter.py')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Target filename with .py extension",
            },
            "content": {
                "type": "string",
                "description": "Complete Python source code",
            },
        },
        "required": ["filename", "content"],
    },
}

_READ_CODE = {
    "name": "read_code",
    "description": "Read an existing workspace source file",
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename with extension (e.g. 'rate_limiter.py')",
            }
        },
        "required": ["filename"],
    },
}

_LIST_WORKSPACE = {
    "name": "list_workspace",
    "description": "List all source files currently in the workspace",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

# ── Exported groups ────────────────────────────────────────────────────────────

_MEMORY = [_CALCULATE, _REMEMBER, _RECALL, _SEARCH_MEMORY]

SCHEMAS = _MEMORY + [_WRITE_ARTIFACT, _READ_ARTIFACT, _LIST_ARTIFACTS]
CODE_SCHEMAS = _MEMORY + [_WRITE_CODE, _READ_CODE, _LIST_WORKSPACE]

# ── Dispatcher ─────────────────────────────────────────────────────────────────

def run(name: str, inputs: dict, memory: dict) -> str:
    # Memory tools
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

    # Design artifact tools
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

    # Code / workspace tools
    if name == "write_code":
        _WORKSPACE_DIR.mkdir(exist_ok=True)
        path = _WORKSPACE_DIR / inputs["filename"]
        path.write_text(inputs["content"])
        lines = inputs["content"].count("\n") + 1
        return f"Written: {path.name} ({lines} lines)"

    if name == "read_code":
        path = _WORKSPACE_DIR / inputs["filename"]
        return path.read_text() if path.exists() else f"File '{inputs['filename']}' not found"

    if name == "list_workspace":
        _WORKSPACE_DIR.mkdir(exist_ok=True)
        files = sorted(f.name for f in _WORKSPACE_DIR.iterdir() if f.suffix == ".py")
        return ", ".join(files) if files else "Workspace is empty"

    return f"Unknown tool: {name}"
