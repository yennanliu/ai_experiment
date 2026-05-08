import math

SCHEMAS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Python math expression, e.g. '15 / 100 * 847'"}
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
            "properties": {
                "key": {"type": "string"},
            },
            "required": ["key"],
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

    return f"Unknown tool: {name}"
