"""Exercise 1 — the sixth tool registers, describes itself, and can never be called.

    **Add a 6th tool: database query.** Implement a simulated SQL tool with an
    in-memory table. The tool accepts a table name and filter conditions (not
    raw SQL). Validate that the table name is in an allowlist and that filter
    operators are restricted to `=`, `>`, `<`, `>=`, `<=`. Return matching rows
    as JSON.

Reading of the exercise: the tool is registered through the lesson's own
`register_tool` with a JSON-Schema definition like the other five, and both
allowlists are enforced inside the function body -- which is where they have to
live, and that is the second finding.

**ANSWER: it works, and nothing can call it.** `simulate_model_decision` is a
fixed if-chain over five keyword branches with a `return []` fallback. Adding a
sixth entry to `TOOL_REGISTRY` puts it in the `tool_definitions` list the
decision function receives and ignores: across 8 database-shaped queries, 0
route to `query_db`. The tool is registered, described, exposed, and unreachable.

**FINDING: the validation has to live inside the function, because the execution
path never validates.** `validate_tool_arguments` exists and is called exactly
once in the whole module -- inside `run_demo`. `execute_tool_call` calls
`func(**args)` directly, so a tool that does not check its own arguments is not
checked at all.

**FINDING: the schema can express one allowlist and not the other.** Of six
invalid payloads the validator catches 2 -- both bad table names, via `enum` --
and passes 4, every one of them a filter problem, because `filters: []` and
`filters: [{"anything": 1}]` are both just arrays to it. The function body
rejects all 6, with codes TABLE, FILTER and OPERATOR.

**FINDING: the registry's own count moves and nothing downstream notices.**
`TOOL_REGISTRY` goes from 5 entries to 6 and `run_function_calling_loop` builds
its `tool_definitions` from it every call, so the sixth definition is sent to a
decision function whose branch list is a literal.

**CONTROL: one branch makes it reachable.** Adding a `("table", "query", "rows",
"where")` test to the front of the chain routes 8 of the 8 database queries to
`query_db`, and the tool then answers them correctly.

Structure: `query_db` is the new tool, `ALLOWED_TABLES` and `OPERATORS` are the
two allowlists, and `routed` counts what the decision function does with it.
"""

from __future__ import annotations

import inspect
import json

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "09-function-calling"
ALLOWED_TABLES = ("users", "orders")
OPERATORS = ("=", ">", "<", ">=", "<=")
TABLES = {"users": [{"name": "Alice", "age": 34, "role": "admin"},
                    {"name": "Bob", "age": 27, "role": "user"},
                    {"name": "Cara", "age": 41, "role": "user"}],
          "orders": [{"id": 1, "total": 120}, {"id": 2, "total": 45}]}
QUERIES = ["query the users table", "show rows in orders where total > 100",
           "select users where age >= 30", "list the orders table",
           "database query for admins", "find rows in users where role = admin",
           "what rows are in orders", "query orders where total < 50"]
BAD_PAYLOADS = [{"table": "secrets", "filters": []},
                {"table": "users", "filters": [["age", "LIKE", 30]]},
                {"table": "users", "filters": [["age", ";DROP", 30]]},
                {"table": "users", "filters": [{"anything": 1}]},
                {"table": "users; DROP TABLE users", "filters": []},
                {"table": "users", "filters": [["age", ">"]]}]
DEFINITION = {"type": "object", "required": ["table"],
              "properties": {"table": {"type": "string", "enum": list(ALLOWED_TABLES)},
                             "filters": {"type": "array"}}}


def compare(value, operator, wanted):
    return {"=": value == wanted, ">": value > wanted, "<": value < wanted,
            ">=": value >= wanted, "<=": value <= wanted}[operator]


def apply_filter(rows, condition):
    """One condition, or an error code naming what was wrong with it."""
    if not isinstance(condition, (list, tuple)) or len(condition) != 3:
        return None, "FILTER"
    column, operator, wanted = condition
    if operator not in OPERATORS:
        return None, "OPERATOR"
    return [r for r in rows if column in r and compare(r[column], operator, wanted)], None


def query_db(table, filters=None):
    """The sixth tool: an allowlisted table name and operator-restricted filters."""
    if table not in ALLOWED_TABLES:
        return {"error": True, "message": f"Table '{table}' not allowed.", "code": "TABLE"}
    rows = TABLES[table]
    for condition in filters or []:
        rows, code = apply_filter(rows, condition)
        if code:
            return {"error": True, "message": f"Bad filter: {condition!r}", "code": code}
    return {"table": table, "rows": json.loads(json.dumps(rows)), "count": len(rows)}


def with_branch(ref):
    """The control: one keyword branch at the front of the chain."""
    original = ref.simulate_model_decision

    def decide(message, tools, history):
        if any(word in message.lower() for word in ("table", "query", "rows", "where")):
            return [{"name": "query_db", "arguments": {"table": "users"}}]
        return original(message, tools, history)
    return decide


def routed(decide, name="query_db"):
    return sum(any(call["name"] == name for call in decide(q, [], [])) for q in QUERIES)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "function_calling")
    ref.register_all_tools()
    before = len(ref.TOOL_REGISTRY)
    ref.register_tool("query_db", "Query an allowlisted table with simple filters",
                      DEFINITION, query_db)
    source = inspect.getsource(ref)
    validated = [ref.validate_tool_arguments("query_db", payload) for payload in BAD_PAYLOADS]
    rejected = [query_db(**payload).get("code") for payload in BAD_PAYLOADS]
    return {
        "before": before, "after": len(ref.TOOL_REGISTRY),
        "routed": routed(ref.simulate_model_decision), "queries": len(QUERIES),
        "routed_with_branch": routed(with_branch(ref)),
        "validate_calls": source.count("validate_tool_arguments(") - 1,
        "executes_directly": "func(**args)" in inspect.getsource(ref.execute_tool_call),
        "validator_passed": sum(not errors for errors in validated),
        "validator_caught": [i for i, errors in enumerate(validated) if errors],
        "body_rejected": sum(code is not None for code in rejected),
        "body_codes": sorted({code for code in rejected if code}),
        "payloads": len(BAD_PAYLOADS),
        "works": query_db("users", [["age", ">", 30]]),
        "definitions": len([t["definition"] for t in ref.TOOL_REGISTRY.values()]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the sixth tool registers and nothing can call it",
            all([result["after"] == result["before"] + 1, result["routed"] == 0,
                 result["works"]["count"] == 2]),
            f"`TOOL_REGISTRY` goes {result['before']} -> {result['after']} and the tool "
            f"answers correctly -- users where age > 30 gives {result['works']['count']} "
            f"rows -- but {result['routed']} of {result['queries']} database-shaped queries "
            "route to it. `simulate_model_decision` is a fixed if-chain over five keyword "
            "branches with a `return []` fallback",
        ),
        practice.Check(
            "FINDING: the execution path never validates, so the tool must validate itself",
            all([result["validate_calls"] == 1, result["executes_directly"]]),
            f"`validate_tool_arguments` is called {result['validate_calls']} time in the "
            "whole module, inside `run_demo`. `execute_tool_call` runs func(**args) "
            "directly, so a tool that does not check its own arguments is not checked -- "
            "which is why both allowlists live in the function body",
        ),
        practice.Check(
            "FINDING: the schema expresses one allowlist and not the other",
            all([result["validator_caught"] == [0, 4], result["validator_passed"] == 4,
                 result["body_rejected"] == result["payloads"]]),
            f"of the {result['payloads']} invalid payloads, the validator catches "
            f"{len(result['validator_caught'])} -- both bad table names, via the enum -- and "
            f"passes the other {result['validator_passed']}, every one of them a filter "
            f"problem. The body rejects all {result['body_rejected']} with codes "
            f"{result['body_codes']}. A JSON Schema of type/required/enum can express a "
            "table allowlist and cannot express an operator one",
        ),
        practice.Check(
            "FINDING: the definition list grows and the branch list is a literal",
            result["definitions"] == result["after"],
            f"`run_function_calling_loop` rebuilds tool_definitions from the registry on "
            f"every call, so all {result['definitions']} definitions are handed to the "
            "decision function -- which never reads its `tools` parameter. Registration and "
            "routing are two unconnected mechanisms here",
        ),
        practice.Check(
            "CONTROL: one branch makes it reachable",
            result["routed_with_branch"] == result["queries"],
            f"a single ('table', 'query', 'rows', 'where') test at the front of the chain "
            f"routes {result['routed_with_branch']} of {result['queries']} database queries "
            "to `query_db`. The tool was never the missing piece; the router was",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
