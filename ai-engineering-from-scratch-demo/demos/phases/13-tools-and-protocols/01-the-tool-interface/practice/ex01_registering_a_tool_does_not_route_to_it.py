"""Exercise 1 — registering a tool does not route to it.

    Add a fourth tool to `code/main.py` called `get_stock_price(ticker)`. Write
    its description as "Use when the user asks for a current stock price by
    ticker. Do not use for historical prices or market summaries." Run the
    harness and confirm the fake decider routes queries mentioning tickers to
    the new tool.

Reading of the exercise: the tool is added exactly as asked and the harness is
then run exactly as asked, because the instruction to "confirm the fake decider
routes" is a prediction and the lesson's own code refutes it. The fourth tool is
built here rather than by editing the lesson (D5), and routed against the
lesson's own `fake_decide`.

**ANSWER: the tool registers fine and is never chosen.** `get_stock_price` joins
the registry as a fourth entry whose schema validates `{"ticker": "AAPL"}`, and
`fake_decide` routes **0 of 5** ticker queries to it -- every one falls through
to "I cannot route that query to any registered tool."

**FINDING: `fake_decide` never reads the registry.** Stripping its docstring and
walking the AST, the names its code touches are `city, float, history, last,
len, match, msg, n, nums, re, user_msg, uuid` -- **`REGISTRY` is not among
them**. It appears in the function exactly once, in the docstring, as the
`tools=[t.input_schema for t in REGISTRY]` line describing what a real provider
would receive. Routing is three hard-coded keyword branches naming `add`,
`get_time` and `get_weather` as string literals.

**FINDING: so "describe" is the step the lesson does not implement.** The
four-step loop is describe -> decide -> execute -> observe, and `describe_registry`
prints the registry to the terminal rather than passing it to the decider. The
registry reaches the human and never reaches the chooser -- which is why adding
a tool changes what is printed and not what is called.

Structure: `stock_tool` builds the fourth tool the exercise specifies,
`decider_reads` walks `fake_decide`'s AST for the names its code uses, and
`routes` asks the lesson's own decider to pick a tool for each ticker query.
"""

from __future__ import annotations

import ast
import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "01-the-tool-interface"
DESCRIPTION = ("Use when the user asks for a current stock price by ticker. "
               "Do not use for historical prices or market summaries.")
TICKER_QUERIES = (
    "what is the stock price of AAPL",
    "price of TSLA ticker",
    "quote for NVDA",
    "how much is MSFT trading at",
    "get me the current stock price for GOOG",
)
PRICES = {"AAPL": 231.4, "TSLA": 412.05, "NVDA": 178.9, "MSFT": 447.2, "GOOG": 188.6}


def stock_tool(ref):
    """The fourth tool exactly as the exercise specifies it."""
    return ref.Tool(
        name="get_stock_price",
        description=DESCRIPTION,
        input_schema={"type": "object",
                      "properties": {"ticker": {"type": "string"}},
                      "required": ["ticker"]},
        executor=lambda args: {"ticker": args["ticker"],
                               "price": PRICES.get(args["ticker"], 0.0)},
    )


def decider_reads(ref):
    """Every name fake_decide's code uses, with its docstring stripped out."""
    body = ast.parse(inspect.getsource(ref.fake_decide)).body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return sorted({node.id for statement in body
                   for node in ast.walk(statement) if isinstance(node, ast.Name)})


def routes(ref):
    """What the lesson's own decider picks for each ticker query."""
    picked = []
    for query in TICKER_QUERIES:
        decision = ref.fake_decide(query, [])
        picked.append(decision["tool_calls"][0]["name"] if "tool_calls" in decision
                      else None)
    return picked


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tool = stock_tool(ref)
    registry = list(ref.REGISTRY) + [tool]
    names = decider_reads(ref)
    source = inspect.getsource(ref.fake_decide)
    return {
        "registry_size": len(registry),
        "tool_name": tool.name, "description": tool.description,
        "schema_ok": ref.validate(tool.input_schema, {"ticker": "AAPL"}) == [],
        "schema_rejects_empty": ref.validate(tool.input_schema, {}),
        "executes": tool.executor({"ticker": "AAPL"}),
        "routed": routes(ref),
        "routed_to_stock": sum(name == tool.name for name in routes(ref)),
        "queries": len(TICKER_QUERIES),
        "decider_names": names,
        "reads_registry": "REGISTRY" in names,
        "registry_in_docstring": "REGISTRY" in (ref.fake_decide.__doc__ or ""),
        "hardcoded": [name for name in ("add", "get_time", "get_weather")
                      if f'"{name}"' in source],
        "describe_prints": "print" in inspect.getsource(ref.describe_registry),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the tool registers fine and is never chosen",
            all([result["registry_size"] == 4, result["tool_name"] == "get_stock_price",
                 result["description"] == DESCRIPTION, result["schema_ok"],
                 result["schema_rejects_empty"] == ["missing required field 'ticker'"],
                 result["executes"] == {"ticker": "AAPL", "price": 231.4},
                 result["routed_to_stock"] == 0]),
            f"get_stock_price joins the registry as entry {result['registry_size']}, its "
            f"schema accepts {{'ticker': 'AAPL'}} and rejects {{}} with "
            f"{result['schema_rejects_empty']}, and it executes to {result['executes']}. "
            f"fake_decide then routes {result['routed_to_stock']} of {result['queries']} "
            f"ticker queries to it -- {result['routed']}",
        ),
        practice.Check(
            "FINDING: fake_decide never reads the registry",
            all([not result["reads_registry"], result["registry_in_docstring"],
                 result["hardcoded"] == ["add", "get_time", "get_weather"]]),
            f"stripping the docstring and walking the AST, the names its code touches are "
            f"{result['decider_names']} -- REGISTRY is not among them, though it does appear "
            f"in the docstring. Routing is three hard-coded keyword branches naming "
            f"{result['hardcoded']} as string literals, so the registry cannot add a fourth",
        ),
        practice.Check(
            "FINDING: describe is the step the lesson does not implement",
            all([result["describe_prints"], not result["reads_registry"]]),
            "the loop is describe -> decide -> execute -> observe, and describe_registry "
            "prints the registry to the terminal rather than passing it to the decider. The "
            "registry reaches the human and never reaches the chooser, which is why adding a "
            "tool changes what is printed and not what is called",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
