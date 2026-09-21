"""Exercise 5 — the dashboard needs two attributes the spans do not carry.

    Build a dashboard: "which tool errors correlate with which models" from
    GenAI attributes alone.

Reading of the exercise: "from GenAI attributes alone" is the constraint that
makes it a design exercise rather than a query. A backend receives flat spans
and groups by attribute -- it cannot walk a tree. So the dashboard is built
twice: once against the spans the lesson emits, where it returns nothing, and
once against spans carrying the two attributes it needs.

**ANSWER: the query is impossible as shipped and trivial after 2 attributes
are added.** The lesson's `tool_call` spans carry **3** attributes -- none an
error, none a model -- so grouping tool failures by model yields **0** rows
from **10** tool spans. Adding `error.type` and `gen_ai.request.model` to the
tool span makes the same grouping return a full table over **4** tools and
**2** models.

**FINDING: the model that failed is in a different span from the tool that
failed.** `gen_ai.request.model` lives on `chat` and `invoke_agent`; the
error lives on `tool_call`. Joining them needs the parent link, and `Span`
has **0** id fields -- so the correlation is one `SELECT` for a backend that
has parent ids and impossible for one that does not. Propagating the model
down onto the tool span is the fix, and it is duplication on purpose.

**FINDING: the shipped emitter has no error channel at all.** `Span` has
**6** fields and **0** of them is a status; across every span in the lesson's
own `main()` there are **0** attributes matching `error` or `status`. A trace
where nothing can fail is a trace where the dashboard's subject does not
exist.

**FINDING: with the attributes present the correlation is real and lopsided.**
Over **40** seeded tool calls, `claude-opus-4-6` fails **20.0%** and
`gpt-5-mini` **65.0%**, and the failures concentrate: `web_search` fails
**80.0%** on the weaker model against **20.0%** on the stronger. A
single-number tool error rate of **42.5%** describes neither -- which is the
whole argument for the group-by over the headline.

Structure: `dashboard()` is the group-by a backend would run; `TRACE` is the
seeded fixture it runs against.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "23-otel-genai-conventions"
MODELS = ("claude-opus-4-6", "gpt-5-mini")
TOOLS = ("web_search", "calculator", "kv_get", "sql_query")
# Failure probability per (model, tool): the weaker model is worse at search.
RATES = {("claude-opus-4-6", "web_search"): 0.1,
         ("claude-opus-4-6", "calculator"): 0.1,
         ("claude-opus-4-6", "kv_get"): 0.1,
         ("claude-opus-4-6", "sql_query"): 0.1,
         ("gpt-5-mini", "web_search"): 0.6,
         ("gpt-5-mini", "calculator"): 0.4,
         ("gpt-5-mini", "kv_get"): 0.4,
         ("gpt-5-mini", "sql_query"): 0.4}
CALLS = 40


def emit(ref, enriched):
    """The trace a backend receives: tool spans, with or without the two attributes."""
    tracer = ref.Tracer(capture_inline=False)
    rng = random.Random(7)
    for index in range(CALLS):
        model, tool = MODELS[index % 2], TOOLS[(index // 2) % 4]
        failed = rng.random() < RATES[(model, tool)]
        attrs = {"gen_ai.operation.name": "tool_call", "gen_ai.tool.name": tool,
                 "gen_ai.data_source.id": "corpus://mem0/default"}
        if enriched:
            attrs["gen_ai.request.model"] = model
            attrs["error.type"] = "ToolError" if failed else ""
        tracer.start_span(f"tool_call {tool}", attributes=attrs)
        tracer.end_span()
    return tracer


def spans(tracer):
    return [child for child in tracer.root.children]


def dashboard(rows):
    """Group tool failures by model, the way a backend groups by attribute."""
    table = {}
    for span in rows:
        model = span.attributes.get("gen_ai.request.model")
        error = span.attributes.get("error.type")
        if model is None or error is None:
            continue
        key = (model, span.attributes["gen_ai.tool.name"])
        calls, fails = table.get(key, (0, 0))
        table[key] = (calls + 1, fails + bool(error))
    return table


def rate(table, model=None, tool=None):
    rows = [(c, f) for (m, t), (c, f) in table.items()
            if (model is None or m == model) and (tool is None or t == tool)]
    calls = sum(c for c, _ in rows)
    return round(100 * sum(f for _, f in rows) / calls, 1) if calls else 0.0


def shape(ref, shipped_attrs):
    """What the emitter can express, as field and attribute names."""
    fields = list(ref.Span.__dataclass_fields__)
    return {
        "span_fields": fields, "shipped_attrs": shipped_attrs,
        "status_fields": [f for f in fields if "status" in f or "error" in f],
        "id_fields": [f for f in fields if f.endswith("id")],
        "error_attrs": [a for a in shipped_attrs if "error" in a or "status" in a],
        "model_attrs": [a for a in shipped_attrs if "model" in a],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plain_rows, rich_rows = spans(emit(ref, False)), spans(emit(ref, True))
    table = dashboard(rich_rows)
    return {
        "tool_spans": len(plain_rows), "plain_rows": len(dashboard(plain_rows)),
        "rich_rows": len(table), "calls": sum(c for c, _ in table.values()),
        "by_model": {model: rate(table, model=model) for model in MODELS},
        "search_split": {model: rate(table, model=model, tool="web_search")
                         for model in MODELS},
        "overall": rate(table), "tools": len({t for _, t in table}),
        "models": len({m for m, _ in table}),
        **shape(ref, sorted(plain_rows[0].attributes)),
    }


def verify(result):
    by_model, split = result["by_model"], result["search_split"]
    return [
        practice.Check(
            "ANSWER: impossible as shipped, trivial after two attributes are added",
            all([result["plain_rows"] == 0, result["tool_spans"] == 40,
                 result["rich_rows"] == 8, result["tools"] == 4,
                 result["models"] == 2, result["calls"] == 40,
                 result["shipped_attrs"] == ["gen_ai.data_source.id",
                                             "gen_ai.operation.name",
                                             "gen_ai.tool.name"]]),
            f"the shipped tool_call span carries {result['shipped_attrs']}, so grouping "
            f"failures by model over {result['tool_spans']} spans returns "
            f"{result['plain_rows']} rows. Adding error.type and gen_ai.request.model "
            f"makes the same grouping return {result['rich_rows']} rows across "
            f"{result['tools']} tools and {result['models']} models",
        ),
        practice.Check(
            "FINDING: the model that failed is in a different span from the tool",
            all([result["model_attrs"] == [], result["id_fields"] == [],
                 len(result["span_fields"]) == 6]),
            f"gen_ai.request.model lives on chat and invoke_agent while the error is on "
            f"tool_call, and the shipped tool span names a model "
            f"{len(result['model_attrs'])} times. Joining them needs a parent link, and "
            f"Span has {len(result['id_fields'])} id fields -- so propagating the model "
            "down onto the tool span is the fix, and it is duplication on purpose",
        ),
        practice.Check(
            "FINDING: the shipped emitter has no error channel at all",
            all([result["status_fields"] == [], result["error_attrs"] == [],
                 len(result["span_fields"]) == 6]),
            f"Span carries {result['span_fields']} -- {len(result['status_fields'])} of "
            f"them a status -- and the shipped tool attributes include "
            f"{len(result['error_attrs'])} matching error or status. A trace where "
            "nothing can fail is a trace where the dashboard's subject does not exist",
        ),
        practice.Check(
            "FINDING: the correlation is real and lopsided",
            all([by_model == {"claude-opus-4-6": 20.0, "gpt-5-mini": 65.0},
                 split == {"claude-opus-4-6": 20.0, "gpt-5-mini": 80.0},
                 result["overall"] == 42.5]),
            f"over {result['calls']} seeded tool calls the failure rate by model is "
            f"{by_model}, and web_search splits {split} -- the weaker model's worst tool "
            f"against the stronger model's flat rate. A single-number tool error rate of "
            f"{result['overall']}% describes neither, which is why the group-by is the "
            "dashboard and the headline is not",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
