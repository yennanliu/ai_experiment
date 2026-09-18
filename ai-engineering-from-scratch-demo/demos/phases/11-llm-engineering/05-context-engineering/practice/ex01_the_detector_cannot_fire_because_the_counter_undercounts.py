"""Exercise 1 — the detector cannot fire, because the tool block is counted as six tokens.

    Add a "token waste detector" to the ContextBudget class. It should flag
    components using more than 30% of the budget and suggest compression
    strategies specific to each component type (summarize history, prune tools,
    re-rank documents).

Reading of the exercise: "the budget" is `ContextBudget.available` --
`max_tokens - generation_reserve`, the number every other method in the class
divides by -- and the detector is run over the allocations
`ContextEngine.assemble` actually produces, for five queries covering the five
intents `classify_intent` knows.

**ANSWER: it never fires.** The largest component across all five queries is 96
tokens against an available budget of 124,000 -- 0.08%. Total utilisation runs
0.079% to 0.114%. Nothing is within two orders of magnitude of 30%.

**FINDING: read as a share of tokens actually used, it fires every time, and
always on the same component.** `retrieved_context` is 52% to 66% of the used
tokens on all five queries, and no other component ever passes 30%. The
detector's answer is decided by which denominator you pick.

**MECHANISM: `count_tokens` is `int(len(text.split()) * 1.3)`.**
`assemble` serialises the selected tools as `json.dumps(list(tools.keys()))` --
a comma-separated list with almost no spaces -- so the tool block books 1 to 6
tokens while `TOOL_REGISTRY` declares 140 to 710 for the same tools. On the
calendar query that is 1 against 180, a 180x under-count, on exactly the
component whose compression strategy the exercise names.

**FINDING: substitute the declared costs and the ranking inverts.** `tools`
is flagged on all 5 queries, and on 4 of them it displaces `retrieved_context`
entirely. The exercise's "prune tools" suggestion is unreachable
under the shipped counter and is the only correct one under the declared cost.

**CONTROL: run the detector on what `assemble` reserves, not on what it books.**
The per-component caps it passes to `allocate` sum to 11,500 tokens: 9.3% of a
128,000-token available budget, and 274% of an 8,192-token one. Against the
small window it flags three components at once -- `tools` at 48%, history at
119%, documents at 72% -- each reserving more than a third of everything
available. Those are the numbers the code itself already believes.

Structure: `detect` is the detector, `shares` computes both denominators, and
`declared` swaps in the registry's own token counts.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "05-context-engineering"
QUERIES = ["how do I fix the failing test", "schedule a meeting for tuesday",
           "what is the api rate limit", "query the database for user stats",
           "send an email to the team"]
THRESHOLD = 0.30
SUGGESTION = {"conversation_history": "summarize history", "tools": "prune tools",
              "retrieved_context": "re-rank documents", "system_prompt": "shorten the prompt",
              "user_query": "none: the query is the task"}
SMALL_WINDOW, RESERVE = 8192, 4000
# the per-component caps `assemble` passes to allocate(), in call order
CAPS = {"system_prompt": 1000, "tools": 2000, "retrieved_context": 3000,
        "conversation_history": 5000, "user_query": 500}


def detect(allocations, denominator, threshold=THRESHOLD):
    """The detector: components over the threshold, with the strategy for each."""
    return {name: SUGGESTION[name] for name, tokens in allocations.items()
            if denominator and tokens / denominator > threshold}


def shares(allocations, available):
    used = sum(allocations.values())
    return {"of_budget": {n: round(t / available, 5) for n, t in allocations.items()},
            "of_used": {n: round(t / used, 3) for n, t in allocations.items()}}


def declared(ref, query, allocations):
    """The same allocations with the tool block priced at TOOL_REGISTRY's own numbers."""
    _, tool_tokens = ref.select_tools(query, token_budget=2000)
    return dict(allocations, tools=tool_tokens)


def run(ref, engine, query):
    budget = engine.assemble(query)
    allocations = dict(budget.allocations)
    priced = declared(ref, query, allocations)
    return {"allocations": allocations, "priced": priced,
            "available": budget.available, "utilization": round(budget.utilization() * 100, 3),
            **shares(allocations, budget.available),
            "flagged_budget": detect(allocations, budget.available),
            "flagged_used": detect(allocations, sum(allocations.values())),
            "priced_flagged": detect(priced, sum(priced.values()))}


def cap_report(available):
    """The detector run on the caps `assemble` passes, not on the tokens it books."""
    return {"caps_total": sum(CAPS.values()),
            "caps_big": round(sum(CAPS.values()) / available, 3),
            "caps_small": round(sum(CAPS.values()) / (SMALL_WINDOW - RESERVE), 2),
            "cap_flagged": sorted(detect(CAPS, SMALL_WINDOW - RESERVE))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    engine = ref.ContextEngine()
    rows = [run(ref, engine, q) for q in QUERIES]
    return {**cap_report(rows[0]["available"]), **columns(rows),
            "available": rows[0]["available"], "suggestion": SUGGESTION["tools"],
            "largest": max(max(r["allocations"].values()) for r in rows)}


def columns(rows):
    return {"utilization": [r["utilization"] for r in rows],
            "flagged_budget": [sorted(r["flagged_budget"]) for r in rows],
            "flagged_used": [sorted(r["flagged_used"]) for r in rows],
            "retrieved_share": [r["of_used"]["retrieved_context"] for r in rows],
            "booked_tools": [r["allocations"]["tools"] for r in rows],
            "declared_tools": [r["priced"]["tools"] for r in rows],
            "priced_flagged": [sorted(r["priced_flagged"]) for r in rows]}


def verify(result):
    booked, real = result["booked_tools"], result["declared_tools"]
    ratio = max(d / b for b, d in zip(booked, real))
    return [
        practice.Check(
            "ANSWER: the detector never fires against the budget",
            all([result["flagged_budget"] == [[]] * len(QUERIES),
                 max(result["utilization"]) < 1]),
            f"the largest component across the five queries is {result['largest']} tokens "
            f"against an available budget of {result['available']:,} -- and total "
            f"utilisation runs {min(result['utilization'])}% to "
            f"{max(result['utilization'])}%. Nothing is within two orders of magnitude of "
            f"{THRESHOLD:.0%}",
        ),
        practice.Check(
            "FINDING: against used tokens it fires every time, always on the same component",
            all([all("retrieved_context" in f for f in result["flagged_used"]),
                 min(result["retrieved_share"]) > THRESHOLD]),
            f"`retrieved_context` is {min(result['retrieved_share']):.0%} to "
            f"{max(result['retrieved_share']):.0%} of the tokens actually used on all five "
            f"queries, and nothing else passes {THRESHOLD:.0%}. The detector's answer is "
            "decided by which denominator you pick, and the exercise names neither",
        ),
        practice.Check(
            "MECHANISM: the tool block is counted as a handful of words",
            all([max(booked) < 10, min(real) > 100, ratio > 100]),
            f"`count_tokens` is int(len(text.split()) * 1.3), and `assemble` serialises the "
            f"tools as json.dumps(list(tools.keys())), which has almost no spaces. Booked: "
            f"{booked}. Declared by TOOL_REGISTRY for the same tools: {real}. Worst "
            f"under-count {ratio:.0f}x, on the component the exercise names first",
        ),
        practice.Check(
            "FINDING: substitute the declared costs and the ranking inverts",
            all([all("tools" in f for f in result["priced_flagged"]),
                 result["suggestion"] == "prune tools"]),
            f"pricing the tool block at the registry's own numbers flags `tools` on all "
            f"five queries: {result['priced_flagged']}. The "
            f"{result['suggestion']!r} strategy is unreachable under the shipped counter and "
            "is the only correct one under the declared cost",
        ),
        practice.Check(
            "CONTROL: run it on what assemble reserves rather than on what it books",
            all([result["cap_flagged"] == ["conversation_history", "retrieved_context",
                                            "tools"],
                 result["caps_big"] < THRESHOLD, result["caps_small"] > 2]),
            f"the per-component caps `assemble` passes to allocate sum to "
            f"{result['caps_total']:,} tokens -- {result['caps_big']:.1%} of a 128,000-token "
            f"available budget, so the detector stays silent there too, and "
            f"{result['caps_small']:.0%} of an 8,192-token one, where they cannot all be "
            f"honoured. Against that window the detector flags {result['cap_flagged']} -- "
            "three components each reserving more than a third of everything available, "
            "which is the finding the exercise was asking for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
