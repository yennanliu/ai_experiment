"""Exercise 4 — 20 and 50 tools do not exist, and all 10 fit inside the budget.

    Implement a priority-based tool selector. Instead of binary
    include/exclude, assign each tool a relevance score to the current query.
    Include tools in descending relevance order until the tool budget is
    exhausted. Compare task performance with 5, 10, 20, and 50 tools included.

Reading of the exercise: relevance is scored against the tool's description and
categories with the lesson's own `score_relevance`, and the budget is the 2,000
tokens `ContextEngine.assemble` passes to `select_tools`. "Task performance"
has no observable here -- there is no task and no model -- so what is compared
is the selected set, which is the only thing either selector decides.

**ANSWER: two of the four points do not exist.** `TOOL_REGISTRY` holds 10
tools. There is no way to include 20 or 50 of them.

**FINDING: the third point is free.** All ten tools together declare 1,580
tokens against the 2,000-token budget, so the budget cannot bind on the whole
registry. `select_tools` is a category filter with a dead accumulator: for the
five intent-covering queries it returns 5, 1, 1, 2 and 2 tools at 710, 180,
140, 360 and 360 tokens, every one of them under budget.

**FINDING: the priority selector chooses the same set for every budget that
matters.** Ordering by relevance instead of by `TOOL_REGISTRY` insertion order
changes which tool is dropped first, but the two selectors return identical sets
at the shipped budget on all five queries. The largest budget at which they
differ is 630 tokens -- below anything the lesson uses.

**FINDING: `classify_intent` falls back to `["code"]` when nothing matches.**
"Who won the 1998 World Cup?" matches no keyword in any category, so the
selector returns the five code tools -- 710 tokens of file, search and shell
access -- for a football question.
The binary include/exclude the exercise wants to replace is not the problem; the
default is.

**ANSWER: task performance cannot be compared.** The lesson has no task and no
model, so the only observable is the set -- and across 5, 10, 20 and 50 the set
is 5-or-fewer, 10, undefined, undefined.

Structure: `priority` is the exercise's selector, `binding` finds the budget at
which it first disagrees with the shipped one, and `FALLBACK` is the default
query.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "05-context-engineering"
QUERIES = ["how do I fix the failing test", "schedule a meeting for tuesday",
           "what is the api rate limit", "query the database for user stats",
           "send an email to the team"]
FALLBACK = "Who won the 1998 World Cup?"
BUDGET = 2000
ASKED_FOR = (5, 10, 20, 50)


def relevance(ref, query, registry):
    """Score each tool against the query, over its description and its categories."""
    names = list(registry)
    blurbs = [f"{registry[n]['description']} {' '.join(registry[n]['categories'])}"
              for n in names]
    return dict(zip(names, ref.score_relevance(query, blurbs)))


def priority(ref, query, registry, token_budget=BUDGET):
    """The exercise's selector: descending relevance, until the budget is exhausted."""
    scores = relevance(ref, query, registry)
    intents = ref.classify_intent(query)
    eligible = [n for n in registry if any(c in intents for c in registry[n]["categories"])]
    chosen, spent = [], 0
    for name in sorted(eligible, key=lambda n: (-scores[n], n)):
        if spent + registry[name]["tokens"] <= token_budget:
            chosen.append(name)
            spent += registry[name]["tokens"]
    return chosen, spent


def binding(ref, query, registry):
    """The largest budget at which the two selectors return different sets."""
    for budget in range(BUDGET, 0, -10):
        shipped = set(ref.select_tools(query, budget)[0])
        if shipped != set(priority(ref, query, registry, budget)[0]):
            return budget
    return 0


def compare(shipped, ordered):
    return {"counts": [len(tools) for tools, _ in shipped],
            "costs": [cost for _, cost in shipped],
            "same_set": [set(t) == set(n) for (t, _), (n, _) in zip(shipped, ordered)],
            "same_order": [list(t) == n for (t, _), (n, _) in zip(shipped, ordered)]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    registry = ref.TOOL_REGISTRY
    shipped = [ref.select_tools(q, BUDGET) for q in QUERIES]
    fallback_tools, fallback_cost = ref.select_tools(FALLBACK, BUDGET)
    return {
        "registry": len(registry), "asked_for": list(ASKED_FOR), "budget": BUDGET,
        "unreachable": [n for n in ASKED_FOR if n > len(registry)],
        "total_tokens": sum(t["tokens"] for t in registry.values()),
        **compare(shipped, [priority(ref, q, registry) for q in QUERIES]),
        "binding": max(binding(ref, q, registry) for q in QUERIES),
        "fallback_intents": ref.classify_intent(FALLBACK),
        "fallback_tools": sorted(fallback_tools), "fallback_cost": fallback_cost,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two of the four comparison points do not exist",
            all([result["registry"] == 10, result["unreachable"] == [20, 50]]),
            f"TOOL_REGISTRY holds {result['registry']} tools and the exercise asks to "
            f"compare {result['asked_for']}; {result['unreachable']} cannot be reached. "
            "There is no larger registry anywhere in the lesson to draw them from",
        ),
        practice.Check(
            "FINDING: the ten-tool point is free -- the budget cannot bind",
            all([result["total_tokens"] < result["budget"], max(result["costs"]) < 800]),
            f"all ten tools together declare {result['total_tokens']} tokens against the "
            f"{result['budget']}-token budget `assemble` passes, so the accumulator in "
            f"`select_tools` is dead code. Per query it returns {result['counts']} tools at "
            f"{result['costs']} tokens -- every one under budget",
        ),
        practice.Check(
            "FINDING: the priority selector picks the same set at every budget that matters",
            all([result["same_set"] == [True] * len(QUERIES),
                 result["binding"] < result["budget"] // 2]),
            f"ordering by relevance rather than by insertion order gives the same set on "
            f"all five queries ({result['same_set']}) and a different order on "
            f"{sum(not o for o in result['same_order'])} of them. The largest budget at "
            f"which the two disagree on the set is {result['binding']} tokens, well below "
            "anything the lesson uses",
        ),
        practice.Check(
            "FINDING: classify_intent falls back to code, so a weather question gets a shell",
            all([result["fallback_intents"] == ["code"], result["fallback_cost"] == 710,
                 "run_command" in result["fallback_tools"]]),
            f"{FALLBACK!r} matches no keyword in any category, and the classifier returns "
            f"{result['fallback_intents']} rather than nothing. The selector then ships "
            f"{result['fallback_tools']} -- {result['fallback_cost']} tokens of file, "
            "search and shell access. The include/exclude is not the problem; the default is",
        ),
        practice.Check(
            "ANSWER: task performance has no observable to compare",
            all([len(result["counts"]) == len(QUERIES), max(result["counts"]) <= 5]),
            f"the lesson has no task and no model, so the only thing either selector "
            f"decides is the set, and the set never exceeds {max(result['counts'])} tools "
            f"for a real query. Across the four requested points the answer is: fewer than "
            "5, exactly 10, undefined, undefined",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
