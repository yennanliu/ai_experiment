"""Exercise 1 — the summarizer returns a full stop when nothing fits.

    Add a `block_summarize` tool that replaces the block value with a
    model-generated summary when `near_limit` returns true. Which trigger
    threshold minimizes both summarization calls and block overflow?

Reading of the exercise: `SleepTimeAgent.run` already does this inline --
`near_limit()` then `_summarize` then `rewrite` -- so the tool is that
sequence given a name and a threshold parameter. The threshold question is
then arithmetic: a check that runs *after* an append can only catch overflow
if the trigger leaves room for one more append, so the frontier is
`1 - longest_append / limit` and the sweep should find it.

**ANSWER: the frontier is **0.60**, and it is exactly `1 - longest/limit`.**
Over the same 12 appends against a 200-character block whose longest append
is **80** characters, thresholds **0.50** and **0.60** overflow **0** times
and **0.70** onward overflow **3** times each. Calls fall from **11** at 0.50
to **7** at 0.60 to **3** at 0.95, so **0.60** is the cheapest threshold that
never overflows -- and the shipped default of **0.8** overflows **3** times.
The check runs after the append, so the trigger has to leave room for one
more of the largest writes.

**FINDING: `_summarize` returns `"."` when no whole sentence fits.** It
breaks on `"."`, keeps sentences while they fit the target, and ends with
`". ".join(picked) + "."` -- so an empty `picked` yields a one-character
block. The lesson's own `task` block is **119** characters with no sentence
break, and consolidating it at target **110** rewrites it to **"."**: a
**99.2%** loss that `rewrite` reports as `task v2 rewritten (1/220)`.

**FINDING: the summarizer can grow its input.** When the whole text fits
under the target it is returned with a `"."` appended, so a **72**-character
value summarizes to **73**. A consolidation pass that runs on a block just
under the trigger makes it bigger.

**FINDING: overflow is not a state the code can reach.** `Block.append`
reports `(len/limit)` and assigns regardless: appending **39** characters to
a block with `limit=10` returns `x v1 (39/10)` and stores all **39**. The
limit is a number in a label, so "block overflow" has to be measured by the
caller comparing two integers the block already printed.

Structure: `block_summarize()` is the tool; `sweep()` runs the same append
script at each threshold against the lesson's own `Block`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "08-memory-blocks-sleep-time-compute"
LIMIT, THRESHOLDS = 200, (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
SHORT = "turn {n} noted a small fact."
LONG = ("turn {n} recorded a long observation about the user's project "
        "and its constraints.")
APPENDS = tuple((SHORT if n % 3 else LONG).format(n=n) for n in range(12))
TASK_VALUE = ("plan 30-lesson agent curriculum, target senior eng "
              "audience=senior+staff eng, cite arXiv and first-party framework docs")
HUMAN_VALUE = ("name=ava role=ships_agents city=Berlin "
               "city=Lisbon (updated from Berlin)")


def block_summarize(ref, block):
    """The tool the exercise asks for: the sleep agent's three lines, named."""
    return block.rewrite(ref._summarize(block.value, block.limit // 2))


def sweep(ref, threshold):
    block = ref.Block(label="human", limit=LIMIT)
    calls, overflows = 0, 0
    for text in APPENDS:
        block.append(text)
        if len(block.value) > block.limit:
            overflows += 1
        if block.near_limit(threshold):
            block_summarize(ref, block)
            calls += 1
    return {"calls": calls, "overflows": overflows, "final": len(block.value)}


def destroyed(ref):
    block = ref.Block(label="task", limit=220)
    block.append(TASK_VALUE)
    before = len(block.value)
    note = block_summarize(ref, block)
    return {"before": before, "after": len(block.value), "value": block.value,
            "note": note, "loss": round(1 - len(block.value) / before, 3)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {threshold: sweep(ref, threshold) for threshold in THRESHOLDS}
    safe = [t for t in THRESHOLDS if rows[t]["overflows"] == 0]
    longest = max(len(text) for text in APPENDS)
    tiny = ref.Block(label="x", limit=10)
    tiny_note = tiny.append("this is much longer than ten characters")
    return {
        "rows": rows, "safe": safe, "frontier": max(safe),
        "longest": longest, "closed_form": round(1 - longest / LIMIT, 2),
        "calls_at_frontier": rows[max(safe)]["calls"],
        "calls_at_min": rows[min(THRESHOLDS)]["calls"],
        "default_overflows": rows[0.8]["overflows"],
        "destroyed": destroyed(ref),
        "grew": (len(HUMAN_VALUE), len(ref._summarize(HUMAN_VALUE, 90))),
        "tiny_note": tiny_note, "tiny_stored": len(tiny.value), "tiny_limit": tiny.limit,
    }


def verify(result):
    rows, gone = result["rows"], result["destroyed"]
    return [
        practice.Check(
            "ANSWER: 0.60 is the cheapest threshold that never overflows",
            all([result["frontier"] == 0.6, result["default_overflows"] == 3,
                 result["calls_at_frontier"] == 7, result["calls_at_min"] == 11,
                 rows[0.95]["calls"] == 3,
                 result["closed_form"] == result["frontier"]]),
            f"over the same 12 appends against a {LIMIT}-character block whose longest "
            f"append is {result['longest']} characters, thresholds {result['safe']} "
            f"overflow 0 times and 0.80 overflows {result['default_overflows']}. Calls "
            f"fall from {result['calls_at_min']} at 0.50 to "
            f"{result['calls_at_frontier']} at {result['frontier']} to "
            f"{rows[0.95]['calls']} at 0.95, and 1 - longest/limit is "
            f"{result['closed_form']} -- the measured frontier to the step",
        ),
        practice.Check(
            "FINDING: _summarize returns '.' when no whole sentence fits",
            all([gone["value"] == ".", gone["after"] == 1, gone["before"] == 119,
                 gone["loss"] == 0.992, gone["note"] == "task v2 rewritten (1/220)"]),
            f"the lesson's own task value is {gone['before']} characters with no sentence "
            f"break, so consolidating it at target 110 leaves {gone['value']!r} -- a "
            f"{gone['loss']:.1%} loss that rewrite reports as {gone['note']!r}. An empty "
            "'picked' list still gets the trailing full stop",
        ),
        practice.Check(
            "FINDING: the summarizer can grow its input",
            all([result["grew"] == (72, 73), result["grew"][1] > result["grew"][0]]),
            f"when the whole text fits under the target it is returned with a '.' "
            f"appended, so a {result['grew'][0]}-character value summarizes to "
            f"{result['grew'][1]}. A consolidation pass on a block just under the trigger "
            "makes it bigger",
        ),
        practice.Check(
            "FINDING: overflow is not a state the code can reach",
            all([result["tiny_note"] == "x v1 (39/10)", result["tiny_stored"] == 39,
                 result["tiny_limit"] == 10]),
            f"appending 39 characters to a block with limit {result['tiny_limit']} "
            f"returns {result['tiny_note']!r} and stores {result['tiny_stored']}. The "
            "limit is a number in a label, so overflow has to be measured by the caller "
            "comparing two integers the block already printed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
