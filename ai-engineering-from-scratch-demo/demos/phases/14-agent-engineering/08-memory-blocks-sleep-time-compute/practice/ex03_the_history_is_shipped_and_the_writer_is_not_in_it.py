"""Exercise 3 — the history is shipped, and the writer is not in it.

    Version blocks. On every write record the old value and a diff. Expose
    `block_history(label)` so operators can debug "why did the agent forget
    X."

Reading of the exercise: `Block` already increments `version` and appends the
previous value to `history` on all three write paths, so "record the old
value" is done. What is missing is the diff, the reader, and -- the part that
decides whether the debugging question can be answered at all -- any record
of *who* wrote. The exercise is therefore implemented as `block_history()`
over the shipped fields, and then measured for what it cannot say.

**ANSWER: `block_history(label)` over the shipped `history` and `version`.**
Three writes to a `task` block give **3** entries with character deltas
**+52**, **+69** and **-120**. Exactly **1** of the three is negative, and it
is the answer to "why did the agent forget X" -- the consolidation rewrite
that left the block holding `"."`.

**FINDING: nothing in the record says who wrote.** `append`, `replace` and
`rewrite` take **0** actor parameters between them, and both agents hold the
same `BlockStore`. So the destructive entry is distinguishable from the two
benign ones only by its sign; had the primary agent truncated the block, the
history would look identical.

**FINDING: the rewrite is the only entry that can lose data, and it is the
cheapest to issue.** `append` can only grow the value and `replace` fails
loudly when `old` is absent, so of the **3** write paths exactly **1** can
shrink a block -- and it is the one the sleep pass calls automatically, with
no argument saying how much loss is acceptable.

**FINDING: the history is full copies and unbounded.** After **3** writes the
block holds **1** character of value and **173** characters of history, and
nothing trims it. A block that is rewritten every sleep pass accumulates a
copy of its pre-consolidation value forever, which is the same storage the
consolidation was run to reclaim.

Structure: `block_history()` is the reader; `unified()` produces the diff
from the values the lesson already stores.
"""

from __future__ import annotations

import difflib
import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "08-memory-blocks-sleep-time-compute"
LIMIT = 120
FIRST = "plan 30-lesson agent curriculum for senior engineers"
SECOND = "audience=senior+staff eng, cite arXiv and first-party framework docs"


def unified(before, after):
    return [line for line in difflib.unified_diff(
        before.split(), after.split(), lineterm="", n=0) if line.startswith(("+", "-"))]


def block_history(store, label):
    """Every version of a block, with the diff that produced it."""
    block = store.get(label)
    rows, values = [], [*block.history, block.value]
    for index in range(1, len(values)):
        before, after = values[index - 1], values[index]
        rows.append({"version": index, "before": before, "after": after,
                     "delta": len(after) - len(before), "diff": unified(before, after)})
    return rows


def build(ref):
    store = ref.BlockStore()
    store.create("task", "the current task scope", limit=LIMIT)
    block = store.get("task")
    block.append(FIRST)
    block.append(SECOND)
    if block.near_limit():
        block.rewrite(ref._summarize(block.value, block.limit // 2))
    return store


def actor_params(ref):
    return [name for method in ("append", "replace", "rewrite")
            for name in inspect.signature(getattr(ref.Block, method)).parameters
            if name in ("actor", "writer", "agent", "by")]


def shrinkable(ref):
    """Which of the three write paths can make a block smaller."""
    probes = []
    grow = ref.Block(label="p", limit=LIMIT)
    grow.append("aaa")
    probes.append(("append", len(grow.value) > 0))
    swap = ref.Block(label="p", limit=LIMIT)
    swap.append("aaa bbb")
    probes.append(("replace", swap.replace("zzz", "")[:5] == "error"))
    cut = ref.Block(label="p", limit=LIMIT)
    cut.append("aaa bbb")
    before = len(cut.value)
    cut.rewrite(".")
    probes.append(("rewrite", len(cut.value) < before))
    return probes


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    store = build(ref)
    rows = block_history(store, "task")
    block = store.get("task")
    return {
        "entries": len(rows), "deltas": [row["delta"] for row in rows],
        "negative": [row["version"] for row in rows if row["delta"] < 0],
        "value": block.value, "version": block.version,
        "diff_of_last": rows[-1]["diff"][:1],
        "actor_params": actor_params(ref),
        "paths": shrinkable(ref),
        "history_chars": sum(len(text) for text in block.history),
        "value_chars": len(block.value),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three entries, one negative, and it is the forgetting",
            all([result["entries"] == 3, result["deltas"] == [52, 69, -120],
                 result["negative"] == [3], result["value"] == ".",
                 result["version"] == 3]),
            f"block_history returns {result['entries']} entries with deltas "
            f"{result['deltas']}; version {result['negative'][0]} is the only negative "
            f"one and it left the block holding {result['value']!r}. That single row is "
            "the answer to 'why did the agent forget X'",
        ),
        practice.Check(
            "FINDING: nothing in the record says who wrote",
            all([result["actor_params"] == [], len(result["deltas"]) == 3]),
            f"append, replace and rewrite take {len(result['actor_params'])} actor "
            "parameters between them, and both agents hold the same BlockStore. The "
            "destructive entry is distinguishable from the benign ones only by its sign; "
            "a primary-agent truncation would leave an identical history",
        ),
        practice.Check(
            "FINDING: one of the three write paths can lose data",
            all([result["paths"] == [("append", True), ("replace", True),
                                     ("rewrite", True)],
                 result["negative"] == [3]]),
            "append can only grow the value, replace fails loudly when old is absent, "
            "and rewrite replaces it outright -- so 1 of the 3 write paths can shrink a "
            "block, and it is the one the sleep pass calls automatically with no "
            "argument saying how much loss is acceptable",
        ),
        practice.Check(
            "FINDING: the history is full copies and unbounded",
            all([result["history_chars"] == 173, result["value_chars"] == 1,
                 result["history_chars"] > result["value_chars"]]),
            f"after three writes the block holds {result['value_chars']} character of "
            f"value and {result['history_chars']} characters of history, and nothing "
            "trims it. A block rewritten every sleep pass keeps a copy of its "
            "pre-consolidation value forever -- the storage consolidation was run to "
            "reclaim",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
