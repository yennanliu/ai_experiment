"""Exercise 3 — the budget truncates the history 14 turns before the manager compresses it.

    Build a "context replay" tool. Given a conversation transcript, replay it
    through the ContextEngine and visualize how the budget allocation changes
    turn by turn. Plot token usage per component over time. Identify the turn
    where context starts getting compressed.

Reading of the exercise: the transcript is a synthetic one -- one user query per
turn through `ContextEngine.chat`, which appends its own simulated assistant
reply -- because the lesson ships no transcript. The replay records
`budget.allocations` and `conversation.stats()` after every turn, which is what
"visualize how the allocation changes" needs as input.

**ANSWER: compression starts at turn 147.** `ConversationManager._compress_if_needed`
fires for the first time there, summarising the two oldest turns and leaving 292
live turns and 1 summary behind.

**FINDING: the context was already being cut at turn 133.** `assemble` calls
`allocate("conversation_history", history_text, max_tokens=5000)`, and
`allocate` truncates silently when the content exceeds its cap. From turn 133
the history component books 4,999 tokens and the words past that are dropped
without a summary, a log line or a change in `stats()`. The exercise's question
has two answers 14 turns apart, and only the later one is observable.

**MECHANISM: the manager and the budget count different strings.**
`_compress_if_needed` sums `count_tokens(turn["content"])` over the turns;
`get_context()` emits section headers and a `role: ` prefix per turn. Just after
the pass the gate reads 4,980 -- under its own 5,000 trigger, so it stops --
while the string it hands the budget is 5,561, 112% of that. The manager
believes it is finished and the budget truncates anyway.

**FINDING: one compression pass barely moves the number.** It drops the two
oldest turns and appends a summary that is a 100-character prefix of each, so
the history is still above the budget's cap immediately afterwards: the
allocation stays pinned at 4,999 and every later turn is truncated too.

**CONTROL: gate on the string it actually emits and compression starts at turn
133**, the same turn the truncation starts, and the two mechanisms stop
disagreeing.

Structure: `replay` walks the transcript recording both views, and `gated`
re-runs the manager with `token_count()` as the trigger.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "05-context-engineering"
TURNS = 200
CAP = 5000


def transcript(n=TURNS):
    return [f"Turn {i}: please look at the failing test in module {i} and fix it."
            for i in range(1, n + 1)]


def replay(ref, engine, queries):
    """One row per turn: the allocations, and both views of the history size."""
    rows = []
    for query in queries:
        budget = engine.chat(query)
        stats = engine.conversation.stats()
        rows.append({"history": budget.allocations.get("conversation_history", 0),
                     "summaries": stats["summaries"], "live": stats["live_turns"],
                     "emitted": stats["tokens"],
                     "gate": sum(ref.count_tokens(t["content"])
                                 for t in engine.conversation.turns)})
    return rows


def first_where(rows, predicate):
    return next((i + 1 for i, row in enumerate(rows) if predicate(row)), None)


def gated(ref, queries):
    """The control: compress when the emitted string exceeds the cap, not the turn sum."""
    manager = ref.ConversationManager(max_history_tokens=CAP)
    for turn, query in enumerate(queries, 1):
        manager.turns.append({"role": "user", "content": query})
        reply = f"[Simulated response to: {query[:50]}...]"     # ContextEngine.chat's own
        manager.turns.append({"role": "assistant", "content": reply})
        if manager.token_count() > CAP:
            return turn
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    queries = transcript()
    rows = replay(ref, ref.ContextEngine(), queries)
    compressed = first_where(rows, lambda r: r["summaries"] > 0)
    truncated = first_where(rows, lambda r: r["history"] >= CAP - 1)
    at = rows[compressed - 1]
    return {
        "compressed": compressed, "truncated": truncated,
        "gap": compressed - truncated,
        "live": at["live"], "summaries": at["summaries"],
        "gate": at["gate"], "emitted": at["emitted"], "cap": CAP,
        "overhead": round(at["emitted"] / at["gate"], 3),
        "after": [rows[compressed - 1 + k]["history"] for k in (0, 1, 2, 3)],
        "pinned": all(r["history"] >= CAP - 1 for r in rows[compressed - 1:]),
        "control": gated(ref, queries),
        "turns": len(queries),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: compression starts at turn 147",
            all([result["compressed"] == 147, result["summaries"] == 1]),
            f"`_compress_if_needed` fires for the first time at turn "
            f"{result['compressed']} of {result['turns']}, summarising the two oldest turns "
            f"and leaving {result['live']} live turns and {result['summaries']} summary "
            "behind. That is the turn `stats()` makes visible",
        ),
        practice.Check(
            "FINDING: the context was already being cut at turn 133",
            all([result["truncated"] == 133, result["gap"] == 14]),
            f"`assemble` passes max_tokens={result['cap']} for the history, and `allocate` "
            f"truncates silently above its cap. From turn {result['truncated']} the "
            f"component books {result['cap'] - 1} tokens and the rest is dropped with no "
            f"summary and no change in stats() -- {result['gap']} turns before the manager "
            "notices. The exercise's question has two answers",
        ),
        practice.Check(
            "MECHANISM: the manager and the budget count different strings",
            all([result["emitted"] > result["cap"], result["emitted"] > result["gate"],
                 result["overhead"] > 1.1, result["gate"] < result["cap"]]),
            f"just after the compression pass the gate -- sum(count_tokens(turn['content'])) "
            f"-- reads {result['gate']}, back under the {result['cap']} it triggers on, "
            f"while the string get_context() produces is {result['emitted']} tokens: "
            f"{result['overhead']:.0%} of it, from the section headers and the 'role: ' "
            "prefix per turn. The manager thinks it is done; the budget still truncates",
        ),
        practice.Check(
            "FINDING: one compression pass barely moves the number",
            all([result["after"] == [result["cap"] - 1] * 4, result["pinned"]]),
            f"the pass drops the two oldest turns and appends a 100-character prefix of "
            f"each, so the history allocation over the compression turn and the next three "
            f"is {result['after']} -- pinned at the cap. Every turn from "
            f"{result['truncated']} onward is truncated, compressed or not",
        ),
        practice.Check(
            "CONTROL: gate on the emitted string and the two mechanisms agree",
            all([result["control"] == result["truncated"],
                 result["control"] < result["compressed"]]),
            f"triggering compression on token_count() rather than on the turn sum moves the "
            f"first compression to turn {result['control']} -- the same turn the silent "
            f"truncation starts. One counter, one answer to the exercise's question",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
