"""Exercise 1 — round-robin ships the bug because the manager speaks between review and fix.

    Run `code/main.py`. Compare the conversation under round-robin vs
    LLM-selected. Which agent dominates under each?

Reading of the exercise: "dominates" is read two ways -- who speaks most, and
whose turn decides the outcome -- and the comparison is taken over what each
conversation *ships*, since two transcripts that end in TERMINATE can end with
different code.

**ANSWER: the coder dominates round-robin, 2 of 4 turns, and nobody
dominates the LLM-style selector, 2/2/2.** The demo's own gloss, "Round-robin
gives every agent an equal turn", is contradicted by its counts {coder: 2,
reviewer: 1, manager: 1}: the chat ends on the coder's second turn.

**FINDING: round-robin ends with the bug unfixed.** The reviewer says "bug
detected -- please fix", the manager says "continue working", and the coder
answers `TERMINATE`. `coder_policy` reads only the *last* non-coder message in
its window, and that is the manager's, which contains neither "review" nor
"fix" -- so the fix request, still two messages back and inside the window,
is ignored. The final code is `return a - b`. The LLM-style run is the only
one of the two that ships `a + b`.

**FINDING: whether round-robin works is decided by team order alone.** Over
the 6 orders of the same three agents, 3 ship the bug and 3 ship the fix, and
the 3 that fail are exactly those where the manager follows the reviewer in
the cycle. The shipped `AGENTS` dict happens to be one of them.

**FINDING: the demo's round cap cuts off 2 of the 3 working orders.** Those
need 8, 9 and 10 turns to reach `TERMINATE`; with `max_rounds=8` only one
finishes. A successful round-robin run also spends a turn on the coder
re-sending "revised code" after approval, because "review: approved"
contains "review".

Structure: `run()` calls the reference `run_groupchat` with its printing
silenced; `shipped()` reads the last code the coder produced.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import itertools

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "10-group-chat-speaker-selection"


def run(ref, team, selector, rounds=8):
    with contextlib.redirect_stdout(io.StringIO()):
        return ref.run_groupchat(team, selector, rounds, "x")


def shipped(pool):
    code = [m.content for m in pool if m.speaker == "coder" and "code" in m.content]
    return "a + b" if code and "a + b" in code[-1] else "a - b"


def orders(ref, rounds):
    """Per team order: (what ships, turns, ended on TERMINATE, manager follows reviewer)."""
    out = {}
    for order in itertools.permutations(ref.AGENTS):
        pool = run(ref, {k: ref.AGENTS[k] for k in order}, ref.round_robin_selector, rounds)
        follows = order[(order.index("reviewer") + 1) % 3] == "manager"
        out[order] = (shipped(pool), len(pool), pool[-1].content == "TERMINATE", follows)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rr = run(ref, ref.AGENTS, ref.round_robin_selector)
    llm = run(ref, ref.AGENTS, ref.llm_style_selector)
    wide = orders(ref, 20)
    return {
        "claim": "equal turn" in inspect.getsource(ref.main),
        "rr": ref.speaker_counts(rr), "llm": ref.speaker_counts(llm),
        "rr_tail": [(m.speaker, m.content) for m in rr[-3:]],
        "rr_ships": shipped(rr), "llm_ships": shipped(llm),
        "fail_orders": sorted(o for o, v in wide.items() if v[0] == "a - b"),
        "follow_orders": sorted(o for o, v in wide.items() if v[3]),
        "working_turns": sorted(v[1] for v in wide.values() if v[0] == "a + b"),
        "finish_at_8": sum(v[0] == "a + b" and v[2] for v in orders(ref, 8).values()),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the coder dominates round-robin; nobody dominates LLM-style",
            result["claim"] and result["rr"] == {"coder": 2, "reviewer": 1, "manager": 1}
            and set(result["llm"].values()) == {2},
            f"round-robin counts {result['rr']} -- not the 'equal turn' the demo claims; "
            f"LLM-style counts {result['llm']}",
        ),
        practice.Check(
            "FINDING: round-robin ends with the bug unfixed",
            result["rr_ships"] == "a - b" and result["llm_ships"] == "a + b"
            and result["rr_tail"][-1] == ("coder", "TERMINATE"),
            f"the last three turns are {result['rr_tail']}: coder_policy reads only the "
            "last non-coder message, the manager's, so the fix request two back is "
            f"ignored -- round-robin ships return {result['rr_ships']}, LLM-style "
            f"return {result['llm_ships']}",
        ),
        practice.Check(
            "FINDING: whether round-robin works is decided by team order alone",
            len(result["fail_orders"]) == 3 and result["fail_orders"] == result["follow_orders"],
            f"of 6 team orders, the 3 that ship the bug are {result['fail_orders']} -- "
            "exactly those where the manager follows the reviewer in the cycle",
        ),
        practice.Check(
            "FINDING: the demo's round cap cuts off 2 of the 3 working orders",
            result["working_turns"] == [8, 9, 10] and result["finish_at_8"] == 1,
            f"the working orders need {result['working_turns']} turns to TERMINATE, so "
            f"under max_rounds=8 {result['finish_at_8']} finishes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
