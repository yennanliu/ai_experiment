"""Exercise 3 — the goal check already exists, one turn late, inside the manager.

    Implement a goal-reached termination: stop when the reviewer returns
    "approved." How often does it trigger before the round cap?

Reading of the exercise: "how often" needs a population, so the check runs on
the 7 conversations the module can produce -- the LLM-style selector and all 6
round-robin team orders -- under the demo's `max_rounds=8`. The check is a
selector wrapper returning None after an approval, which `run_groupchat`
already treats as the end.

**ANSWER: it triggers in 4 of 7 conversations, and strictly before the cap in
3.** The LLM-style run stops at turn 5 instead of 6. Among round-robin orders
it stops at turns 6, 7 and 8 -- the last *on* the cap. In the other 3 orders,
including the demo's own round-robin run, it never fires: the coder says
`TERMINATE` over unfixed code first (exercise 1), and no approval exists.

**FINDING: the manager already is this check, one turn later.**
`manager_policy` returns TERMINATE exactly when a reviewer message containing
"approved" is in the pool -- which is the exercise's condition. The LLM-style
selector always hands the manager the turn after an approval, so the new
check saves exactly 1 turn there: the manager's.

**FINDING: under round-robin the goal check rescues 2 conversations the cap
was cutting off.** The orders that need 9 and 10 turns to reach TERMINATE
now stop at 7 and 8, within the cap, with the approved code. And every
round-robin success loses the wasted turn where the coder re-sends
"revised code" after approval, because "review: approved" contains "review".

**FINDING: the goal check cannot help the conversations that fail.** Its 3
misses are exactly the 3 conversations that ship `a - b`: a termination rule
reads the outcome and cannot create one.

Structure: `goal()` is the termination rule, a wrapper over any selector.
"""

from __future__ import annotations

import contextlib
import io
import itertools

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "10-group-chat-speaker-selection"
CAP = 8


def goal(selector):
    """Stop once the last message is a reviewer approval."""
    def select(pool, team):
        if pool and pool[-1].speaker == "reviewer" and "approved" in pool[-1].content:
            return None
        return selector(pool, team)
    return select


def run(ref, team, selector, rounds=CAP):
    with contextlib.redirect_stdout(io.StringIO()):
        return ref.run_groupchat(team, selector, rounds, "x")


def approved(pool):
    return any(m.speaker == "reviewer" and "approved" in m.content for m in pool)


def conversations(ref):
    yield "llm", ref.AGENTS, ref.llm_style_selector
    for order in itertools.permutations(ref.AGENTS):
        yield order, {k: ref.AGENTS[k] for k in order}, ref.round_robin_selector


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {}
    for label, team, selector in conversations(ref):
        plain, checked = run(ref, team, selector, 20), run(ref, team, goal(selector))
        rows[label] = {"plain": len(plain), "checked": len(checked),
                       "fired": approved(checked), "ships_fix": approved(plain)}
    fired = {k: v for k, v in rows.items() if v["fired"]}
    return {
        "rows": rows, "fired": len(fired),
        "before_cap": sum(v["checked"] < CAP for v in fired.values()),
        "rr_turns": sorted(v["checked"] for k, v in fired.items() if k != "llm"),
        "demo_rr": rows[tuple(ref.AGENTS)]["fired"],
        "rescued": sorted(v["checked"] for v in fired.values() if v["plain"] > CAP),
        "misses_are_failures": all(not v["ships_fix"] for v in rows.values() if not v["fired"]),
        "manager_checks": ref.manager_policy([ref.Msg("reviewer", "review: approved")]),
    }


def verify(result):
    llm = result["rows"]["llm"]
    return [
        practice.Check(
            "ANSWER: it triggers in 4 of 7 conversations, strictly before the cap in 3",
            result["fired"] == 4 and result["before_cap"] == 3
            and result["rr_turns"] == [6, 7, 8] and not result["demo_rr"],
            f"LLM-style stops at turn {llm['checked']}, round-robin orders at "
            f"{result['rr_turns']} (the last on the cap of {CAP}); the demo's own "
            "round-robin run never fires -- the coder terminates over unfixed code first",
        ),
        practice.Check(
            "FINDING: the manager already is this check, one turn later",
            result["manager_checks"] == "TERMINATE" and llm["plain"] - llm["checked"] == 1,
            "manager_policy returns TERMINATE exactly when an approval is in the pool, and "
            f"the LLM-style selector gives it the next turn, so the check saves "
            f"{llm['plain'] - llm['checked']} turn: the manager's",
        ),
        practice.Check(
            "FINDING: under round-robin it rescues 2 conversations the cap was cutting off",
            result["rescued"] == [7, 8],
            f"the orders needing 9 and 10 turns to TERMINATE now stop at {result['rescued']}, "
            "within the cap, with the approved code",
        ),
        practice.Check(
            "FINDING: the goal check cannot help the conversations that fail",
            result["misses_are_failures"],
            "its 3 misses are exactly the 3 conversations that ship a - b -- a "
            "termination rule reads the outcome and cannot create one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
