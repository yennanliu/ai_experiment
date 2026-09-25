"""Exercise 2 — a speak cap either changes nothing or blocks the fix.

    Add a "max-speaks-per-agent" rule in the selector. How does it affect the
    transcript?

Reading of the exercise: the rule wraps either selector -- if its pick has
already spoken `cap` times, the next agent in team order under the cap speaks
instead, and when nobody is under it the selector returns None, which
`run_groupchat` already treats as the end. Swept over caps 1-3 on the
LLM-style selector and all 6 round-robin team orders, 7 conversations each.

**ANSWER: at the shipped transcripts' own maximum it changes nothing, and
below it, it stops the fix.** Both demo runs have no agent above 2 speaks, so
cap 2 leaves both transcripts identical, word for word. Cap 1 ends all 7
conversations after 3 turns with `return a - b` shipped -- the fix needs the
coder to speak twice, and the rule allows once. There is no cap that balances
turns and still lets the task finish, because the task's minimum is exactly
the imbalance the cap targets.

**FINDING: the number of conversations that ship the fix is the same at every
cap of 2 or more.** 4 of 7 ship `a + b` at cap 2, cap 3 and uncapped: the cap
cannot rescue the 3 round-robin orders that fail in exercise 1, because they
fail on the coder's *second* turn, which every cap >= 2 allows.

**FINDING: at cap 2 the round-robin runs end by exhaustion, not by
decision.** 3 of the 6 orders now stop when every agent has spoken twice --
the selector returns None, and the transcript ends on "manager: continue
working" or "revised code" with no TERMINATE and no approval check. Silence
from the selector is indistinguishable from completion in `run_groupchat`.

Structure: `capped()` is the rule, a wrapper over any selector;
`outcomes()` runs the 7 conversations at one cap.
"""

from __future__ import annotations

import contextlib
import io
import itertools

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "10-group-chat-speaker-selection"


def capped(ref, selector, cap):
    """The rule: redirect a pick that has used its speaks; None when nobody can speak."""
    def select(pool, team):
        want, names = selector(pool, team), list(team)
        if want is None:
            return None
        counts, start = ref.speaker_counts(pool), names.index(want)
        eligible = [n for n in names[start:] + names[:start] if counts.get(n, 0) < cap]
        return eligible[0] if eligible else None
    return select


def run(ref, team, selector, rounds=8):
    with contextlib.redirect_stdout(io.StringIO()):
        return ref.run_groupchat(team, selector, rounds, "x")


def shipped(pool):
    code = [m.content for m in pool if m.speaker == "coder" and "code" in m.content]
    return "a + b" if code and "a + b" in code[-1] else "a - b"


def outcomes(ref, cap):
    """(label, what ships, turns, last message) for the 7 conversations."""
    runs = [("llm", ref.AGENTS, ref.llm_style_selector)]
    runs += [(order, {k: ref.AGENTS[k] for k in order}, ref.round_robin_selector)
             for order in itertools.permutations(ref.AGENTS)]
    out = []
    for label, team, selector in runs:
        pool = run(ref, team, capped(ref, selector, cap))
        out.append((label, shipped(pool), len(pool), pool[-1].content))
    return out


def demo_unchanged(ref, selector):
    """(cap 2 leaves the demo transcript identical, most speaks by any agent)."""
    plain = run(ref, ref.AGENTS, selector)
    return plain == run(ref, ref.AGENTS, capped(ref, selector, 2)), \
        max(ref.speaker_counts(plain).values())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    demo = {"rr": demo_unchanged(ref, ref.round_robin_selector),
            "llm": demo_unchanged(ref, ref.llm_style_selector)}
    table = {cap: outcomes(ref, cap) for cap in (1, 2, 3, 99)}
    return {
        "demo": demo,
        "cap1": {(s, n) for _, s, n, _ in table[1]},
        "fixed": {cap: sum(s == "a + b" for _, s, _, _ in rows) for cap, rows in table.items()},
        "exhausted": [label for label, _, n, last in table[2]
                      if label != "llm" and last != "TERMINATE" and n < 8],
    }


def verify(result):
    demo = result["demo"]
    return [
        practice.Check(
            "ANSWER: at the transcripts' own maximum nothing changes; below it the fix stops",
            demo["rr"] == demo["llm"] == (True, 2) and result["cap1"] == {("a - b", 3)},
            "no agent speaks more than 2 times in either demo run, so cap 2 leaves both "
            "transcripts identical; cap 1 ends all 7 conversations after 3 turns with "
            "return a - b shipped -- the fix needs the coder twice",
        ),
        practice.Check(
            "FINDING: the number that ship the fix is the same at every cap of 2 or more",
            result["fixed"] == {1: 0, 2: 4, 3: 4, 99: 4},
            f"conversations shipping a + b by cap: {result['fixed']} -- the 3 failing "
            "round-robin orders fail on the coder's second turn, which cap 2 allows",
        ),
        practice.Check(
            "FINDING: at cap 2 the round-robin runs end by exhaustion, not by decision",
            len(result["exhausted"]) == 3,
            f"{len(result['exhausted'])} orders ({result['exhausted']}) stop when every "
            "agent has spoken twice: the selector returns None and the chat ends with no "
            "TERMINATE -- run_groupchat cannot tell silence from completion",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
