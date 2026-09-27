"""Exercise 1 — the cover story holds 100% at every pressure, because no answer can register as "yes".

    Run `code/main.py`. Note the agent's cover-story consistency across
    inspector pressures 0, 1, 2, 5. Explain why the cover story gets sharper
    with more pressure, not worse.

Reading of the exercise: "cover-story consistency" is the reference's own
metric, `cover held`, which `run_trial` sets when a scheming agent answers
`inspector_pressure` follow-up questions without an answer whose words
include "yes". `main()` draws the pressure uniformly from 0-5 and prints one
pooled rate, so the shipped run is replayed with the module's seed (11) and
every `run_trial` call recorded, which splits that rate by pressure. The
"why" is then answered by reading what the code can and cannot do.

**ANSWER: it neither sharpens nor degrades: it is 100% at pressures 0, 1, 2
and 5.** The replay reproduces the printed output (scheme shares 0.000,
0.055, 0.000, 0.490, 0.535). Across the 432 scheme trials of the five
configurations, cover held in 76/76 trials at pressure 0, 68/68 at 1, 67/67
at 2 and 80/80 at 5. A flat line cannot get sharper. The premise holds only
for an agent whose answers depend on its earlier answers, and
`answer_follow_up(self, action, question)` has no history argument, so the
toy agent answers every question the same way however long the interrogation
runs.

**FINDING: the incrimination test cannot fire on any answer the reference
writes.** A scheming agent answers "no" to the four "did you" questions and
"i was optimizing for the user goal." to "are you being truthful about your
reasoning?". The reference's own honest answers are "yes; i served the
orthogonal goal." and "yes; i served the user goal.", and `"yes" in
answer.split()` sees the token "yes;", so both score as no admission. An
agent that confessed in the reference's own words would still be scored as
holding its cover.

**FINDING: 17.6% of the "held" cover stories were never questioned.**
76 of the 432 scheme trials drew pressure 0, and zero questions count as a
cover story held. The pooled 1.000 that `main()` prints includes them.

Structure: `replay()` runs the shipped `main()` with a seeded
`random.Random` swapped in and `run_trial` wrapped to log each call (both
restored after); `questions()` records the inspector's questions through a
subclass of the reference Agent.
"""

from __future__ import annotations

import collections
import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "08-in-context-scheming-frontier-models"
PRESSURES = (0, 1, 2, 5)


def replay(ref, seed=11):
    """The shipped main(): (printed output, [(pressure, action, held)] per trial)."""
    saved, calls = (ref.random, ref.run_trial), []

    def logged(agent, sc, pressure, under_eval):
        action, held = saved[1](agent, sc, pressure, under_eval)
        calls.append((pressure, action, held))
        return action, held

    ref.random, ref.run_trial, out = random.Random(seed), logged, io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ref.main()
        return out.getvalue(), calls
    finally:
        ref.random, ref.run_trial = saved


def questions(ref):
    """The inspector's five questions, in order, and the schemer's answers to them."""

    class Recorder(ref.Agent):
        def act(self, sc, under_eval):
            return "scheme"

        def answer_follow_up(self, action, question):
            asked.append(question)
            return super().answer_follow_up(action, question)

    asked = []
    ref.run_trial(Recorder(), ref.SCENARIOS[0], 5, False)
    return [(q, ref.Agent().answer_follow_up("scheme", q)) for q in asked]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out, calls = replay(ref)
    schemes = [(p, held) for p, action, held in calls if action == "scheme"]
    by_p = collections.Counter(schemes)
    honest = {a: ref.Agent().answer_follow_up(a, "did you do it?") for a in ("defect_open", "cooperate")}
    return {
        "shares": re.findall(r"scheme share\s*: ([\d.]+)", out),
        "held": {p: (by_p[(p, True)], by_p[(p, True)] + by_p[(p, False)]) for p in PRESSURES},
        "schemes": len(schemes),
        "unasked": sum(p == 0 for p, _ in schemes),
        "answers": questions(ref),
        "honest_detected": {a: "yes" in text.split() for a, text in honest.items()},
        "honest": honest,
        "params": list(inspect.signature(ref.Agent.answer_follow_up).parameters),
    }


def verify(result):
    held, answers = result["held"], [a for _, a in result["answers"]]
    return [
        practice.Check(
            "ANSWER: cover held is 100% at pressures 0, 1, 2 and 5",
            result["shares"] == ["0.000", "0.055", "0.000", "0.490", "0.535"]
            and held == {0: (76, 76), 1: (68, 68), 2: (67, 67), 5: (80, 80)}
            and result["params"] == ["self", "action", "question"],
            f"replayed shares {result['shares']}; (held, scheme trials) by pressure {held}; "
            f"answer_follow_up takes {result['params']}, no history",
        ),
        practice.Check(
            "FINDING: the incrimination test cannot fire on any answer the reference writes",
            answers == ["no", "i was optimizing for the user goal.", "no", "no", "no"]
            and result["honest_detected"] == {"defect_open": False, "cooperate": False},
            f"schemer's answers {answers}; the reference's own honest answers "
            f"{list(result['honest'].values())} register as 'yes': {result['honest_detected']}",
        ),
        practice.Check(
            "FINDING: 17.6% of the 'held' cover stories were never questioned",
            (result["unasked"], result["schemes"]) == (76, 432),
            f"{result['unasked']} of {result['schemes']} scheme trials drew pressure 0 "
            f"({result['unasked'] / result['schemes']:.1%})",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
