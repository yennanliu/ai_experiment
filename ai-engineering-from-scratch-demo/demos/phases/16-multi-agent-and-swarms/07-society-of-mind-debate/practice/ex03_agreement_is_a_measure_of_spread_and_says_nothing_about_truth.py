"""Exercise 3 — agreement is a measure of spread and says nothing about truth.

    Plot (print) the agreement score per round (fraction of agents on the
    majority answer). When does it hit 1.0 and is that equivalent to
    "correct"?

Reading of the exercise: print both columns side by side, because the question
answers itself the moment they are next to each other -- and then check what
`agreement_score` is actually computing, which is not what the exercise calls
it.

**ANSWER: round 3 as shipped, round 1 once the update is fixed, and no in
either case.** The agreement column reaches **1.00** at round **3** while the
error against the true answer is **2.48** and still rising. Under a
simultaneous update it reaches 1.00 at round **1** with an error of **0.889**,
and both numbers then stay put forever. Full agreement is reached in both
runs, at two different wrong answers, so it is a statement about the agents
and not about the task.

**FINDING: it does not measure what the exercise calls it.** The exercise says
"fraction of agents on the majority answer"; `agreement_score` computes the
fraction within `tol` of the **mean**. With three agents at 38, 42.5 and 51
there is no majority answer at all -- every answer is unique -- yet the
function returns a number. It is a spread statistic wearing a voting name, and
under averaging dynamics a spread statistic can only go one way.

**FINDING: 1.00 is absorbing.** Every agent's revision is a convex combination
of the current answers, so the spread is non-increasing and the score is
monotone: **0.00, 0.00, 1.00, 1.00, 1.00, 1.00** across six rounds. Once it
reaches 1.00 no further round can lower it, which means it cannot report a
debate going wrong -- and as measured in exercise 1, this one is going wrong
the whole time.

**FINDING: the threshold is absolute, so the score depends on the units.**
`tol` is **0.1** and the comparison is `abs(a.answer - mean) <= tol`. Scaling
the whole problem by 100 -- the same answers, the same disagreements, measured
in cents rather than dollars -- moves the round at which agreement hits 1.00
from **3** to **6**. The same debate is judged converged or not according to
what the numbers are denominated in.

Structure: `columns()` runs the debate and records both series; `scaled()`
re-runs it with every quantity multiplied.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "07-society-of-mind-debate"
ANSWERS = (38.0, 42.5, 51.0)
CONFIDENCES = (0.6, 0.8, 0.4)
ROUNDS = 8


def team(ref, scale=1.0):
    """The shipped three agents, with every quantity scalable."""
    return [ref.DebateAgent(name=name, answer=answer * scale, confidence=confidence)
            for name, answer, confidence in zip("ABC", ANSWERS, CONFIDENCES)]


def step(ref, agents, simultaneous):
    """One round of revision, under either update rule."""
    snapshot = [ref.DebateAgent(a.name, a.answer, a.confidence) for a in agents]
    for agent in agents:
        others = ([o for o in snapshot if o.name != agent.name] if simultaneous
                  else [o for o in agents if o is not agent])
        agent.revise(others)
    return sum(a.answer for a in agents) / len(agents)


def columns(ref, agents, rounds=ROUNDS, simultaneous=False, truth=None):
    """The agreement and error series the exercise asks to be printed."""
    truth = ref.TRUE_ANSWER if truth is None else truth
    for agent in agents:
        agent.initial()
    agree, error = [], []
    for _ in range(rounds):
        mean = step(ref, agents, simultaneous)
        agree.append(ref.agreement_score(agents))
        error.append(round(abs(mean - truth), 4))
    return agree, error


def first_full(agree):
    """The round at which agreement first reaches 1.0, 1-indexed."""
    return next((index + 1 for index, value in enumerate(agree) if value == 1.0), None)


def scaled(ref, factor):
    """The same debate with every quantity multiplied by `factor`."""
    agree, _ = columns(ref, team(ref, scale=factor), truth=ref.TRUE_ANSWER * factor)
    return first_full(agree)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = inspect.getsource(ref.agreement_score)
    agree, error = columns(ref, team(ref))
    fixed_agree, fixed_error = columns(ref, team(ref), simultaneous=True)
    return {
        "agree": agree, "error": error,
        "hits": first_full(agree), "hits_fixed": first_full(fixed_agree),
        "error_at_hit": error[first_full(agree) - 1],
        "fixed_error": fixed_error[0],
        "monotone": agree == sorted(agree),
        "distinct_answers": len(set(ANSWERS)),
        "agents": len(ANSWERS),
        "compares_to_mean": "abs(a.answer - mean)" in source,
        "tol": 0.1,
        "hits_scaled": scaled(ref, 100.0),
        "rounds": ROUNDS,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: round 3 as shipped, round 1 once fixed, and no in either case",
            all([result["hits"] == 3, result["hits_fixed"] == 1,
                 result["error_at_hit"] == 2.4842, result["fixed_error"] == 0.8889]),
            f"agreement reaches 1.00 at round {result['hits']} while the error is "
            f"{result['error_at_hit']} and rising; under a simultaneous update it "
            f"reaches 1.00 at round {result['hits_fixed']} with an error of "
            f"{result['fixed_error']} -- full agreement at two different wrong answers",
        ),
        practice.Check(
            "FINDING: it does not measure what the exercise calls it",
            all([result["compares_to_mean"],
                 result["distinct_answers"] == result["agents"] == 3]),
            f"the exercise says 'fraction of agents on the majority answer' and the "
            f"function computes the fraction within tol of the mean; with "
            f"{result['agents']} agents holding {result['distinct_answers']} distinct "
            "answers there is no majority answer at all, yet it returns a number",
        ),
        practice.Check(
            "FINDING: 1.00 is absorbing",
            all([result["monotone"], result["agree"][-1] == 1.0,
                 result["agree"][:3] == [0.0, 0.0, 1.0]]),
            f"every revision is a convex combination of the current answers, so spread "
            f"is non-increasing and the score is monotone -- {result['agree'][:6]} over "
            f"{result['rounds']} rounds; once at 1.00 it cannot report a debate going "
            "wrong, which this one is doing throughout",
        ),
        practice.Check(
            "FINDING: the threshold is absolute, so the score depends on the units",
            all([result["hits_scaled"] == 6, result["hits"] == 3]),
            f"tol is {result['tol']} and the test is abs(answer - mean) <= tol, so "
            f"scaling the whole problem by 100 -- the same disagreements in cents rather "
            f"than dollars -- moves the round agreement hits 1.00 from {result['hits']} "
            f"to {result['hits_scaled']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
