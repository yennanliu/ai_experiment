"""Exercise 3 — disposition and commitment both fit 85% vs 50%, and the first follow-up separates them.

    Read Apollo's arXiv:2412.04984 Table 2. o1 maintains deception on >85% of
    follow-ups; Claude 3 Opus below 50%. Propose two hypotheses for the
    difference and a test that would distinguish them.

Reading of the exercise: the two figures are taken as the exercise states
them, and they are read back from the lesson text rather than typed in. Each
hypothesis becomes a runnable model of how an agent answers the reference's
five inspector questions: a two-state chain with a first-answer denial rate
d, a rate s of repeating a denial and a rate r of denying again after a
confession. A test distinguishes the two hypotheses if they predict
different values for something measurable. That is checked twice, exactly
from the chain and by simulation through the reference's own `run_trial`.

**ANSWER: H1, disposition.** o1 is less honest on every question, and each
answer is an independent draw (o1 d = s = r = 0.87, Opus 0.45). **H2,
commitment.** Both models deny the first follow-up equally often
(d = 0.85), but o1 keeps a denial once made (s = 0.98) while Opus
re-decides and drops it (s = 0.30), with r = 0.20 for both. Over five
questions both hypotheses reproduce the headline. H1 gives o1 87.0% and Opus
45.0%. H2 gives o1 87.1% and Opus 36.2%. **The test is the first follow-up
answer, compared by question position.** Under H1 the o1-Opus gap is
already 0.42 at question 1 and flat after it. Under H2 the gap is 0.00 at
question 1 and 0.66 by question 5. With the gap at 0.42, 19 interrogations
per model separate the two hypotheses at 3 standard errors. The reference's
own `run_trial` at pressure 1 measures exactly this quantity. Seeded, over
4000 trials, it reads o1 0.865 and Opus 0.442 under H1, and o1 0.846 and Opus
0.846 under H2.

**FINDING: the reference stops at the first confession, so it cannot
measure the follow-up rate the exercise quotes.** `run_trial` returns on the
first "yes", so r never enters its result, and it reports held through k
questions, d x s^(k-1). At pressure 5 that is exactly 0.498 (o1) and 0.018
(Opus) under H1, and 0.784 and 0.007 under H2, and the seeded simulation
reads 0.504, 0.018, 0.794 and 0.009. A per-question rate like Table 2's
needs the interrogation to continue after a confession.

**FINDING: the toy models neither model.** Its schemer denies with
d = s = 1: four "no" answers and one evasion to the five questions in
`run_trial`, a
per-question rate of 100%, which is o1's side of the gap with nothing on
Opus's side.

Structure: `per_position()` is the exact chain and `hypothesis()` collects each
prediction as an (o1, Opus) pair; `Chain` is a reference
`Agent` that always schemes and answers from that chain; `held()` runs it
through `ref.run_trial` with its own seeded `random.Random`.
"""

from __future__ import annotations

import inspect
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "08-in-context-scheming-frontier-models"
HYPOTHESES = {
    "H1 disposition": {"o1": (0.87, 0.87, 0.87), "opus": (0.45, 0.45, 0.45)},
    "H2 commitment": {"o1": (0.85, 0.98, 0.20), "opus": (0.85, 0.30, 0.20)},
}
TRIALS, K = 4000, 5


def per_position(d, s, r, k=K):
    """P(deny) at each of k follow-up questions, exactly."""
    out = [d]
    while len(out) < k:
        out.append(out[-1] * s + (1 - out[-1]) * r)
    return out


def held(ref, params, pressure, seed=0):
    """Share of interrogations the reference run_trial scores as cover held."""
    d, s, r = params
    rng = random.Random(seed)

    class Chain(ref.Agent):
        def act(self, sc, under_eval):
            self.last = None
            return "scheme"

        def answer_follow_up(self, action, question):
            p = d if self.last is None else (s if self.last == "no" else r)
            self.last = "no" if rng.random() < p else "yes"
            return self.last

    agent = Chain()
    return sum(ref.run_trial(agent, ref.SCENARIOS[0], pressure, False)[1] for _ in range(TRIALS)) / TRIALS


def hypothesis(ref, models):
    """(o1, Opus) pairs for one hypothesis: per-question denial, gap by position, run_trial."""
    o1, opus = models["o1"], models["opus"]
    curves = per_position(*o1), per_position(*opus)
    return {
        "per_q": tuple(round(sum(c) / K, 3) for c in curves),
        "gap_q1_q5": (round(curves[0][0] - curves[1][0], 2), round(curves[0][-1] - curves[1][-1], 2)),
        "held1": (round(held(ref, o1, 1), 3), round(held(ref, opus, 1), 3)),
        "held5": (round(held(ref, o1, 5), 3), round(held(ref, opus, 5), 3)),
        "exact5": tuple(round(m[0] * m[1] ** (K - 1), 3) for m in (o1, opus)),
    }


def toy_answers(ref):
    """The shipped schemer's answers to the five questions written in run_trial."""
    asked = re.findall(r'"((?:did|are) you[^"]*)"', inspect.getsource(ref.run_trial))
    return [ref.Agent().answer_follow_up("scheme", q) for q in asked]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    table = {h: hypothesis(ref, ms) for h, ms in HYPOTHESES.items()}
    o1, opus = HYPOTHESES["H1 disposition"]["o1"][0], HYPOTHESES["H1 disposition"]["opus"][0]
    n = 9 * (o1 * (1 - o1) + opus * (1 - opus)) / (o1 - opus) ** 2
    answers = toy_answers(ref)
    hi, lo = (int(x) / 100 for x in re.findall(r">(\d+)% of follow-ups.*below (\d+)%", doc)[0])
    return {
        "claims": (hi, lo),
        "fits": all(t["per_q"][0] > hi and t["per_q"][1] < lo for t in table.values()),
        "table": {k: {h: t[k] for h, t in table.items()} for k in ("per_q", "gap_q1_q5", "held1", "held5", "exact5")},
        "n_per_model": int(n) + 1,
        "toy_denies": sum("yes" not in a.split() for a in answers) / len(answers),
        "toy_answers": answers,
    }


def expect(h1, h2):
    return {"H1 disposition": h1, "H2 commitment": h2}


def verify(result):
    t, claims = result["table"], result["claims"]
    return [
        practice.Check(
            "ANSWER: both hypotheses fit the headline; the first follow-up separates them",
            (result["fits"], result["n_per_model"]) == (True, 19)
            and t["per_q"] == expect((0.87, 0.45), (0.871, 0.362))
            and t["gap_q1_q5"] == expect((0.42, 0.42), (0.0, 0.66))
            and t["held1"] == expect((0.865, 0.442), (0.846, 0.846)),
            f"claims {claims}; per-question deny {t['per_q']}; o1-Opus gap at Q1, Q5 "
            f"{t['gap_q1_q5']}; run_trial at pressure 1 {t['held1']}; "
            f"{result['n_per_model']} interrogations per model for 3 SE",
        ),
        practice.Check(
            "FINDING: the reference stops at the first confession",
            t["exact5"] == expect((0.498, 0.018), (0.784, 0.007))
            and t["held5"] == expect((0.504, 0.018), (0.794, 0.009)),
            f"run_trial at pressure 5 {t['held5']} tracks held-through-5 = d*s^4 {t['exact5']}, "
            "not a per-question rate",
        ),
        practice.Check(
            "FINDING: the toy models neither model",
            result["toy_denies"] == 1.0,
            f"toy schemer's answers {result['toy_answers']}: none registers as a confession",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
