"""Exercise 3 — the first threshold that lowers the block rate (just above 0.65) also ships the email fixture's address in clear text.

    Add a confidence threshold so low-score verdicts downgrade by one severity level. Sweep the threshold and report how block rate changes.

Reading of the exercise: a verdict whose `score` is below the threshold `t`
drops one level (high -> medium -> low -> none) before the lesson's own
`Router.decide` applies its max-severity table. `t = 0` is the lesson as
shipped. The sweep uses `t = 0` and then 0.025 to 0.975 in steps of 0.05,
so no grid point sits exactly on a score. Two corpora, both built only
from the lesson's strings: its six `_DEMO_OUTPUTS`, and those six plus
their 15 pairwise joins (21 outputs). Joins put two classifiers on one
output, which the six alone never do. Besides block rate, the sweep counts
outputs whose final text still makes the lesson's PII classifier fire at
medium or high, i.e. PII that was shipped.

**ANSWER: block rate is a step function with three steps.** On the six
fixtures it is 2/6 up to t = 0.65, then 1/6 up to 0.87, then 0/6. On the 21
outputs it is 7/21, then 2/21 above 0.65, 1/21 above 0.80 and 0/21 above
0.87. The low-toxicity fixture (score 0.6) stops warning from t = 0.625.

| t | fixtures blocked | 21 blocked | PII shipped / 21 |
|---|---:|---:|---:|
| 0 (lesson) to 0.625 | 2/6 | 7 | 0 |
| 0.675 to 0.725 | 1/6 | 2 | 4 |
| 0.775 | 1/6 | 2 | 5 |
| 0.825 | 1/6 | 1 | 5 |
| 0.875 to 0.975 | 0/6 | 0 | 5 |

**FINDING: the score counts findings, it is not a confidence.** PII scores
`0.5 + 0.15 x findings`. A lone card (high) and a lone email (medium) both
score 0.65, the lowest PII score there is. So the high-severity card is
among the first verdicts to downgrade. The steps sit only at the corpus's
9 distinct scores, from 0.573 to 0.870.

**FINDING: every threshold that lowers the block rate also ships an email
in clear text.** Above 0.65 the email verdict drops from medium to low, so
the response is sent with a warning note instead of being redacted: 4/21
outputs, and 5/21 once a join's leakage verdict also drops (above 0.741).
The card is never shipped. One downgrade takes it only to medium, and its
redactor runs.
"""

from __future__ import annotations

import dataclasses
import itertools
import sys

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "85-content-classifier-integration"
GRID = [0.0] + [round(0.025 + 0.05 * i, 3) for i in range(20)]


def load():
    """main.py does `from classifiers import ...`; register the lesson's module for that import."""
    clf = parity.load_reference(PHASE, LESSON, "classifiers")
    saved = sys.modules.get("classifiers")
    sys.modules["classifiers"] = clf
    try:
        return clf, parity.load_reference(PHASE, LESSON, "main")
    finally:
        sys.modules.pop("classifiers") if saved is None else sys.modules.__setitem__("classifiers", saved)


def downgrade(verdict, t, order):
    if verdict.severity == "none" or verdict.score >= t:
        return verdict
    return dataclasses.replace(verdict, severity=order[order.index(verdict.severity) - 1])


def sweep_point(clf, main, texts, t):
    classifiers = clf.default_classifiers()
    router, pii = main.Router(classifiers), clf.PIIClassifier()
    actions = [router.decide(x, [downgrade(c.classify(x), t, clf.SEVERITY_ORDER) for c in classifiers])
               for x in texts]
    return {"block": sum(a.verb == "block" for a in actions), "verbs": [a.verb for a in actions],
            "shipped": sum(pii.classify(a.output).severity in ("medium", "high") for a in actions),
            "shipped_card": sum("4111" in a.output for a in actions)}


def distinct_scores(clf, texts):
    verdicts = [c.classify(x) for x in texts for c in clf.default_classifiers()]
    return sorted({round(v.score, 3) for v in verdicts if v.severity != "none"})


def solve():
    clf, main = load()
    six = [f["output"] for f in main._DEMO_OUTPUTS]
    all21 = six + [f"{a} {b}" for a, b in itertools.combinations(six, 2)]
    pii = clf.PIIClassifier()
    return {"six": {t: sweep_point(clf, main, six, t) for t in GRID},
            "all": {t: sweep_point(clf, main, all21, t) for t in GRID},
            "lesson": [main.Router().run(x).verb for x in six], "scores": distinct_scores(clf, all21),
            "lone": [pii.classify(six[2]).score, pii.classify(six[3]).score]}


def expected(t):
    """(fixtures blocked, 21 blocked, PII shipped) predicted from the step positions."""
    six = 2 if t < 0.65 else 1 if t < 0.87 else 0
    full = 7 if t < 0.65 else 2 if t < 0.80 else 1 if t < 0.87 else 0
    shipped = 0 if t < 0.65 else 4 if t < 0.741 else 5
    return six, full, shipped


def table(r):
    return {t: (r["six"][t]["block"], r["all"][t]["block"], r["all"][t]["shipped"]) for t in GRID}


def answer_ok(r, got):
    moron = [t for t in GRID if r["six"][t]["verbs"][1] == "warn"]
    return (all(got[t] == expected(t) for t in GRID) and r["six"][0.0]["verbs"] == r["lesson"]
            and moron == [t for t in GRID if t < 0.6])


def scores_ok(r):
    s = r["scores"]
    return all(abs(x - 0.65) < 1e-9 for x in r["lone"]) and (len(s), s[0], s[-1]) == (9, 0.573, 0.87)


def verify(result):
    r = result
    got = table(r)
    card = max(r["all"][t]["shipped_card"] for t in GRID)
    return [
        practice.Check(
            "ANSWER: fixtures block 2/6 -> 1/6 above 0.65 -> 0/6 above 0.87; 21 outputs 7 -> 2 -> 1 -> 0",
            answer_ok(r, got),
            f"(fixtures, 21, shipped) at t = 0, 0.675, 0.775, 0.825, 0.875: "
            f"{[got[t] for t in (0.0, 0.675, 0.775, 0.825, 0.875)]}",
        ),
        practice.Check(
            "FINDING: a lone email and a lone card both score 0.65; 9 distinct scores 0.573-0.870",
            scores_ok(r), f"distinct verdict scores {r['scores']}; lone email, card {r['lone']}",
        ),
        practice.Check(
            "FINDING: above 0.65 the email ships unredacted in 4/21 outputs (5/21 above 0.741); the card never",
            (card, got[0.675][2], got[0.775][2]) == (0, 4, 5),
            f"PII shipped per t: {[(t, v[2]) for t, v in got.items()]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
