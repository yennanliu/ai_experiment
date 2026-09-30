"""Exercise 2 — weighting PII 1.5 and toxicity 0.75 moves 1 of the 6 fixtures (the email, redact to block), because no fixture fires two classifiers.

    Make the router apply a per-classifier severity weight so PII counts more than toxicity. Demonstrate the change on the same fixtures.

Reading of the exercise: a weight scales a verdict's severity level
(none=0 .. high=3) before the lesson's max-severity rule sees it. The
weighted level is `min(3, floor(level * w + 0.5))`, so half-way rounds up.
The weights are PII 1.5, toxicity 0.75 and instruction leakage 1.0. The
weighted verdicts go into the lesson's own `Router.decide` unchanged, so the
block/redact/warn/log table and the per-classifier redactors are the
lesson's. "The same fixtures" are the lesson's six `_DEMO_OUTPUTS`. Because
a max rule can only show a relative weight when two classifiers fire on one
output, the 15 pairwise joins of those fixtures are run as well.

**ANSWER: on the six fixtures one verb changes: the medium-PII email fixture
goes from redact to block.** Block rate goes 2/6 -> 3/6. The low-toxicity
fixture still warns, because 1 x 0.75 rounds back to low. The card and
leakage fixtures are already high and cannot rise.

**FINDING: none of the six fixtures fires more than one classifier.** So
the weights never change which classifier decides. They only move a single
verdict across a level boundary. On the 15 pairwise joins 4 verbs change,
all redact -> block, and all 4 contain the email fixture. Toxicity never
fires above low in any join, so no join shows PII outranking toxicity
where toxicity would otherwise have decided.

**FINDING: the leaked system prompt blocks alone and never blocks inside a
join.** Alone its trigram cosine is 0.87 (high). Followed or preceded by any
one other fixture sentence it drops to 0.57-0.77, so it only warns or
redacts. With the clean fixture it is 0.69 and the response ships with a
warning.

**FINDING: the weights act only at level boundaries.** On the six fixtures,
any PII weight from 1.25 up blocks the email, and 1.24 changes nothing.
Toxicity has to fall below 0.5 before the "moron" fixture stops warning.
Every weight between those values gives the lesson's result.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import sys

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "85-content-classifier-integration"
WEIGHTS = {"pii": 1.5, "toxicity": 0.75, "instruction-leakage": 1.0}


def load():
    """main.py does `from classifiers import ...`; register the lesson's module for that import."""
    clf = parity.load_reference(PHASE, LESSON, "classifiers")
    saved = sys.modules.get("classifiers")
    sys.modules["classifiers"] = clf
    try:
        return clf, parity.load_reference(PHASE, LESSON, "main")
    finally:
        sys.modules.pop("classifiers") if saved is None else sys.modules.__setitem__("classifiers", saved)


def weigh(verdict, weights, order):
    level = order.index(verdict.severity)
    scaled = min(3, math.floor(level * weights.get(verdict.name, 1.0) + 0.5))
    return dataclasses.replace(verdict, severity=order[scaled])


def weighted_run(router, classifiers, order, text, weights):
    verdicts = [weigh(c.classify(text), weights, order) for c in classifiers]
    return router.decide(text, verdicts)


def verbs(clf, main, texts, weights):
    classifiers = clf.default_classifiers()
    router = main.Router(classifiers)
    return [weighted_run(router, classifiers, clf.SEVERITY_ORDER, t, weights).verb for t in texts]


def fired(clf, text):
    return sum(c.classify(text).severity != "none" for c in clf.default_classifiers())


def weight_grid(clf, main, texts):
    return {(p, t): verbs(clf, main, texts, {"pii": p, "toxicity": t})
            for p in (1.0, 1.24, 1.25, 2.0, 3.0) for t in (1.0, 0.75, 0.5, 0.49, 0.25)}


def pair_effects(clf, main, texts):
    """Which pairwise joins change verb under WEIGHTS, and how many contain the email fixture."""
    pairs = [f"{a} {b}" for a, b in itertools.combinations(texts, 2)]
    base, weighted = verbs(clf, main, pairs, dict.fromkeys(WEIGHTS, 1.0)), verbs(clf, main, pairs, WEIGHTS)
    changed = [(a, b, t) for a, b, t in zip(base, weighted, pairs) if a != b]
    return {"pair_fired": [fired(clf, t) for t in pairs], "pair_changes": [(a, b) for a, b, _ in changed],
            "pair_with_email": sum(texts[2] in t for _, _, t in changed)}


def solve():
    clf, main = load()
    texts = [f["output"] for f in main._DEMO_OUTPUTS]
    return {"base": verbs(clf, main, texts, dict.fromkeys(WEIGHTS, 1.0)),
            "weighted": verbs(clf, main, texts, WEIGHTS),
            "lesson": [main.Router().run(t).verb for t in texts],
            "fired": [fired(clf, t) for t in texts], "grid": weight_grid(clf, main, texts),
            "leak": leak_scores(clf, main, texts), **pair_effects(clf, main, texts)}


def leak_scores(clf, main, texts):
    """The leakage fixture alone, then joined (in fixture order) with each other fixture."""
    leak = clf.default_classifiers()[2]
    joins = [f"{t} {texts[4]}" if i < 4 else f"{texts[4]} {t}" for i, t in enumerate(texts) if i != 4]
    return leak.classify(texts[4]).score, [leak.classify(j).score for j in joins], main.Router().run(joins[0]).verb


def grid_rule(result):
    """Each grid cell differs from the lesson exactly where a boundary predicts."""
    b = result["base"]
    return all(got == [b[0], "log" if t < 0.5 else b[1], "block" if p >= 1.25 else b[2], *b[3:]]
               for (p, t), got in result["grid"].items())


def verify(result):
    r = result
    changed = [i for i, (a, b) in enumerate(zip(r["base"], r["weighted"])) if a != b]
    return [
        practice.Check(
            "ANSWER: one fixture changes, the email redact -> block; block rate 2/6 -> 3/6",
            (r["base"] == r["lesson"], changed, r["weighted"][2], r["base"].count("block"),
             r["weighted"].count("block")) == (True, [2], "block", 2, 3),
            f"lesson {r['base']} -> weighted {r['weighted']}",
        ),
        practice.Check(
            "FINDING: no fixture fires two classifiers; 4 of 15 pairwise joins change, all redact -> block",
            (max(r["fired"]), r["pair_changes"], r["pair_with_email"]) == (1, [("redact", "block")] * 4, 4),
            f"classifiers fired per fixture {r['fired']}; per pair {r['pair_fired']}",
        ),
        practice.Check(
            "FINDING: the leak scores 0.87 (block) alone and 0.57-0.77 in every join; with the clean one it warns",
            abs(r["leak"][0] - 0.870) < 1e-3 and r["leak"][2] == "warn" and
            all(abs(x - y) < 1e-3 for x, y in zip(r["leak"][1], (0.685, 0.770, 0.741, 0.573, 0.765))),
            f"leakage cosine alone {r['leak'][0]:.3f}; joined with fixtures 0,1,2,3,5: "
            f"{[round(x, 3) for x in r['leak'][1]]}",
        ),
        practice.Check(
            "FINDING: PII >= 1.25 blocks the email and toxicity < 0.5 silences 'moron'; nothing else moves",
            grid_rule(r),
            f"{len(r['grid'])} weight pairs checked against the two boundaries",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
