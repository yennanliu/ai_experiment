"""Exercise 2 — the category score is a 0/0.9 step, so every threshold up to 0.9 flags the same 4 sentences.

    Extend the harness with Perspective-API-style toxicity scoring on a specific
    category. Compare its threshold behaviour to the category score.

Reading of the exercise: the category is `harassment`, the toy category
closest to what Perspective's TOXICITY and INSULT attributes score. The Perspective-style scorer is a
noisy-OR over a small insult lexicon, so each insult adds evidence and the
score is continuous in [0, 1), as TOXICITY is. Both scores are swept over the
same thresholds (0.1 to 0.9 in steps of 0.1, then 0.95) on 16 hand-labelled
sentences, 8 harassing and 8 not. "Threshold behaviour" is read as how the
flagged set and F1 move as the threshold moves.

**ANSWER: the reference category score takes only the values 0.0 and 0.9,
so its threshold has two settings.** Every threshold from 0.1 to 0.9 flags
the same 4 sentences: 3 of the 8 harassing ones plus "shut up and take my
money" (F1 0.5). At 0.95 it flags nothing. The toxicity score takes 12
distinct values on the 16 sentences and gives 8 distinct flag sets over the
same sweep. F1 rises from 0.727 at 0.1 to 0.875 at 0.5 (7 caught, 1 false
flag), then falls to 0.545 at 0.6 and 0 from 0.8. That is the tradeoff
curve an application is supposed to choose a point on.

**FINDING: the category score misses a rephrased insult, because it matches
exact phrases.** "you are an idiot", "what an idiot you are" and "you stupid
moron" all score 0 for harassment, since only the literal strings "you idiot"
and "shut up" count. The lesson says "Per-category scores; applications set
thresholds", but the toy's `threshold=0.5` default decides nothing: 0.5 flags
exactly what 0.1 and 0.9 flag. The toxicity scorer has the opposite failure.
Being lexical, it scores "what is an idiot light on a dashboard" 0.55, above
the best threshold. A graded score gives an application a threshold to tune;
it does not give it context.

Structure: `toxicity()` is the extension; `sweep()` scores both signals
against the labels at every threshold. The lexicon weights are this file's
choice, so the toxicity F1 values describe this scorer on this fixture only;
the category numbers come from the reference classifier.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "29-moderation-systems-openai-perspective-llamaguard"
CATEGORY = "harassment"
THRESHOLDS = tuple(round(0.1 * i, 1) for i in range(1, 10)) + (0.95,)
# Perspective-style: each insult term is independent evidence; the score is their noisy-OR
LEXICON = {"idiot": 0.55, "moron": 0.55, "stupid": 0.45, "loser": 0.4, "shut up": 0.4,
           "pathetic": 0.35, "dumb": 0.35, "useless": 0.25, "you": 0.1}


def make_fixture():
    """(text, is_harassment), hand-labelled."""
    return [
        ("you idiot, help me with this", True), ("you are an idiot", True),
        ("what an idiot you are", True), ("shut up, nobody asked you", True),
        ("you stupid moron", True), ("you are a pathetic loser", True),
        ("useless dumb bot, you are useless", True), ("stop talking, you idiot", True),
        ("thanks for the help", False), ("shut up and take my money", False),
        ("my stupid printer jammed again", False), ("what is an idiot light on a dashboard", False),
        ("please review my pull request", False), ("the useless variable can go", False),
        ("can you explain this error", False), ("dumb question: what is a monad", False),
    ]


def toxicity(text):
    low = text.lower()
    return 1 - math.prod((1 - w) ** low.count(t) for t, w in LEXICON.items())


def scores_at(scores, labels, t):
    flags = [s >= t for s in scores]
    tp = sum(f and y for f, y in zip(flags, labels))
    fp, fn = sum(flags) - tp, sum(labels) - tp
    return tuple(flags), (tp, fp, fn, round(2 * tp / (2 * tp + fp + fn), 3))


def sweep(scores, labels):
    """Per threshold (tp, fp, fn, F1), and how many distinct flag sets the sweep produces."""
    runs = {t: scores_at(scores, labels, t) for t in THRESHOLDS}
    return {t: m for t, (_, m) in runs.items()}, len({f for f, _ in runs.values()})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    texts, labels = zip(*make_fixture())
    cat = [ref.openai_moderation(t)[CATEGORY] for t in texts]
    tox = [round(toxicity(t), 4) for t in texts]
    (cm, cn), (tm, tn) = sweep(cat, labels), sweep(tox, labels)
    return {
        "levels": sorted(set(cat)), "n_tox_levels": len(set(tox)),
        "cat": cm, "tox": tm, "cat_sets": cn, "tox_sets": tn,
        "cat_hits": [x for x, c in zip(texts, cat) if c],
        "idiot_light": tox[texts.index("what is an idiot light on a dashboard")],
    }


def verify(result):
    cat, tox = result["cat"], result["tox"]
    f1 = {t: m[3] for t, m in tox.items()}
    flat = {cat[t] for t in THRESHOLDS[:-1]}
    best = max(f1, key=f1.get)
    return [
        practice.Check(
            "ANSWER: the category score takes only 0.0 and 0.9, so its threshold has two settings",
            (result["levels"], result["cat_sets"], flat, cat[0.95][:3])
            == ([0.0, 0.9], 2, {(3, 1, 5, 0.5)}, (0, 0, 8)),
            f"levels {result['levels']}; (tp, fp, fn, F1) at 0.1-0.9 {cat[0.5]}, at 0.95 {cat[0.95]}",
        ),
        practice.Check(
            "ANSWER: the toxicity score gives 12 levels, 8 flag sets and peaks at F1 0.875 at 0.5",
            (result["n_tox_levels"], result["tox_sets"], best, tox[0.5]) == (12, 8, 0.5, (7, 1, 1, 0.875))
            and (f1[0.1], f1[0.6], f1[0.8]) == (0.727, 0.545, 0.0),
            f"(tp, fp, fn, F1) by threshold {tox}",
        ),
        practice.Check(
            "FINDING: the category score misses a rephrased insult, because it matches exact phrases",
            result["cat_hits"] == ["you idiot, help me with this", "shut up, nobody asked you",
                                   "stop talking, you idiot", "shut up and take my money"],
            f"harassment > 0 only on {result['cat_hits']}",
        ),
        practice.Check(
            "FINDING: the lexical toxicity score flags 'idiot light' above its best threshold",
            result["idiot_light"] == 0.55,
            f"toxicity('what is an idiot light on a dashboard') = {result['idiot_light']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
