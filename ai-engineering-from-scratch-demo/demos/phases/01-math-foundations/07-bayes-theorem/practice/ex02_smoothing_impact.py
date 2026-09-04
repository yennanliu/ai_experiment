"""Exercise 2 — sweep Laplace smoothing, and the smoothing=0 failure.

    **Smoothing impact.** Run the spam classifier with smoothing values of 0.01,
    0.1, 1.0, and 10.0. How do the top word probabilities change? What happens
    with smoothing=0 and a word that appears only in ham?

Reading of the exercise: the last sentence names the outcome to reproduce, so
that is the check that matters. With smoothing=0 an unseen word has likelihood
0, `log(0)` raises — the model does not merely become confident, it stops being
computable. Checks 4 and 5 record exactly which failure the lesson's code hits,
rather than predicting one.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "07-bayes-theorem"
SMOOTHINGS = [0.01, 0.1, 1.0, 10.0]

DOCS = ["win free money now", "free money click here", "claim your free prize",
        "meeting at three tomorrow", "lunch tomorrow with the team",
        "project deadline moved to friday"]
LABELS = ["spam", "spam", "spam", "ham", "ham", "ham"]
HAM_ONLY = "deadline"                      # appears in ham, never in spam


def train(ref, smoothing):
    model = ref.NaiveBayes(smoothing=smoothing)
    model.train(DOCS, LABELS)
    return model


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bayes")
    rows = {}
    for smoothing in SMOOTHINGS:
        model = train(ref, smoothing)
        rows[smoothing] = {
            "top_spam": model.top_words("spam", 3),
            "p_free_spam": math.exp(model._log_likelihood("free", "spam")),
            "p_unseen_spam": math.exp(model._log_likelihood(HAM_ONLY, "spam")),
            "proba": model.predict_proba(f"free money {HAM_ONLY}"),
        }
    zero = train(ref, 0.0)
    try:
        zero._log_likelihood(HAM_ONLY, "spam")
        outcome = "returned a value — no error"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    return {"rows": rows, "zero": outcome,
            "vocab": len(train(ref, 1.0).vocab)}


def verify(result):
    rows = result["rows"]
    unseen = [rows[s]["p_unseen_spam"] for s in SMOOTHINGS]
    seen = [rows[s]["p_free_spam"] for s in SMOOTHINGS]
    rising = all(a < b for a, b in zip(unseen, unseen[1:]))
    falling = all(a > b for a, b in zip(seen, seen[1:]))
    return [
        practice.Check(f"all {len(SMOOTHINGS)} smoothing values train",
                       len(rows) == len(SMOOTHINGS),
                       f"vocabulary {result['vocab']} words over {len(DOCS)} documents"),
        practice.Check("more smoothing raises the probability of an unseen word",
                       rising,
                       "P('deadline'|spam): " + ", ".join(
                           f"α={s}→{rows[s]['p_unseen_spam']:.5f}" for s in SMOOTHINGS)),
        practice.Check("…and lowers it for a word the class actually contains",
                       falling,
                       "P('free'|spam): " + ", ".join(
                           f"α={s}→{rows[s]['p_free_spam']:.4f}" for s in SMOOTHINGS)
                       + " — smoothing is mass taken from the observed and given to the unseen"),
        practice.Check("smoothing=0 makes an unseen word uncomputable, not just unlikely",
                       "Error" in result["zero"] or "error" in result["zero"],
                       f"_log_likelihood('{HAM_ONLY}', 'spam') with α=0 -> {result['zero']} "
                       f"— log(0), so the whole document score is lost, not just this word"),
        practice.Check("at α=10 the classifier stops discriminating",
                       abs(rows[10.0]["proba"]["spam"] - 0.5)
                       < abs(rows[0.01]["proba"]["spam"] - 0.5),
                       f"P(spam) for 'free money {HAM_ONLY}': α=0.01 → "
                       f"{rows[0.01]['proba']['spam']:.4f}, α=10 → "
                       f"{rows[10.0]['proba']['spam']:.4f} — heavy smoothing washes the "
                       f"evidence out toward the prior"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
