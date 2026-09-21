"""Exercise 1 — low confidence and unknown category reach the same handler.

    Implement routing with a confidence threshold. Below threshold ->
    escalate to human. Where does the threshold land for a tier-1 support use
    case?

Reading of the exercise: `route` takes a classifier typed `str -> str`, so a
confidence has nowhere to travel. The threshold is added by widening that to
`str -> (str, float)`, and "where does it land" is a cost question, not a
preference: escalating costs one human touch, a misroute costs the reply plus
the re-contact. The threshold is therefore swept against a cost model and
reported with the ratio it depends on.

**ANSWER: at a 5:1 misroute-to-escalation cost ratio the threshold lands at
0.3.** Over **20** tickets the margin-based classifier is right **14** times.
Routing everything costs **30**; thresholding at **0.3** escalates **9**,
misroutes **0**, and costs **9**. Tier-1 support is the case where the ratio
is high, which is why the answer is "escalate readily".

**FINDING: the threshold is stable and the cost is not.** At ratios 2, 5 and
10 the optimum stays at **0.3** while routing everything costs **12**, **30**
and **60**. That is a statement about *calibration*, not about the ratio:
every error this classifier makes sits at confidence **0.25** or below, so
one cut separates them at any price. A threshold moves with the ratio only
when the score is miscalibrated.

**FINDING: `route` cannot carry a confidence.** The classifier is called as
`classifier(input_text)` and its return value is used directly as a
dictionary key, so the only way to express "unsure" in the shipped signature
is to return a label that is not in `handlers`.

**FINDING: which is exactly what the default handler already does.**
`handlers.get(label) or handlers.get("default")` sends an unknown label to
`default`, so a deliberate escalation and a classifier that invented a
category produce byte-identical results -- **2** different events, **1**
observable outcome, and the returned label is the only clue.

Structure: `classify()` returns a label and the margin behind it; `sweep()`
scores every threshold under a stated cost model.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "12-anthropic-workflow-patterns"
KEYWORDS = {
    "refund": ("refund", "money", "charge", "billing", "invoice"),
    "bug": ("crash", "error", "broken", "stack", "fails"),
    "sales": ("pricing", "quote", "volume", "demo", "contract"),
}
TICKETS = (
    ("i want a refund for this charge", "refund"),
    ("the billing invoice is wrong", "refund"),
    ("please refund my money", "refund"),
    ("my card was charged twice", "refund"),
    ("the cli crash dumps a stack", "bug"),
    ("upload fails with an error", "bug"),
    ("the export is broken", "bug"),
    ("it crash on ctrl-c", "bug"),
    ("do you offer volume pricing", "sales"),
    ("can i get a quote for a demo", "sales"),
    ("we need a contract and pricing", "sales"),
    ("send me the volume contract", "sales"),
    ("the refund page fails with an error", "refund"),
    ("pricing page is broken", "sales"),
    ("charge me monthly on the contract", "refund"),
    ("hello", "sales"),
    ("thanks for the help", "bug"),
    ("is anyone there", "refund"),
    ("the demo crash on launch", "sales"),
    ("billing quote please", "refund"),
)
THRESHOLDS = tuple(round(0.1 * k, 1) for k in range(11))


def classify(text):
    """Label plus the margin behind it, which is the confidence the router lacks."""
    words = set(text.lower().split())
    scores = sorted(((len(words & set(terms)), label)
                     for label, terms in KEYWORDS.items()), reverse=True)
    top, runner = scores[0], scores[1]
    total = sum(count for count, _ in scores)
    return top[1], (top[0] - runner[0]) / (total + 1)


def route_with_threshold(ref, text, handlers, threshold):
    label, confidence = classify(text)
    if confidence < threshold:
        return "escalated", handlers["default"](text)
    return ref.route(text, lambda _: label, handlers)


def handlers_for():
    return {name: (lambda t, n=name: f"handled {n}") for name in KEYWORDS} | {
        "default": lambda t: "escalate to human"}


def sweep(ref, ratio):
    handlers, rows = handlers_for(), []
    for threshold in THRESHOLDS:
        escalated = misrouted = 0
        for text, truth in TICKETS:
            label, _ = route_with_threshold(ref, text, handlers, threshold)
            escalated += label == "escalated"
            misrouted += label not in ("escalated", truth)
        rows.append((ratio * misrouted + escalated, threshold, escalated, misrouted))
    return sorted(rows)[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    best = sweep(ref, 5)
    handlers = handlers_for()
    unknown = ref.route("hello", lambda _: "no_such_label", handlers)
    deliberate = ref.route("hello", lambda _: "default", handlers)
    return {
        "tickets": len(TICKETS),
        "correct": sum(classify(text)[0] == truth for text, truth in TICKETS),
        "best": {"cost": best[0], "threshold": best[1], "escalated": best[2],
                 "misrouted": best[3]},
        "route_all": {ratio: ratio * sum(classify(text)[0] != truth
                                         for text, truth in TICKETS)
                      for ratio in (2, 5, 10)},
        "by_ratio": {ratio: sweep(ref, ratio)[1] for ratio in (2, 5, 10)},
        "cost_at_best": {ratio: sweep(ref, ratio)[0] for ratio in (2, 5, 10)},
        "worst_error_confidence": max(
            classify(text)[1] for text, truth in TICKETS if classify(text)[0] != truth),
        "unknown": unknown, "deliberate": deliberate,
        "same_output": unknown[1] == deliberate[1],
    }


def verify(result):
    best = result["best"]
    return [
        practice.Check(
            "ANSWER: at a 5:1 cost ratio the threshold lands at 0.3",
            all([best["threshold"] == 0.3, best["cost"] == 9,
                 best["escalated"] == 9, best["misrouted"] == 0,
                 result["correct"] == 14, result["tickets"] == 20]),
            f"over {result['tickets']} tickets the classifier is right "
            f"{result['correct']} times, so routing everything costs "
            f"{result['route_all'][5]}. Thresholding at {best['threshold']} escalates "
            f"{best['escalated']} and misroutes {best['misrouted']}, for a cost of "
            f"{best['cost']} -- tier-1 support is where that ratio is high",
        ),
        practice.Check(
            "FINDING: the threshold is stable and the cost is not",
            all([set(result["by_ratio"].values()) == {0.3},
                 result["route_all"] == {2: 12, 5: 30, 10: 60},
                 result["worst_error_confidence"] == 0.25]),
            f"at ratios 2, 5 and 10 the optimum stays at {result['by_ratio'][5]} while "
            f"routing everything costs {result['route_all']}. Every error this "
            f"classifier makes sits at confidence {result['worst_error_confidence']} or "
            "below, so one cut separates them at any price -- the threshold moves with "
            "the ratio only when the score is miscalibrated",
        ),
        practice.Check(
            "FINDING: route cannot carry a confidence",
            all([result["unknown"][0] == "no_such_label",
                 result["unknown"][1] == "escalate to human"]),
            f"the classifier's return value is used directly as a dictionary key, so the "
            f"only way to say 'unsure' in the shipped signature is to return a label "
            f"that is not in handlers -- which returns {result['unknown']!r}, the "
            "default handler's output under a label nobody registered",
        ),
        practice.Check(
            "FINDING: the default handler already collapses the two",
            all([result["same_output"] is True,
                 result["deliberate"][0] == "default",
                 result["unknown"][0] != result["deliberate"][0]]),
            f"a deliberate escalation and a classifier that invented a category both "
            f"return {result['deliberate'][1]!r} ({result['same_output']}). Two "
            "different events, one observable outcome, and the returned label is the "
            "only thing that distinguishes them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
