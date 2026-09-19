"""Exercise 3 — sixty guesses score 1.00 and the honest three score a third.

    Write a JSON schema for temporal grounding that a VLM can learn to emit.
    Include error cases.

Reading of the exercise: the schema is written against the lesson's own `Event`
and `evaluate_grounding`, and every constraint in it is justified by running the
evaluator on the case that constraint exists to exclude. "Include error cases"
is taken as the instruction to find them rather than to list plausible ones, so
each is a measurement.

**ANSWER: the schema is five required fields and three constraints**, and every
constraint below is there because the evaluator mis-scores the case it rules
out:

```json
{"events": [{"event": "jump", "start": 4.1, "end": 4.3, "confidence": 0.82}],
 "duration": 10.0}
```
with `end > start` (strictly), `0 <= start < end <= duration`, and event spans of
the same label required to be disjoint.

**ERROR CASE 1 -- a zero-length event scores 0.0 when it is exactly right.**
`iou` returns `0.0` whenever the union is zero, so predicting `blink` at
[3.0, 3.0] against a ground truth of [3.0, 3.0] is recorded as a **miss**. Hence
`end > start` strictly, and instantaneous events encoded with a stated minimum
span.

**ERROR CASE 2 -- a reversed interval scores 0.0 and is not rejected.**
[5.0, 4.0] against [4.0, 5.0] gives a negative intersection, clamped to 0, and
passes through the evaluator as an ordinary miss. Hence the ordering constraint,
which a schema can enforce and the evaluator cannot.

**ERROR CASE 3 -- one prediction is credited for two ground-truth events.**
Matching is by label with no assignment step, so a single `jump` at [4.1, 4.6]
scores **0.667** against *both* [4.0, 4.5] and [4.2, 4.7] and takes recall to
**1.00**. Hence the disjointness constraint.

**FINDING: and the whole metric is recall, so guessing wins.** Sixty half-second
windows -- twenty per label, no model required -- score **1.00**, while the
lesson's own three honest predictions score **0.33** on its own demo. False
positives are not counted at all: adding a `wave` prediction that matches no
ground truth changes nothing. The schema needs a `confidence` field because the
metric needs a precision term, and neither exists yet.

Structure: `SCHEMA` is the proposed contract, `score` runs the lesson's own
evaluator on one case, and `CASES` pairs each constraint with the measurement
that justifies it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "17-video-language-temporal-grounding"
SCHEMA = {
    "required": ("event", "start", "end", "confidence"),
    "envelope": ("events", "duration"),
    "constraints": ("end > start", "0 <= start < end <= duration",
                    "same-label spans disjoint"),
}
TOLERANCE, WINDOW, LABELS = 0.3, 0.5, ("jump", "turn", "sit")
TRUTH = (("jump", 4.0, 4.5), ("turn", 6.0, 6.5), ("sit", 8.5, 9.5))
HONEST = (("jump", 4.1, 4.7), ("turn", 5.8, 6.2), ("sit", 9.2, 9.6))


def events(ref, rows):
    return [ref.Event(*row) for row in rows]


def score(ref, predictions, truth=TRUTH):
    return ref.evaluate_grounding(events(ref, predictions), events(ref, truth))


def sweep(duration=10.0, window=WINDOW, labels=LABELS):
    """Every window of `window` seconds, for every label -- no model involved."""
    steps = int(duration / window)
    return [(label, round(i * window, 2), round(i * window + window, 2))
            for label in labels for i in range(steps)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    zero_length = score(ref, [("blink", 3.0, 3.0)], (("blink", 3.0, 3.0),))
    reversed_span = score(ref, [("jump", 5.0, 4.0)], (("jump", 4.0, 5.0),))
    double = score(ref, [("jump", 4.1, 4.6)],
                   (("jump", 4.0, 4.5), ("jump", 4.2, 4.7)))
    guessed = sweep()
    return {
        "schema": SCHEMA, "constraints": len(SCHEMA["constraints"]),
        "fields": len(SCHEMA["required"]),
        "zero_length_iou": round(zero_length["details"][0][1], 3),
        "zero_length_hit": zero_length["details"][0][2],
        "iou_of_identical_points": ref.iou(3.0, 3.0, 3.0, 3.0),
        "reversed_iou": round(reversed_span["details"][0][1], 3),
        "reversed_rejected": False,
        "double_recall": double["recall"],
        "double_ious": [round(row[1], 3) for row in double["details"]],
        "honest_recall": round(score(ref, HONEST)["recall"], 3),
        "guessed_recall": round(score(ref, guessed)["recall"], 3),
        "guesses": len(guessed),
        "false_positive_recall": round(
            score(ref, [("jump", 4.1, 4.7), ("wave", 0.0, 10.0)],
                  (("jump", 4.0, 4.5),))["recall"], 3),
        "tolerance": TOLERANCE,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five fields and three constraints, each justified by a measurement",
            all([result["fields"] == 4, result["constraints"] == 3,
                 SCHEMA["envelope"] == ("events", "duration")]),
            f"the object carries {SCHEMA['required']} inside an {SCHEMA['envelope']} "
            f"envelope, under {result['constraints']} constraints: "
            f"{SCHEMA['constraints']}. Each of the three exists because the lesson's own "
            "evaluator mis-scores the case it rules out",
        ),
        practice.Check(
            "ERROR CASE 1: a zero-length event scores 0.0 when it is exactly right",
            all([result["zero_length_iou"] == 0.0, not result["zero_length_hit"],
                 result["iou_of_identical_points"] == 0.0]),
            f"iou returns {result['iou_of_identical_points']} whenever the union is zero, so "
            f"predicting blink at [3.0, 3.0] against a ground truth of [3.0, 3.0] is a miss. "
            "Hence end > start strictly, with instantaneous events given a stated minimum "
            "span rather than a point",
        ),
        practice.Check(
            "ERROR CASE 2: a reversed interval scores 0.0 and is not rejected",
            all([result["reversed_iou"] == 0.0, not result["reversed_rejected"]]),
            "[5.0, 4.0] against [4.0, 5.0] gives a negative intersection, clamped to 0 by "
            "max(0.0, ...), and passes through as an ordinary miss with no error raised. A "
            "schema can enforce the ordering; the evaluator cannot and does not",
        ),
        practice.Check(
            "ERROR CASE 3: one prediction is credited for two ground-truth events",
            all([result["double_recall"] == 1.0,
                 result["double_ious"] == [0.667, 0.667]]),
            f"matching is by label with no assignment step, so a single jump at [4.1, 4.6] "
            f"scores {result['double_ious']} against both [4.0, 4.5] and [4.2, 4.7] and takes "
            f"recall to {result['double_recall']}. Hence same-label spans must be disjoint",
        ),
        practice.Check(
            "FINDING: the whole metric is recall, so guessing wins",
            all([result["guessed_recall"] == 1.0, result["guesses"] == 60,
                 result["honest_recall"] == 0.333,
                 result["false_positive_recall"] == 1.0]),
            f"{result['guesses']} half-second windows -- twenty per label, no model involved "
            f"-- score {result['guessed_recall']}, while the lesson's own three honest "
            f"predictions score {result['honest_recall']} on its own demo. And a prediction "
            "matching no ground truth changes nothing. The schema needs a confidence field "
            "because the metric needs a precision term, and neither exists yet",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
