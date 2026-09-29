"""Exercise 2 -- the trained model scores 0 on exact match and on soft accuracy, below the 1/128 chance level.

    Implement soft-accuracy VQA: multiple human answers per question, accuracy is `min(human_count / 3, 1)` if any matches. Replicates VQA v2.

Reading of the exercise: two scorers are built. `soft_stated` is the formula
as the exercise writes it. `soft_official` is what VQA v2's evaluation code
actually does (GT-Vision-Lab `vqaEval.py`,
https://raw.githubusercontent.com/GT-Vision-Lab/VQA/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py,
read 2026-09-29): for each of the 10 annotators, drop that one and apply
min(matches / 3, 1) to the other 9, then average the 10. The lesson's suite
has one answer per question, so each question gets 10 labelled annotations:
6 x the suite's answer and 2 x the first token of each of the two variant
reference captions, a stand-in for annotator disagreement. Both scorers run on
the VQA predictions the lesson's own `main()` makes (its `vqa_exact_match` is
wrapped and restored to record them), before and after training.

**ANSWER: `soft_stated` and `soft_official` below; the lesson's trained model
scores 0.0 on exact match and 0.0 on both soft scores.** Before training it
scores 0.02 / 0.0333 / 0.032. As sanity rows, the suite's own answers score
1.0 on all three. Always answering the variant token scores 0.2 on exact
match (10 of 50 variants have shift 0 and equal the answer), 0.733 stated and
0.68 official.

**FINDING: the exercise's formula is not VQA v2's.** By number of annotators
who gave the answer, the stated formula scores 0.333 / 0.667 / 1.0 at 1 / 2 / 3.
The official 9-of-10 average scores 0.3 / 0.6 / 0.9 and reaches 1.0 only at 4.

**FINDING: the suite's VQA cannot be answered, and the lesson's "VQA improving
above random" is false for its own run.** The answer is the first token of
the caption, which depends on the sample index i. The question is random
tokens, and the image carries only its brightness class i % 7. Any rule that
sees only the image tops out at 0.14 exact match. The model goes from
0.02 to 0.0 after training, below the 1/128 = 0.0078 chance level the
lesson's table gives.

Structure: `run_lesson()` records predictions and the suite; `annotations()`
builds the 10 answers; `image_only_ceiling()` takes the best answer per
brightness class.
"""


from __future__ import annotations

import contextlib
import io
from collections import Counter

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "63-multimodal-eval"


def run_lesson(ref):
    """The lesson's main() as shipped, recording its VQA predictions and its eval suite."""
    seen, saved_em, saved_ev = {"preds": [], "suite": None}, ref.vqa_exact_match, ref.evaluate

    def em(preds, refs):
        seen["preds"].append(list(preds))
        return saved_em(preds, refs)

    def evaluate(model, suite):
        seen["suite"] = suite
        return saved_ev(model, suite)

    ref.vqa_exact_match, ref.evaluate = em, evaluate
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.main()
    finally:
        ref.vqa_exact_match, ref.evaluate = saved_em, saved_ev
    return seen["preds"], seen["suite"]


def soft_stated(pred, answers):
    """The exercise's formula: min(#humans who gave the prediction / 3, 1)."""
    return min(answers.count(pred) / 3, 1.0)


def soft_official(pred, answers):
    """vqaEval.py: the stated formula against each 9-annotator subset, averaged over the 10."""
    return sum(min(answers[:i].count(pred) + answers[i + 1:].count(pred), 3) / 3
               for i in range(len(answers))) / len(answers)


def annotations(suite):
    """10 answers per question: 6 x the suite's answer, 2 x each variant caption's first token."""
    out = []
    for t, c in zip(suite.vqa, suite.caps):
        alt = [r[0] for r in c.references[1:]] + [t.answer_id] * 2
        out.append([t.answer_id] * 6 + [alt[0]] * 2 + [alt[1]] * 2)
    return out


def score(fn, preds, answers):
    return round(sum(fn(p, a) for p, a in zip(preds, answers)) / len(preds), 4)


def image_only_ceiling(suite):
    """Best exact match of any rule that sees only the image's brightness class (i % 7)."""
    groups = {}
    for i, t in enumerate(suite.vqa):
        groups.setdefault(i % 7, []).append(t.answer_id)
    return sum(Counter(g).most_common(1)[0][1] for g in groups.values()) / len(suite.vqa)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    (before, after), suite = run_lesson(ref)
    ans = annotations(suite)
    gold = [t.answer_id for t in suite.vqa]
    runs = {"before": before, "after": after, "gold": gold, "variant": [a[6] for a in ans]}
    table = {k: [round(fn(1, [1] * k + [0] * (10 - k)), 3) for k in range(5)]
             for k, fn in (("stated", soft_stated), ("official", soft_official))}
    return {
        "table": table,
        "em": {k: score(lambda p, a: float(p == a[0]), v, ans) for k, v in runs.items()},
        "stated": {k: score(soft_stated, v, ans) for k, v in runs.items()},
        "official": {k: score(soft_official, v, ans) for k, v in runs.items()},
        "ceiling": image_only_ceiling(suite),
        "distinct_answers": len(set(gold)),
        "doc_claim": "VQA improving above random" in parity.doc_text(PHASE, LESSON),
        "chance": round(1 / 128, 4),
    }


def verify(result):
    r, row = result, ("before", "after", "gold", "variant")
    got = [[r[m][k] for k in row] for m in ("em", "stated", "official")]
    em = r["em"]
    return [
        practice.Check(
            "ANSWER: the trained model scores 0 on exact match and on both soft scores",
            got == [[0.02, 0.0, 1.0, 0.2], [0.0333, 0.0, 1.0, 0.7333], [0.032, 0.0, 1.0, 0.68]],
            f"EM / stated / official for {row}: {got}",
        ),
        practice.Check(
            "FINDING: the exercise's formula is not VQA v2's",
            r["table"] == {"stated": [0.0, 0.333, 0.667, 1.0, 1.0], "official": [0.0, 0.3, 0.6, 0.9, 1.0]},
            f"0-4 matching annotators of 10: {r['table']}",
        ),
        practice.Check(
            "FINDING: the suite's VQA cannot be answered, and 'VQA improving above random' is false",
            (r["doc_claim"], r["ceiling"], em["after"] < r["chance"] < em["before"]) == (True, 0.14, True),
            f"image-only ceiling {r['ceiling']} over {r['distinct_answers']} distinct answers; "
            f"EM {em['before']} -> {em['after']} vs chance {r['chance']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
