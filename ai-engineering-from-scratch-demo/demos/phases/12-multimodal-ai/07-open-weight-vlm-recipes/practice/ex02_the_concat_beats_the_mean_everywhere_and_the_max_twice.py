"""Exercise 2 — the concat beats the mean everywhere and the max twice.

    Cambrian-1 finds that concatenating DINOv2 + SigLIP outperforms either alone
    on vision-centric benchmarks but adds no signal on MMMU. Predict which
    benchmarks gain and which stay flat.

Reading of the exercise: the prediction is made and then immediately checked,
because the lesson's own `compare_encoders` table already contains all three
rows -- SigLIP alone, DINOv2 alone, and the concatenation -- so this is one of
the rare exercises whose answer is sitting in the evidence it was written
against. Each benchmark is scored twice: against the better of the two parts,
and against their mean.

**ANSWER: CV-Bench gains, DocVQA loses, and MMMU gains slightly -- which is not
what the exercise says.** Against the better single encoder the concat is
**+2.0** on CV-Bench, **+1.0** on MMMU and **-1.0** on DocVQA. The premise
"adds no signal on MMMU" is contradicted by the lesson's own table, in the
direction of a gain.

**FINDING: against the *mean* of its parts the concat wins everywhere, and by
much more.** +3.0 on MMMU, +4.5 on CV-Bench, **+10.5** on DocVQA. Concatenating
two encoders reliably beats averaging them; whether it beats the better one is a
different question, and it is the only question that matters when you already
have to pick one.

**FINDING: the DocVQA gap is where the two encoders disagree most.** SigLIP
scores 75.0 there and DINOv2 **52.0** -- a 23-point gap, against 4.0 on MMMU and
5.0 on CV-Bench. That is exactly the benchmark where the concat fails to beat
the better part: adding a weak encoder's features costs something, and it costs
most where it is weakest.

**FINDING: so "vision-centric benchmarks gain" is a restatement of which
encoder is which.** DINOv2 beats SigLIP on CV-Bench alone, and CV-Bench is the
one benchmark where the concat beats both parts by more than a point. The
prediction the exercise asks for reduces to reading the DINOv2 row.

Structure: `encoder_rows` parses the lesson's printed table, `against` scores
the concat relative to the max and the mean of its two parts, and `disagreement`
is the per-benchmark gap between the two singles.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "07-open-weight-vlm-recipes"
BENCHMARKS = ("MMMU", "CV-Bench", "DocVQA")
SIGLIP = "SigLIP SO400m/14 @ 384"
DINOV2 = "DINOv2 ViT-g/14 @ 224"
CONCAT = "SigLIP + DINOv2 concat"


def encoder_rows(ref):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.compare_encoders()
    rows = {}
    for line in buffer.getvalue().splitlines():
        numbers = re.findall(r"\d+\.\d", line)
        if len(numbers) == len(BENCHMARKS):
            rows[line[:32].strip()] = [float(value) for value in numbers]
    return rows


def against(rows, reducer):
    """Concat minus a reduction of its two parts, per benchmark."""
    parts = (rows[SIGLIP], rows[DINOV2])
    return {name: round(rows[CONCAT][i] - reducer(part[i] for part in parts), 1)
            for i, name in enumerate(BENCHMARKS)}


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = encoder_rows(ref)
    versus_max, versus_mean = against(rows, max), against(rows, mean)
    gaps = {name: round(abs(rows[SIGLIP][i] - rows[DINOV2][i]), 1)
            for i, name in enumerate(BENCHMARKS)}
    return {
        "siglip": rows[SIGLIP], "dinov2": rows[DINOV2], "concat": rows[CONCAT],
        "versus_max": versus_max, "versus_mean": versus_mean,
        "gains": [name for name, delta in versus_max.items() if delta > 0],
        "losses": [name for name, delta in versus_max.items() if delta < 0],
        "mean_wins": all(delta > 0 for delta in versus_mean.values()),
        "gaps": gaps, "widest_gap": max(gaps, key=gaps.get),
        "dinov2_wins": [name for i, name in enumerate(BENCHMARKS)
                        if rows[DINOV2][i] > rows[SIGLIP][i]],
        "best_gain": max(versus_max, key=versus_max.get),
    }


def verify(result):
    versus_max, versus_mean, gaps = result["versus_max"], result["versus_mean"], result["gaps"]
    # Indexed eagerly by both the ok-expression and the detail below, so an empty
    # list must fail the check rather than raise out of Check construction.
    worst = result["losses"][0] if result["losses"] else None
    return [
        practice.Check(
            "ANSWER: CV-Bench gains, DocVQA loses, MMMU gains slightly",
            all([versus_max == {"MMMU": 1.0, "CV-Bench": 2.0, "DocVQA": -1.0},
                 result["gains"] == ["MMMU", "CV-Bench"],
                 result["losses"] == ["DocVQA"]]),
            f"against the better single encoder the concat is {versus_max}. The exercise's "
            f"premise is that it 'adds no signal on MMMU'; the lesson's own table says "
            f"{versus_max['MMMU']:+}, and the benchmark that actually goes backwards is "
            f"{worst}",
        ),
        practice.Check(
            "FINDING: against the mean of its parts the concat wins everywhere",
            all([versus_mean == {"MMMU": 3.0, "CV-Bench": 4.5, "DocVQA": 10.5},
                 result["mean_wins"]]),
            f"against the mean of SigLIP and DINOv2 the concat is {versus_mean} -- positive "
            "on all three and up to ten times the margin it has against the max. "
            "Concatenating two encoders reliably beats averaging them; beating the better "
            "one is a different question, and the only one that matters when you must pick",
        ),
        practice.Check(
            "FINDING: the DocVQA gap is where the two encoders disagree most",
            all([gaps == {"MMMU": 4.0, "CV-Bench": 5.0, "DocVQA": 23.0},
                 result["widest_gap"] == "DocVQA",
                 result["widest_gap"] == worst]),
            f"the two singles differ by {gaps}. DocVQA is a {gaps['DocVQA']}-point gap -- "
            f"SigLIP {result['siglip'][2]} against DINOv2 {result['dinov2'][2]} -- and it is "
            "the one benchmark where the concat fails to beat the better part. Adding a weak "
            "encoder's features costs something, and it costs most where it is weakest",
        ),
        practice.Check(
            "FINDING: 'vision-centric benchmarks gain' restates which encoder is which",
            all([result["dinov2_wins"] == ["CV-Bench"],
                 result["best_gain"] == "CV-Bench"]),
            f"DINOv2 beats SigLIP on {result['dinov2_wins']} and nowhere else, and "
            f"{result['best_gain']} is where the concat beats both parts by most. The "
            "prediction the exercise asks for reduces to reading the DINOv2 row",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
