"""Exercise 1 — watermark z = 25.30 vs human -1.64, and the 95% threshold flags 4.7% of iid text but 52% of repeated text.

    Run `code/main.py`. Report z-scores for watermarked 1000-token generation
    vs human-authored text. Identify the false-positive rate at the 95%
    confidence threshold.

Reading of the exercise: the z-scores are read off the shipped run (the
module's own seed 61). "The 95% confidence threshold" is the one-sided
z >= 1.645 cut, which the reference never uses (it only reports z >= 4). The
false-positive rate there is measured two ways: exactly, because the reference's
green test on uniform tokens is a fair coin per position, so the green count is
Binomial(1000, 1/2); and empirically, over 200 seeded "human" draws from the
reference's own `unwatermarked_sample`.

**ANSWER: z = 25.30 for the watermarked 1000 tokens, -1.64 for the human
text, and 4.7% false positives at z >= 1.645.** 25.30 is exactly the expected
value 2 x (0.9 - 0.5) x sqrt(1000). At 95% a human text is flagged when at
least 527 of its 1000 tokens are green: probability 4.68% exactly, and 10 of
200 seeded draws (5.0%) measured. The reference's own FPR line (z >= 4 over
100 draws) prints 0.000, but zero hits in 100 only bounds the rate below
2.95% at 95% confidence, so the takeaway's "<1% FPR at z=4" is not what its
sample shows; the exact rate at z >= 4 is 0.0029%.

**FINDING: "human text" here is iid uniform tokens, and real repetition
breaks the 5% figure.** The detector scores every position, including repeats
of a context it has already scored. A human text that is one 5-token phrase
repeated (boilerplate, a chorus, a table) has only 5 distinct contexts, so its
green fraction is a multiple of 1/5 and z sits at +-6.32, +-18.97 or +-31.6.
Over 100 seeded phrases, 52% score above 1.645, against a nominal 5%.

Structure: `seeded()` swaps a `random.Random` into the reference and restores
it; `shipped()` parses the printed run; `binomial_tail()` is the exact rate.
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "23-watermarking-synthid-stable-signature-c2pa"
Z95, N, HUMAN_DRAWS, PHRASES = 1.645, 1000, 200, 100


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def shipped(ref):
    """The printed z-scores and FPR of main() on the module's own seed 61."""
    log = io.StringIO()
    with seeded(ref, 61), contextlib.redirect_stdout(log):
        ref.main()
    text = log.getvalue()
    grab = {k: float(re.search(rf"{k}\s*: (-?[\d.]+)", text).group(1))
            for k in ("watermarked z-score", "unwatermarked z-score", "over 100 human draws")}
    return grab, "<1% FPR at z=4" in text


def binomial_tail(n, z):
    """P(green count >= expected + z*sd) for a fair coin per position."""
    cut = math.ceil(n / 2 + z * math.sqrt(n) / 2)
    return cut, sum(math.comb(n, k) for k in range(cut, n + 1)) / 2 ** n


def human_fpr(ref):
    with seeded(ref, 0):
        prefix = [ref.random.randrange(ref.VOCAB) for _ in range(ref.K)]
        zs = [ref.detect(ref.unwatermarked_sample(N, prefix)) for _ in range(HUMAN_DRAWS)]
    return sum(z >= Z95 for z in zs)


def repeated_fpr(ref):
    """Human text that is one random 5-token phrase repeated to 1000+K tokens."""
    zs = []
    for s in range(PHRASES):
        rng = random.Random(s)
        phrase = [rng.randrange(ref.VOCAB) for _ in range(5)]
        zs.append(round(ref.detect((phrase * (N // 5 + 1))[:N + ref.K]), 2))
    return sum(z >= Z95 for z in zs) / PHRASES, sorted({abs(z) for z in zs})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    printed, claims_under_1pct = shipped(ref)
    cut, exact95 = binomial_tail(N, Z95)
    rep_rate, rep_levels = repeated_fpr(ref)
    return {
        "printed": printed, "claims_under_1pct": claims_under_1pct,
        "expected_wm_z": round(2 * (0.9 - 0.5) * math.sqrt(N), 2),
        "cut95": cut, "exact95": round(exact95, 4), "exact4": round(binomial_tail(N, 4)[1], 7),
        "human_hits": human_fpr(ref), "zero_in_100_bound": round(1 - 0.05 ** (1 / 100), 4),
        "repeated_rate": rep_rate, "repeated_levels": rep_levels,
    }


def verify(result):
    p = result["printed"]
    return [
        practice.Check(
            "ANSWER: z = 25.30 watermarked, -1.64 human, 4.7% false positives at z >= 1.645",
            (p["watermarked z-score"], result["expected_wm_z"], p["unwatermarked z-score"],
             result["cut95"], result["exact95"], result["human_hits"])
            == (25.3, 25.3, -1.64, 527, 0.0468, 10),
            f"printed {p}; exact FPR at 95% = {result['exact95']} (>= {result['cut95']} green); "
            f"measured {result['human_hits']}/{HUMAN_DRAWS}",
        ),
        practice.Check(
            "FINDING: the printed 0.000 at z >= 4 cannot support '<1% FPR'",
            (p["over 100 human draws"], result["claims_under_1pct"], result["zero_in_100_bound"],
             result["exact4"]) == (0.0, True, 0.0295, 0.000029),
            f"0/100 bounds the rate below {result['zero_in_100_bound']:.2%}; exact at z >= 4 "
            f"is {result['exact4']:.4%}",
        ),
        practice.Check(
            "FINDING: repeated human text is flagged 52% of the time at the 95% threshold",
            result["repeated_rate"] == 0.52 and result["repeated_levels"] == [6.32, 18.97, 31.62],
            f"{PHRASES} repeated 5-token phrases: {result['repeated_rate']:.0%} over 1.645; "
            f"|z| takes only the values {result['repeated_levels']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
