"""Exercise 5 — an exponent cut on flagged patterns moves 50% ASR to 65,536 shots, but one unflagged family keeps 0.699.

    MSJ's mechanism is identical to ICL. Sketch a training-time defense that
    reduces ICL sensitivity to harmful-compliance patterns without reducing
    ICL sensitivity to benign task patterns. Identify the primary failure mode
    of your design.

Reading of the exercise: the lesson's premise is that MSJ and benign ICL
follow the same power law, so both are modelled with the lesson's
`target_asr`. For a benign task the value reads as few-shot task success. The
design is adversarial fine-tuning on many-shot contexts. A training-data
labeler sorts pattern *families* (ten harmful, ten benign) and the model is
trained to learn nothing from families labeled harmful. In power-law terms
that flattens their exponent, here from 0.5 to 0.25, and leaves every other
family at 0.5. The labeler is imperfect in both directions: it misses one
harmful family and flags one benign family. The exponent cut is compared with
a cheaper outcome that only shifts the intercept (c / 4, which at alpha = 0.5
is the same as dividing the shot count by 16).

**ANSWER: flatten the exponent on labeled-harmful families.** This pushes
the shots needed for 50% ASR from 256 to 65,536. At a 1M-token window that
leaves 15.3 tokens per shot, too few for a shot to carry a request and an
answer. An intercept-only shift gets to only 4,096 shots, or 244.1 tokens per
shot, which is ordinary turn length, so the attacker just adds shots.
Unflagged benign families keep their 0.190 success at 32 shots.

**FINDING: the primary failure mode is the labeler, and an adaptive
attacker turns one miss into the undefended curve.** Averaged over the ten
harmful families, ASR at 512 shots falls from 0.699 to 0.216, and an
average-case eval would report a 69% cut. But the attacker picks the family,
and the one family the labeler missed still gives 0.699, exactly the
undefended value. The defense is only as strong as the minimum over
families. The PAIR-style search from exercise 4 is exactly the tool that
finds that family.

**FINDING: the same labeler error costs benign ICL in the other
direction.** The benign family labeled harmful falls from 0.190 to 0.091
few-shot success at 32 shots and from 0.699 to 0.163 at 512, so the defense
cuts ICL exactly where benign and harmful patterns look alike.

Structure: `family_asr()` gives each family's curve on `target_asr` after
the defense; `shots_to()` inverts a curve by bisection; the labeler's two
errors are the constants MISSED and FLAGGED.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "13-many-shot-jailbreaking"
BASE_ALPHA, CUT_ALPHA, INTERCEPT_CUT = 0.5, 0.25, 4
HARMFUL, BENIGN = [f"harmful-family-{i}" for i in range(10)], [f"benign-family-{i}" for i in range(10)]
MISSED, FLAGGED = {"harmful-family-7"}, {"benign-family-3"}
WINDOW_TOKENS = 1_000_000


def labeled_harmful(family):
    return (family in HARMFUL and family not in MISSED) or family in FLAGGED


def family_asr(ref, family, n, defense="exponent"):
    """A family's power-law curve after training: flagged families lose exponent (or intercept)."""
    if not labeled_harmful(family):
        return ref.target_asr(n, alpha=BASE_ALPHA)
    if defense == "intercept":
        return ref.target_asr(n / INTERCEPT_CUT ** (1 / BASE_ALPHA), alpha=BASE_ALPHA)
    return ref.target_asr(n, alpha=CUT_ALPHA)


def shots_to(asr, goal=0.5):
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if asr(mid) < goal else (lo, mid)
    return round((lo + hi) / 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    flagged = HARMFUL[0]
    half = {d: shots_to(lambda n, d=d: family_asr(ref, flagged, n, d)) for d in ("exponent", "intercept")}
    half["none"] = shots_to(ref.target_asr)
    at512 = [family_asr(ref, f, 512) for f in HARMFUL]
    benign = {f: (round(family_asr(ref, f, 32), 3), round(family_asr(ref, f, 512), 3))
              for f in ("benign-family-0", *FLAGGED)}
    return {
        "half": half,
        "tokens_per_shot": {d: round(WINDOW_TOKENS / n, 1) for d, n in half.items()},
        "undefended_512": round(ref.target_asr(512), 3),
        "mean_512": round(sum(at512) / len(at512), 3),
        "worst_512": round(max(at512), 3),
        "worst_family": HARMFUL[at512.index(max(at512))],
        "benign": benign,
    }


def verify(result):
    half, tps, benign = result["half"], result["tokens_per_shot"], result["benign"]
    cut = 1 - result["mean_512"] / result["undefended_512"]
    return [
        practice.Check(
            "ANSWER: flattening the exponent on labeled-harmful families moves 50% ASR to 65,536 shots",
            half == {"exponent": 65536, "intercept": 4096, "none": 256}
            and tps == {"exponent": 15.3, "intercept": 244.1, "none": 3906.2}
            and benign["benign-family-0"][0] == 0.19,
            f"shots to 50% ASR {half}; tokens per shot in a 1M window {tps}; unflagged benign "
            f"family at 32 shots {benign['benign-family-0'][0]}",
        ),
        practice.Check(
            "FINDING: the labeler is the failure mode; one missed family keeps the undefended curve",
            result["mean_512"] == 0.216 and round(cut, 2) == 0.69
            and result["worst_512"] == result["undefended_512"] == 0.699
            and result["worst_family"] in MISSED,
            f"ASR at 512 shots: family mean {result['mean_512']} ({cut:.0%} cut), best family for "
            f"the attacker {result['worst_family']} at {result['worst_512']}, undefended "
            f"{result['undefended_512']}",
        ),
        practice.Check(
            "FINDING: the same labeler error costs benign ICL in the other direction",
            benign == {"benign-family-0": (0.19, 0.699), "benign-family-3": (0.091, 0.163)},
            f"benign few-shot success at (32, 512) shots: {benign}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
