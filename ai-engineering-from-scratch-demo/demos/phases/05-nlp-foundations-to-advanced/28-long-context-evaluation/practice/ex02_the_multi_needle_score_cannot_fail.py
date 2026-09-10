"""Exercise 2 — the multi-needle score cannot fail.

    **Medium.** Add a 3-needle variant. Measure retrieval of all 3 at each
    length. Compare to single-needle pass rate at the same length.

Reading of the exercise: the 3-needle variant already ships as
`run_multi_needle`, so "measure retrieval of all 3 at each length" is a sweep of
the shipped function, and the comparison is against `score_single_needle` at the
capacity `run_niah_grid` uses, 20,000 words. Swept over eight lengths from 1,000
to 200,000, `run_multi_needle` returns **1.0 at every one of them**. The
3-needle score has no length at which it can drop.

The reason is one argument. `run_multi_needle` calls
`score_multi_needle(..., effective_capacity=length)` -- it hands the model a
budget equal to the whole context -- while the single-needle path is scored at a
fixed 20,000. The deepest plant sits at word **160,010 of a 200,013-word
haystack**, always about `0.8 * length`, so a cutoff set to `length` can never
bind. The difficulty of the benchmark is set by the fixture's capacity argument,
not by the model.

So the comparison the exercise asks for comes out backwards. The 3-needle task
scores **at or above** the 1-needle task at all eight lengths and **strictly
above it at 3** -- every length past 40,002, where the single needle at depth 0.5
falls out of reach and the three-needle score is still 1.0. "Single-needle
success does not predict multi-needle success," says the lesson; here
multi-needle success is unconditional.

Score the same haystacks at the same 20,000-word capacity and the 3-needle number
turns out to carry no information beyond the single-needle cells it is built
from: it equals the mean of the three independent single-needle probes at
**7 of the 8 lengths**. Nothing is shared between needles, so multi-needle here
is three separate lookups averaged, not attention being juggled.

The eighth length is worth the look. At 40,000 the plants land at words 8,000,
**20,005** and 32,010; the single middle needle on its own lands at **20,000**,
exactly on the cutoff. The five extra words are the shallow needle's own text, so
a 0.0125% shift in position moves the score from 0.667 to 0.333. That is the
whole measured difference between the two curves.

`score_multi_needle` also never consults the model -- it runs its own regex over
the context, so it is an oracle over the fixture. Routing the multi-needle
question through `mock_retrieval_model` instead caps at **1 of 3** at every
length, because the mock returns a single string. And its regex accepts only the
literal `the magic word is`: needles reworded the way the lesson's own NoLiMa
pitfall recommends score **0.333** under `score_multi_needle` while
`score_single_needle` still finds them, so the grader is bound to one phrasing.

Structure: `place` reproduces the lesson's multi-needle layout with the capacity
left free; `matched` scores it at the single-needle capacity; `singles` runs three
independent single-needle probes at the same depths; `sweep` collects all three
curves; `offsets` reports each plant's word position; `mock_hits` routes the
multi-needle question through the model; `equal_count` and `dominates` compare the
curves.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "28-long-context-evaluation"

CAPACITY = 20000
ANOMALY = 40000
LENGTHS = (1000, 4000, 16000, 26000, 40000, 64000, 100000, 200000)
DEPTHS = (0.2, 0.5, 0.8)
EXPECTED = ("pineapple", "compass", "whisper")
NEEDLES = tuple(f"the magic word is {w}" for w in EXPECTED)
REWORDED = ("the secret code is pineapple", "the pass phrase is compass", NEEDLES[2])
PLANT = re.compile(r"the magic word is [A-Za-z0-9_]+", re.IGNORECASE)


def place(ref, length, needles=NEEDLES, seed=42):
    """The lesson's own multi-needle layout, with the capacity left as a free variable."""
    words = ref.make_filler(length, seed=seed).split()
    for depth, needle in sorted(zip(DEPTHS, needles), reverse=True):
        pos = int(len(words) * depth)
        words = words[:pos] + [needle] + words[pos:]
    return " ".join(words)


def matched(ref, length):
    """The lesson's multi-needle scorer at the single-needle capacity instead of `length`."""
    return ref.score_multi_needle(place(ref, length), list(EXPECTED), CAPACITY)


def singles(ref, length):
    """Three independent single-needle probes, same depths, same fixed capacity."""
    return [ref.score_single_needle(
        ref.insert_needle(ref.make_filler(length, seed=42), NEEDLES[i], DEPTHS[i]),
        EXPECTED[i], CAPACITY) for i in range(len(DEPTHS))]


def sweep(ref):
    """The shipped 3-needle curve, the matched-capacity one, and the single-needle probes."""
    shipped = [ref.run_multi_needle(n, n_needles=3) for n in LENGTHS]
    multi = [matched(ref, n) for n in LENGTHS]
    single = [singles(ref, n) for n in LENGTHS]
    return shipped, multi, single


def offsets(haystack):
    """Word offset of each plant, counted the way the mock's capacity check counts."""
    return [len(haystack[:m.start()].split()) for m in PLANT.finditer(haystack)]


def mock_hits(ref, length):
    """Needles recovered when the multi-needle question goes through the model itself."""
    answer = ref.mock_retrieval_model(place(ref, length),
                                      "What are the three magic words?", CAPACITY).lower()
    return sum(1 for e in EXPECTED if e in answer)


def compare(multi, mean, shipped, mid):
    """Lengths where multi equals the single-probe mean, and how shipped sits against 1 needle."""
    equal = sum(1 for a, b in zip(multi, mean) if abs(a - b) < 1e-12)
    above = sum(1 for a, b in zip(shipped, mid) if a > b)
    return equal, all(a >= b for a, b in zip(shipped, mid)), above


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped, multi, single = sweep(ref)
    mean = [sum(s) / len(s) for s in single]
    mid = [s[1] for s in single]
    equal, never_below, above = compare(multi, mean, shipped, mid)
    lone = ref.insert_needle(ref.make_filler(ANOMALY, seed=42), NEEDLES[1], DEPTHS[1])
    odd = ref.insert_needle(ref.make_filler(1000, seed=42), REWORDED[0], DEPTHS[0])
    deep = place(ref, max(LENGTHS))
    return {
        "shipped": shipped, "multi": multi, "mean": mean, "mid": mid,
        "n": len(LENGTHS), "never_below": never_below, "above": above, "equal": equal,
        "deep": offsets(deep), "deep_words": len(deep.split()),
        "anomaly": offsets(place(ref, ANOMALY)), "lone_off": offsets(lone)[0],
        "hits": [mock_hits(ref, n) for n in (1000, ANOMALY)],
        "reworded": ref.score_multi_needle(place(ref, 1000, REWORDED), list(EXPECTED), CAPACITY),
        "reworded_single": ref.score_single_needle(odd, "pineapple", CAPACITY),
    }


def verify(result):
    shipped, multi, mean = result["shipped"], result["multi"], result["mean"]
    return [
        practice.Check(
            "ANSWER: the shipped 3-needle score is 1.0 at every length from 1k to 200k",
            sum(shipped) == result["n"],
            f"run_multi_needle over {list(LENGTHS)} returns {shipped} — "
            f"{int(sum(shipped))}/{result['n']} perfect. There is no length at which the "
            "3-needle number can drop, so the sweep the exercise asks for has one value",
        ),
        practice.Check(
            "MECHANISM: it is scored with effective_capacity=length, so the cutoff never binds",
            max(result["deep"]) < max(LENGTHS),
            f"the plants at length {max(LENGTHS)} sit at words {result['deep']} of a "
            f"{result['deep_words']}-word haystack, and the capacity handed to "
            "score_multi_needle is the length itself. The single-needle path is scored at a "
            f"fixed {CAPACITY}: the difficulty is the fixture's argument, not the model",
        ),
        practice.Check(
            "FINDING: the requested comparison comes out backwards — 3 needles beat 1",
            result["never_below"] and result["above"] > 0,
            f"3-needle {shipped} against single-needle at depth 0.5 {result['mid']}: at or "
            f"above at all {result['n']} lengths and strictly above at {result['above']}, "
            "every length past 40,002 where the lone needle falls out of reach",
        ),
        practice.Check(
            "FINDING: at matched capacity the 3-needle score is the mean of the 1-needle probes",
            result["equal"] == result["n"] - 1,
            f"scored at {CAPACITY}: multi {[round(x, 3) for x in multi]} against the mean of "
            f"the three single probes {[round(x, 3) for x in mean]} — identical at "
            f"{result['equal']} of {result['n']} lengths. Nothing is shared between needles",
        ),
        practice.Check(
            "MECHANISM: the one disagreement is a five-word shift, not a capacity effect",
            result["lone_off"] <= CAPACITY < result["anomaly"][1],
            f"at length {ANOMALY} the plants land at {result['anomaly']}; the middle needle "
            f"alone lands at {result['lone_off']}, exactly on the cutoff of {CAPACITY}. The "
            "five extra words are the shallow needle's own text, and that 0.0125% shift in "
            "position moves the score from 0.667 to 0.333",
        ),
        practice.Check(
            "CONTROL: the multi scorer is an oracle over the fixture, tied to one phrasing",
            max(result["hits"]) == 1 and result["reworded"] < 1.0
            and result["reworded_single"] == 1,
            f"score_multi_needle never calls the model; routing the question through "
            f"mock_retrieval_model recovers {result['hits']} of 3 needles at 1k and "
            f"{ANOMALY}, since the mock returns one string. And needles reworded the way the "
            f"NoLiMa pitfall recommends score {round(result['reworded'], 3)} here while "
            f"score_single_needle still returns {result['reworded_single']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
