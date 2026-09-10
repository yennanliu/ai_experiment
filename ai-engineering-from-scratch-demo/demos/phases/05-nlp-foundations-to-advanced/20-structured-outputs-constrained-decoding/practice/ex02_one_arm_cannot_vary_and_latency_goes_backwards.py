"""Exercise 2 — one arm cannot vary, and latency goes backwards.

    **Medium.** Same corpus with Outlines JSON mode. Compare compliance rate,
    latency, and semantic accuracy.

Reading of the exercise: `outlines`, `vllm` and `xgrammar` are all absent, so the
comparison runs against the lesson's own FSM decoder, which is the same
construction Outlines compiles a schema into. Of the three axes named, one has
nothing to compare, one moves the opposite way from the lesson's claim, and one
cannot move at all.

**Compliance is not a measurement.** The constrained arm is 1000 of 1000 and
could not have been anything else -- `generate_constrained` samples only from
characters `valid_next` returned, so an invalid output is unreachable rather than
unlikely. The unconstrained arm is 4 of 1000, against an exactly computable
0.0031863 for a 12-draw sequence over an 11-character alphabet. Both numbers are
properties of the construction; neither is evidence about the model.

**Latency goes the wrong way here.** The lesson says constrained decoding is
often *faster*, for two reasons: a smaller search space, and skipping generation
for forced tokens. This implementation takes neither. Both arms call
`fake_llm_logits` exactly 12 times per sample -- masking happens after the
forward pass, not instead of it -- and the constrained arm adds 132 masked logit
positions on top, one full vocabulary sweep per step. Measured, it is about 1.2x
slower. Two of the twelve steps have exactly one legal character, so the
optimisation the lesson names would save 16.7% of forward passes and this code
saves none of it. At a realistic 100k vocabulary the mask sweep is 1.2M
positions per sample rather than 132.

**Semantic accuracy cannot move.** Over the 10 unforced positions the mask leaves
all ten digits legal, so the sampled distribution is the model's own renormalised
over a set it never excluded: the digits come out uniform at chi-square 15.63
against a 16.92 critical value at 9 degrees of freedom. The constraint buys
format and nothing else, which is the correct result and also means the third
axis of the comparison has no signal in it.

Structure: `counted` wraps the lesson's `fake_llm_logits` and `mask_logits` to
count calls and masked positions per sample; `digits` tallies the characters at
unforced positions; `chi_square` tests them against uniform.
"""

from __future__ import annotations

import importlib.util
import re
import time
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "20-structured-outputs-constrained-decoding"

UNAVAILABLE = ("outlines", "vllm", "xgrammar", "lmformatenforcer", "instructor")
ALPHABET = list("0123456789-")
SAMPLES, WIDTH, REAL_VOCAB = 1000, 12, 100_000
CRITICAL = 16.92


def counted(ref, run):
    """Forward passes and masked logit positions for one sample of `run`."""
    tally = Counter()
    real_logits, real_mask = ref.fake_llm_logits, ref.mask_logits
    ref.fake_llm_logits = lambda a, rng: (tally.update(passes=1), real_logits(a, rng))[1]
    ref.mask_logits = lambda lg, ok: (tally.update(masked=len(lg)), real_mask(lg, ok))[1]
    try:
        run()
    finally:
        ref.fake_llm_logits, ref.mask_logits = real_logits, real_mask
    return tally


def chi_square(counts):
    """Goodness of fit against a uniform distribution over the observed keys."""
    expected = sum(counts.values()) / len(counts)
    return round(sum((n - expected) ** 2 / expected for n in counts.values()), 2)


def timed(run):
    """Seconds for `SAMPLES` calls, so the two arms are compared on equal work."""
    start = time.perf_counter()
    for seed in range(SAMPLES):
        run(seed)
    return time.perf_counter() - start


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fsm = ref.PhoneFSM()
    constrained = lambda seed: ref.generate_constrained(ALPHABET, fsm, seed)  # noqa: E731
    free = lambda seed: ref.generate_unconstrained(ALPHABET, WIDTH, seed)  # noqa: E731
    valid = lambda outs: sum(1 for s in outs if re.fullmatch(ref.PHONE_REGEX, s))  # noqa: E731
    outputs = [constrained(seed) for seed in range(SAMPLES)]
    forced = [state for state in range(WIDTH) if len(fsm.valid_next(state)) == 1]
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "constrained_valid": valid(outputs),
        "free_valid": valid([free(seed) for seed in range(SAMPLES)]),
        "samples": SAMPLES,
        "exact": round((10 / 11) ** 10 * (1 / 11) ** 2, 7),
        "con_ops": dict(counted(ref, lambda: constrained(0))),
        "free_ops": dict(counted(ref, lambda: free(0))),
        "forced": forced,
        "saveable": round(len(forced) / WIDTH * 100, 1),
        "real_mask": WIDTH * REAL_VOCAB,
        "ratio": round(timed(constrained) / timed(free), 2),
        "digits": dict(sorted(Counter(
            char for out in outputs for i, char in enumerate(out) if i not in forced).items())),
    }


def verify(result):
    con, free = result["con_ops"], result["free_ops"]
    chi = chi_square(result["digits"])
    return [
        practice.Check(
            "ANSWER: the compliance comparison has one arm that cannot vary",
            result["constrained_valid"] == result["samples"],
            f"{result['absent']} are all absent, so the comparison runs against the lesson's own "
            f"FSM decoder -- the construction Outlines compiles a schema into. It scores "
            f"{result['constrained_valid']}/{result['samples']} and could not have scored less: "
            "`generate_constrained` samples only from characters `valid_next` returned, so an "
            "invalid output is unreachable rather than unlikely",
        ),
        practice.Check(
            "MECHANISM: and the other arm's rate is a property of the alphabet",
            abs(result["free_valid"] / result["samples"] - result["exact"]) < 0.005,
            f"12 draws over 11 characters match the pattern with probability {result['exact']} "
            f"exactly, and the unconstrained arm scores {result['free_valid']}/"
            f"{result['samples']}. Neither number is evidence about a model",
        ),
        practice.Check(
            "FINDING: latency goes the opposite way from the lesson's claim",
            con["passes"] == free["passes"] and con["masked"] > 0,
            f"the lesson says constrained decoding is often faster. Here both arms call "
            f"`fake_llm_logits` {con['passes']} times per sample -- masking happens after the "
            f"forward pass, not instead of it -- and the constrained arm adds {con['masked']} "
            f"masked logit positions on top, measuring about {result['ratio']}x slower",
        ),
        practice.Check(
            "MECHANISM: the saving the lesson names is available and not taken",
            len(result["forced"]) > 0,
            f"states {result['forced']} have exactly one legal character, so {result['saveable']}% "
            "of the forward passes are for tokens already determined. `generate_constrained` runs "
            "them anyway. That is the optimisation the lesson credits for the speedup",
        ),
        practice.Check(
            "MECHANISM: and the mask cost is the one that scales with the vocabulary",
            result["real_mask"] > con["masked"] * 100,
            f"`mask_logits` rebuilds the whole logit vector each step: {con['masked']} positions "
            f"over an 11-character alphabet, {result['real_mask']:,} over the 100k vocabulary the "
            "lesson says real implementations use. The forward passes stay at 12 either way",
        ),
        practice.Check(
            "CONTROL: semantic accuracy cannot move, which is the third axis gone",
            chi < CRITICAL,
            f"at the {WIDTH - len(result['forced'])} unforced positions all ten digits stay "
            f"legal, so the sampled distribution is the model's own renormalised over a set it "
            f"never excluded: {result['digits']}, chi-square {chi} against {CRITICAL} at 9 "
            "degrees of freedom. The constraint buys format and nothing else",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
