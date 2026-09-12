"""Exercise 2 — the masks all land in the first sixth of the sentence.

    **Medium.** Implement BART's `text_infill` noise: replace random spans with
    a single `<mask>` token, and the decoder must infer the correct span length
    plus contents. Show one example.

Reading of the exercise: `text_infill` already exists in `code/main.py`, so
"implement" is read as *run it at the rates the lesson uses and check it does
what BART's noise does*. One example is shown, and then the same call is run 500
times per rate so the example is not the evidence.

**ANSWER: one example at rate 0.3 on a 30-token sentence** --

    <mask> brown fox <mask> <mask> <mask> in time saves nine language models
    learn statistical patterns subword tokenization helps rare words today and
    again forever more

Four masks hide 9 of the 30 tokens and the source is 25 long. The corrupted text
does not say how many tokens each `<mask>` swallowed: measured, one mask hides
anywhere from **1 to 6** tokens. That non-invertibility is the objective -- it is
what the exercise means by "the decoder must infer the correct span length".

**FINDING: every mask lands at the front.** The mean relative position of a
`<mask>` is **0.158** at rate 0.15, where uniform would be 0.5. `text_infill`
walks left to right opening a span with a fixed probability of 0.3 per position
and decrementing a budget of `int(n * rate)`, so at the lesson's own default the
budget is spent inside the first few positions and **the last two thirds of every
sentence are never corrupted**. Raise the rate to 0.6 and the mean moves to
0.494, which is the tell: the skew is the budget running out, not the noise.

**FINDING: the 0.3 is not the rate.** `rate` sets only the budget; the per-position
probability of opening a span is hardcoded. So `rate` controls *how much* is
masked and never *where*, and those are the same knob in BART.

**FINDING: zero-length spans never occur.** BART's text infilling inserts a
`<mask>` covering nothing, which is what teaches the model that a mask can mean
"delete me". `max(1, ...)` forbids it: over 7,811 sampled spans the lengths are
1-6 and **0 appears zero times**.

**CONTROL: the total is right even though the placement is not.** The realised
hidden fraction is 13.3% at rate 0.15 and 29.8% at 0.30 -- unlike `corrupt_spans`
next door, which under-masks by half.

Structure: `positions` measures where the masks land; `hidden` measures how many
tokens they swallow; `lengths` samples the span-length expression directly.
"""

from __future__ import annotations

import collections
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "08-t5-bart-encoder-decoder"
MEAN_SPAN, TRIALS, LENGTH, MASK = 3.0, 500, 30, "<mask>"
TEXT = ("the quick brown fox jumps over the lazy dog a stitch in time saves nine language "
        "models learn statistical patterns subword tokenization helps rare words today and "
        "again forever more")


def positions(ref, sentence, rate, trials=TRIALS):
    """Mean relative position of a <mask> in the corrupted source; 0.5 would be uniform."""
    seen = [index / max(1, len(out) - 1)
            for seed in range(trials)
            for out in [ref.text_infill(sentence, rate, MEAN_SPAN, random.Random(seed))]
            for index, token in enumerate(out) if token == MASK]
    return statistics.fmean(seen), len(seen)


def hidden(ref, sentence, rate, trials=TRIALS):
    """(mean hidden fraction, mean tokens per mask) over many corruptions."""
    fractions, per_mask = [], []
    for seed in range(trials):
        out = ref.text_infill(sentence, rate, MEAN_SPAN, random.Random(seed))
        masks = sum(1 for token in out if token == MASK)
        gone = len(sentence) - (len(out) - masks)
        fractions.append(gone / len(sentence))
        per_mask += [gone / masks] if masks else []
    return statistics.fmean(fractions), statistics.fmean(per_mask)


def lengths(rate, trials=2_000, length=LENGTH):
    """The span-length expression itself, sampled: max(1, min(gauss, budget, n - i))."""
    counts = collections.Counter()
    for seed in range(trials):
        rng, budget, index = random.Random(seed), int(length * rate), 0
        while index < length:
            if budget > 0 and rng.random() < 0.3:
                span = max(1, min(int(rng.gauss(MEAN_SPAN, 1.0)), budget, length - index))
                counts[span] += 1
                budget, index = budget - span, index + span
            else:
                index += 1
    return counts


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sentence = TEXT.split()[:LENGTH]
    example = ref.text_infill(sentence, 0.3, MEAN_SPAN, random.Random(3))
    spans = lengths(0.3)
    masks = sum(1 for t in example if t == MASK)
    return {
        "example": example, "masks": masks, "length": len(sentence),
        "example_hidden": len(sentence) - (len(example) - masks),
        "position": {rate: positions(ref, sentence, rate)[0] for rate in (0.15, 0.3, 0.6)},
        "hidden": {rate: hidden(ref, sentence, rate) for rate in (0.15, 0.3)},
        "spans": dict(sorted(spans.items())), "sampled": sum(spans.values()),
        "zero": spans[0],
    }


def verify(result):
    place, hide, spans = result["position"], result["hidden"], result["spans"]
    return [
        practice.Check(
            "ANSWER: one mask hides 1 to 6 tokens, and the source does not say which",
            min(spans) == 1 and max(spans) >= 5,
            f"the example at rate 0.3 carries {result['masks']} masks hiding "
            f"{result['example_hidden']} of {result['length']} tokens in a source of "
            f"{len(result['example'])}. Sampling the span-length expression "
            f"{result['sampled']:,} times gives lengths {list(spans)} -- so the corrupted text "
            "cannot say how many tokens a mask swallowed, which is the objective",
        ),
        practice.Check(
            "FINDING: every mask lands at the front -- mean position 0.158, not 0.5",
            place[0.15] < 0.25 and place[0.6] > 0.4,
            f"mean relative <mask> position by rate: "
            f"{ {r: round(v, 3) for r, v in place.items()} }, against 0.5 for uniform. "
            "text_infill walks left to right opening a span with probability 0.3 per position "
            f"while decrementing a budget of int(n * rate), so at rate 0.15 the budget is gone "
            "inside the first sixth and the rest of the sentence is never corrupted",
        ),
        practice.Check(
            "FINDING: the 0.3 is hardcoded, so rate sets how much and never where",
            place[0.6] > 2 * place[0.15],
            f"raising rate from 0.15 to 0.6 moves the mean position from {place[0.15]:.3f} to "
            f"{place[0.6]:.3f}, which is the tell: the skew is the budget running out, not the "
            "noise model. In BART the two are the same knob; here `rate` buys budget and the "
            "per-position probability of opening a span is a literal 0.3",
        ),
        practice.Check(
            "FINDING: zero-length spans never occur, so a mask can never mean 'delete me'",
            result["zero"] == 0 and min(spans) == 1,
            f"BART's text infilling inserts a <mask> covering nothing, which is what teaches the "
            f"model that a mask can be removed. max(1, ...) forbids it: over {result['sampled']:,} "
            f"sampled spans the lengths are {min(spans)}-{max(spans)} and 0 appears "
            f"{result['zero']} times",
        ),
        practice.Check(
            "CONTROL: the total is right even though the placement is not",
            abs(hide[0.15][0] - 0.133) < 0.02 and abs(hide[0.3][0] - 0.30) < 0.02,
            f"realised hidden fraction {hide[0.15][0]:.1%} at rate 0.15 and {hide[0.3][0]:.1%} at "
            f"0.30, with {hide[0.3][1]:.2f} tokens per mask. The budget arithmetic is sound -- "
            "unlike corrupt_spans next door, which asks for 15% and delivers 7.8%. What is wrong "
            "here is only where the budget gets spent",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
