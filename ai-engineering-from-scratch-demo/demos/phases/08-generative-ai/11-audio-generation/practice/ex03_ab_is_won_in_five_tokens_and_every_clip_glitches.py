"""Exercise 3 — the A/B is won in five tokens, and every 10-second clip still glitches.

    **Hard.** Use HuggingFace transformers to run MusicGen-small locally.
    Generate a 10-second clip with three different prompts; A/B for style
    adherence.

Reading of the exercise: `transformers`, `torch` and `torchaudio` are absent and
MusicGen-small is ~2 GB of weights, so the real model cannot run here. What runs
instead is the same experiment on the lesson's own token model (`DESIGN D11`):
a "10-second clip" is 500 tokens, MusicGen's 50 Hz frame rate times 10 s; the
three prompts are style 0, style 1, and a third style slot the model was never
trained on (the lesson has only two, so a third prompt has nowhere to go). The
A/B judge prefers whichever of two clips has the higher log-likelihood under the
prompt's own model, using the lesson's `probs`; a glitch is a step outside the
style's training step set.

**ANSWER: for the two trained prompts, A/B adherence is 100%.** Over 100 pairs
of 500-token clips per prompt, the clip generated for the prompt wins every time.

**FINDING: 10 seconds is ~100x more audio than the A/B needs.** Shrinking the
clips, the judge already wins **90%** and **94%** of pairs at 2 tokens, and
**100%** at 5 -- a tenth of a second at 50 Hz. Clip-level adherence saturates
almost immediately, so a 10-second A/B cannot tell a good generator from a
mediocre one.

**FINDING: and yet no 10-second clip is clean.** At temperature 1.0 the average
500-token clip carries **11.0** off-pattern tokens and **0 of 200** clips are
glitch-free; at 0.7, **1.2** per clip and **63 of 200** are clean. A perfect A/B
score and a glitch in every clip are the same measurement read two ways: length
makes style easier to recognise and a clean clip harder to get.

**FINDING: the third prompt cannot be A/B'd at all.** Calling `generate` with
`style=2` raises **IndexError**; giving it an untrained table instead makes its
model uniform, so every clip scores the same likelihood under it and **100%**
of its A/B pairs are ties. Adherence to a prompt the model never learned is not
low, it is undefined.

**CONTROL: the real stack is absent, and that is checked rather than assumed.**
`find_spec` returns None for `transformers`, `torch` and `torchaudio`.

Structure: `train` mirrors `main()` and adds the untrained slot; `ab` runs the
judge over pairs; `glitches` counts off-pattern steps.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import random

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "11-audio-generation"
STEPS = {0: {0, 1, 2}, 1: {2, 3, 4}}
CLIP, PAIRS, SHORT = 500, 100, (2, 3, 5, 20)
NEEDED = ("transformers", "torch", "torchaudio")


def train(ref):
    """The lesson's table trained as main() trains it, plus an untrained third slot."""
    rng, counts = random.Random(42), ref.init_counts()
    for _ in range(500):
        for style in (0, 1):
            ref.update_counts(counts, ref.make_tokens(style, length=20, rng=rng), style)
    counts.append([[1.0] * 16 for _ in range(16)])
    return counts


def loglik(ref, counts, seq, style):
    return sum(
        math.log(ref.probs(counts, style, a)[b]) for a, b in itertools.pairwise(seq)
    )


def ab(ref, counts, style, other, length, rng):
    """(win rate, tie rate) of the prompt's own clip against the other prompt's clip."""
    wins = ties = 0
    for _ in range(PAIRS):
        mine = ref.generate(counts, style, 0, length, rng)
        theirs = ref.generate(counts, other, 0, length, rng)
        gap = loglik(ref, counts, mine, style) - loglik(ref, counts, theirs, style)
        wins, ties = wins + (gap > 0), ties + (gap == 0)
    return wins / PAIRS, ties / PAIRS


def glitches(seq, style):
    return sum((b - a) % 16 not in STEPS[style] for a, b in itertools.pairwise(seq))


def glitch_stats(ref, counts, temp, rng):
    """(mean glitches per 10-second clip, clean clips, clips) over both styles."""
    counts_per = [
        glitches(ref.generate(counts, s, 0, CLIP, rng, temp), s)
        for s in (0, 1)
        for _ in range(PAIRS)
    ]
    return sum(counts_per) / len(counts_per), counts_per.count(0), len(counts_per)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts, rng = train(ref), random.Random(5)
    try:
        ref.generate(ref.init_counts(), 2, 0, 5, rng)
        third = "accepted"
    except IndexError:
        third = "IndexError"
    return {
        "full": [ab(ref, counts, s, 1 - s, CLIP, rng) for s in (0, 1)],
        "short": {
            n: [ab(ref, counts, s, 1 - s, n, rng)[0] for s in (0, 1)] for n in SHORT
        },
        "untrained": ab(ref, counts, 2, 0, CLIP, rng),
        "third": third,
        "glitch": {t: glitch_stats(ref, counts, t, rng) for t in (1.0, 0.7)},
        "absent": [m for m in NEEDED if importlib.util.find_spec(m) is None],
    }


def verify(result):
    short, hot, cool = result["short"], result["glitch"][1.0], result["glitch"][0.7]
    return [
        practice.Check(
            "ANSWER: both trained prompts win their 10-second A/B every time",
            all(win == 1.0 for win, _ in result["full"]),
            f"over {PAIRS} pairs of {CLIP}-token clips per prompt, win rates are "
            f"{result['full'][0][0]:.0%} and {result['full'][1][0]:.0%}",
        ),
        practice.Check(
            "FINDING: the A/B saturates at 5 tokens, 100x short of 10 seconds",
            min(short[5]) == 1.0 and min(short[2]) < 1.0,
            "win rates by clip length -- "
            + ", ".join(f"{n} tokens: {a:.0%}/{b:.0%}" for n, (a, b) in short.items())
            + ". Five tokens is a tenth of a second at 50 Hz",
        ),
        practice.Check(
            "FINDING: yet no 10-second clip at temperature 1.0 is clean",
            hot[1] == 0 and hot[0] > 5 and cool[1] > 0,
            f"temperature 1.0: {hot[0]:.1f} off-pattern tokens per clip, {hot[1]} of {hot[2]} "
            f"clean; 0.7: {cool[0]:.1f} per clip, {cool[1]} of {cool[2]} clean. Length makes "
            "style easier to recognise and a clean clip harder to get",
        ),
        practice.Check(
            "FINDING: the third prompt cannot be A/B'd -- it crashes or ties",
            result["third"] == "IndexError" and result["untrained"][1] == 1.0,
            f"generate(style=2) raises {result['third']}; with an untrained slot the prompt's "
            f"model is uniform and {result['untrained'][1]:.0%} of its A/B pairs are ties",
        ),
        practice.Check(
            "CONTROL: the real MusicGen stack is absent, checked rather than assumed",
            result["absent"] == list(NEEDED),
            f"{result['absent']} return None from find_spec, so MusicGen-small cannot be loaded",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
