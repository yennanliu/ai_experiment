"""Exercise 1 — the sequences match their style, and the "alternating" style is a ramp.

    **Easy.** Run `code/main.py` and set style explicitly. Verify the generated
    sequences match the style's pattern.

Reading of the exercise: `main()` already loops over both styles, so "set style
explicitly" is read as calling the lesson's own `generate` with `style=0` and
`style=1` directly, after training the lesson's own count table exactly as
`main()` does (seed 42, 500 sequences of 20 per style). "The style's pattern" is
taken from the generator that made the training data, `make_tokens`: each step
moves forward by `{0, 1, 2}` for style 0 and `{2, 3, 4}` for style 1 (mod 16).
A generated step outside that set is a glitch.

**ANSWER: yes, up to a glitch rate the smoothing sets.** Over 1000 sequences per
style at temperature 1.0, **2.3%** (style 0) and **2.2%** (style 1) of steps are
off-pattern, so only **64%** and **65%** of 20-token sequences are entirely clean.
At the 0.7 temperature `main()` also uses, the rate drops to **0.28%** and
**0.24%**. The glitch rate is the add-one prior: `init_counts` starts every cell
at 1.0, which leaves **2.6%** of each row's mass on the 13 tokens no training
step ever reached.

**FINDING: style 0 is not alternating.** `main()` labels it "speech-like
(alternating)" and the doc says "alternating low and high tokens", but
`make_tokens(0, ...)` is `(i + jitter) % 16`: a **+1 ramp with a stutter**.
**100%** of its 9500 training steps go forward by 0, 1 or 2 -- none alternate.
Both styles are ramps that differ only in speed, and they share the +2 step
(25% of each style's steps), so no single step can say which style made it.

**FINDING: "explicitly" works for exactly two values, and fails silently for
some others.** `generate(counts, 2, ...)` raises **IndexError**, but `style=-1`
is accepted and returns style 1's ramp, because Python reads `counts[-1]` as the
last table; `make_tokens` likewise treats any non-zero style as style 1.

**CONTROL: the glitches come from the model, not the data.** The training
sequences themselves are **0.0%** off-pattern for both styles.

Structure: `train` mirrors `main()`'s loop; `off_rate` scores generated steps
against the style's step set.
"""

from __future__ import annotations

import itertools
import random

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "11-audio-generation"
STEPS = {0: {0, 1, 2}, 1: {2, 3, 4}}
SAMPLES, LENGTH, TEMPS = 1000, 20, (1.0, 0.7)


def steps(seq):
    """Forward step sizes between consecutive tokens, mod the vocabulary."""
    return [(b - a) % 16 for a, b in itertools.pairwise(seq)]


def train(ref):
    """The lesson's count table, trained exactly as main() trains it; plus the data."""
    rng, counts, data = random.Random(42), ref.init_counts(), {0: [], 1: []}
    for _ in range(500):
        for style in (0, 1):
            seq = ref.make_tokens(style, length=LENGTH, rng=rng)
            ref.update_counts(counts, seq, style)
            data[style].append(seq)
    return counts, data


def off_rate(seqs, style):
    """(fraction of off-pattern steps, fraction of sequences with none)."""
    flags = [[s not in STEPS[style] for s in steps(seq)] for seq in seqs]
    total = sum(len(f) for f in flags)
    return sum(sum(f) for f in flags) / total, sum(not any(f) for f in flags) / len(
        flags
    )


def sampled(ref, counts, style, temp):
    rng = random.Random(1)
    return [ref.generate(counts, style, 0, LENGTH, rng, temp) for _ in range(SAMPLES)]


def third_style(ref, counts):
    """What generate() does with style=2, which has no table."""
    try:
        ref.generate(counts, 2, 0, 5, random.Random(0))
    except IndexError:
        return "IndexError"
    return "accepted"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts, data = train(ref)
    rates = {
        (s, t): off_rate(sampled(ref, counts, s, t), s) for s in (0, 1) for t in TEMPS
    }
    row = ref.probs(counts, 0, 5)
    style0 = [s for seq in data[0] for s in steps(seq)]
    return {
        "rates": rates,
        "prior_mass": 1 - sum(row[(5 + d) % 16] for d in STEPS[0]),
        "forward": sum(s in STEPS[0] for s in style0) / len(style0),
        "shared": style0.count(2) / len(style0),
        "n_steps": len(style0),
        "style2": third_style(ref, counts),
        "alias": ref.generate(counts, -1, 0, 10, random.Random(0)),
        "data_off": [off_rate(data[s], s)[0] for s in (0, 1)],
    }


def verify(result):
    rates = result["rates"]
    hot, cool = (rates[(0, 1.0)], rates[(1, 1.0)]), (rates[(0, 0.7)], rates[(1, 0.7)])
    alias = result["alias"]
    return [
        practice.Check(
            "ANSWER: sequences match their style, up to the smoothing's glitch rate",
            all(r[0] < 0.03 for r in hot) and all(r[0] < 0.005 for r in cool),
            f"at temperature 1.0, {hot[0][0]:.1%} and {hot[1][0]:.1%} of steps are off-pattern "
            f"for styles 0 and 1, leaving {hot[0][1]:.0%} and {hot[1][1]:.0%} of 20-token "
            f"sequences clean; at 0.7 the rate is {cool[0][0]:.2%} and {cool[1][0]:.2%}. The "
            f"add-one prior leaves {result['prior_mass']:.1%} of each row on unseen tokens",
        ),
        practice.Check(
            "FINDING: the 'alternating' style 0 is a +1 ramp with a stutter",
            result["forward"] == 1.0 and 0.2 < result["shared"] < 0.3,
            f"{result['forward']:.0%} of style 0's {result['n_steps']} training steps go forward "
            "by 0, 1 or 2 -- none alternate, despite main()'s 'speech-like (alternating)' label. "
            f"Both styles are ramps, and the +2 step ({result['shared']:.0%} of style 0's) "
            "belongs to both, so no single step identifies the style",
        ),
        practice.Check(
            "FINDING: style=2 crashes but style=-1 silently aliases style 1",
            result["style2"] == "IndexError"
            and all(s in STEPS[1] for s in steps(alias)),
            f"generate(counts, 2, ...) raises {result['style2']}, while style=-1 returns "
            f"{alias} -- style 1's ramp, because counts[-1] is the last table",
        ),
        practice.Check(
            "CONTROL: the training data itself has no off-pattern steps",
            result["data_off"] == [0.0, 0.0],
            f"training sequences are {result['data_off'][0]:.1%} and {result['data_off'][1]:.1%} "
            "off-pattern, so every glitch above comes from the model's smoothing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
