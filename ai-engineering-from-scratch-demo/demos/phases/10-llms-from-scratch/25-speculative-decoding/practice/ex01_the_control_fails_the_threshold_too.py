"""Exercise 1 — plain target sampling scores 0.0175 against its own distribution, so the 0.01 bar fails the control.

    Implement the exact rejection rule and empirically verify it. Run 10K
    samples via `speculative_decode` and via plain target sampling; compute TV
    distance between the two output distributions. Should be < 0.01.

Reading of the exercise: the rejection rule is already exact in
`speculative_step`, and the lesson's own `verify_distribution` already runs both
arms, so the work is in the threshold. Plain sampling is scored against the
target it was drawn from -- the control -- and a deliberately biased scheme is
run beside it, because a test that cannot separate correct from incorrect is not
a verification whatever number it reports.

**ANSWER: both arms fail the 0.01 bar, and the speculative one scores lower than
the control.**

    vocab    plain vs target    speculative vs target    biased (accept everything)
      32         0.0175               0.0163                     0.1201
     128         0.0371               0.0360                        --
     512         0.0737               0.0722                        --

**MECHANISM: the number is a sampling floor, not a bias.** With 10,000 draws
over a 32-token vocabulary, an empirical histogram sits about **0.018** in total
variation from the distribution it was drawn from, by construction. Raising the
vocabulary to 512 raises the floor to **0.074**, because the same 10,000 samples
are spread over 16x as many bins. The threshold the exercise names is below the
noise of the measurement it names.

**FINDING: the test does discriminate -- 7x above the floor.** Accepting every
draft token without the rejection rule scores **0.1201** against the target,
against a floor of 0.0175. So the right bar is "indistinguishable from the
control", which both correct arms pass at a margin of 0.001, and not a fixed
constant.

**FINDING: the speculative arm beats plain sampling, which is also noise.** It
scores 0.0012 *lower* at vocab 32, and lower at 128 and 512 too. Two unbiased
estimators of the same distribution differ by their draws; the sign of that
difference carries nothing, and a threshold test on it would report whichever
arm happened to be luckier.

Structure: `arm` runs the lesson's own `verify_distribution` at one vocabulary;
`biased` samples straight from the draft, which is what the rejection rule exists
to correct.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "25-speculative-decoding"
SAMPLES, DRAFT_K, HINT, THRESHOLD = 10_000, 4, 0.7, 0.01
VOCABULARIES = (32, 128, 512)


def pair(ref, vocab, seed=0):
    rng = np.random.default_rng(seed)
    target = ref.make_target(vocab, rng)
    return target, ref.make_draft(target, HINT, rng), rng


def arm(ref, vocab):
    """The lesson's own two-arm check at one vocabulary."""
    target, draft, rng = pair(ref, vocab)
    plain_tv, spec_tv = ref.verify_distribution(target, draft, DRAFT_K, SAMPLES, rng)
    return {"plain": plain_tv, "spec": spec_tv, "gap": spec_tv - plain_tv}


def biased(ref, vocab):
    """Accept every draft token: the scheme the rejection rule exists to correct."""
    target, draft, rng = pair(ref, vocab, seed=0)
    samples = [ref.sample(draft, rng) for _ in range(SAMPLES)]
    return ref.total_variation(ref.empirical_dist(samples, vocab), target)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {vocab: arm(ref, vocab) for vocab in VOCABULARIES}
    return {
        "rows": rows,
        "biased": biased(ref, VOCABULARIES[0]),
        "floor_growth": rows[512]["plain"] / rows[32]["plain"],
        "margin": rows[32]["gap"],
        "discrimination": biased(ref, VOCABULARIES[0]) / rows[32]["plain"],
        "threshold": THRESHOLD,
    }


def column(rows, field, fmt):
    return ", ".join(f"vocab {vocab} {format(row[field], fmt)}"
                     for vocab, row in rows.items())


def verify(result):
    rows = result["rows"]
    small = rows[32]
    return [
        practice.Check(
            "ANSWER: both arms fail the 0.01 bar, and the speculative one scores lower",
            small["plain"] > THRESHOLD < small["spec"] and small["spec"] < small["plain"],
            "plain target sampling scores " + column(rows, "plain", ".4f")
            + " against the distribution it was drawn from, and the speculative arm "
            + column(rows, "spec", ".4f")
            + f". Both are above the {THRESHOLD} the exercise names, at every vocabulary, and "
            f"the speculative arm is {abs(small['gap']):.4f} *below* the control",
        ),
        practice.Check(
            "MECHANISM: the number is a sampling floor, not a bias",
            result["floor_growth"] > 3,
            f"with {SAMPLES:,} draws over a 32-token vocabulary an empirical histogram sits about "
            f"{small['plain']:.3f} in total variation from the distribution it came from, by "
            f"construction. Raising the vocabulary to 512 raises that to {rows[512]['plain']:.3f} "
            f"-- {result['floor_growth']:.1f}x -- because the same {SAMPLES:,} samples are spread "
            "over 16x as many bins. The threshold is below the noise of the measurement",
        ),
        practice.Check(
            "FINDING: the test does discriminate -- 7x above the floor",
            result["discrimination"] > 5,
            f"accepting every draft token without the rejection rule scores "
            f"{result['biased']:.4f} against the target, {result['discrimination']:.1f}x the "
            f"{small['plain']:.4f} floor. So the check is sound and its bar is not: the right one "
            "is 'indistinguishable from the control', which both correct arms pass at a margin of "
            f"{abs(result['margin']):.4f}",
        ),
        practice.Check(
            "FINDING: the speculative arm beating plain sampling is also noise",
            all(row["gap"] < 0 for row in rows.values()) and abs(small["gap"]) < 0.01,
            "the speculative arm scores lower at " + column(rows, "gap", "+.4f")
            + " -- every vocabulary. Two unbiased estimators of the same distribution differ by "
            "their draws, so the sign of that difference carries nothing, and a threshold test on "
            "it would report whichever arm happened to be luckier on the seed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
