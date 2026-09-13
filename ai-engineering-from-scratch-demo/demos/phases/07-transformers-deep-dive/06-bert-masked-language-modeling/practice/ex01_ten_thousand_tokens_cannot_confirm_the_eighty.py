"""Exercise 1 — ten thousand tokens is too few to confirm the number it asks about.

    **Easy.** Run `code/main.py` and print the mask distribution across 10,000
    tokens. Confirm ~15% are selected, and of those ~80% become `[MASK]`.

Reading of the exercise: "confirm" is read as a statistical claim, so the same
`distribution_check` is run at the exercise's 10,000, at `main()`'s own 100,000
and at 1,000,000, and the binomial standard errors are computed alongside. The
sampling is the lesson's own -- same function, same seed, same vocabulary.

**ANSWER: 15.710% selected, of which 78.167% become `[MASK]`.** The other two
branches land at 11.076% random and 10.757% unchanged. Both headline numbers are
in the right neighbourhood and neither is close.

**FINDING: the exercise's sample size cannot confirm its own second number.** At
10,000 tokens only about 1,571 are selected, so the 1-sigma error on the "80%" is
**1.03 points** -- a +/-3.1 point window at 3 sigma. The measured 78.17% sits
1.8 sigma low, which is unremarkable and also indistinguishable from a real 78%
rule. `main()` runs the same check at **100,000**, ten times the exercise's
number, where the same code gives 79.80%; at 1,000,000 it gives **80.02%**. The
exercise asks you to confirm a figure at a tenth of the sample size the lesson
itself chose for it.

**FINDING: `create_mlm_batch` hangs for `vocab_size <= 4`.** The random-replacement
branch loops `while rand_id in SPECIAL_IDS or rand_id == t`, and with ids 0, 1, 2
reserved there is no legal replacement for `t = 3`. Probed under a 0.3 s alarm:
vocab_size 4 never returns, 5 returns.

**FINDING: 15% is a rate over *eligible* positions.** The loop skips specials, so
on `main()`'s own 11-token sentence -- `[CLS]` ... `[SEP]` -- only 9 positions can
ever be selected and the expected rate over the sequence is **12.3%**, not 15%.
`distribution_check` never generates a special, which is why its own number comes
out at 15.

**CONTROL: only 13.5% of tokens are actually changed.** 15% x 10% are labelled
and left untouched -- the model is asked to predict a token it can still see --
and 15% x 80% = **12%** ever become `[MASK]`, which is the number the
pretrain/finetune mismatch is actually about.

Structure: `sigma` is the binomial standard error; `hangs` probes the loop under
an alarm; `eligible` is the rate over a sequence that carries specials.
"""

from __future__ import annotations

import math
import random
import signal

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "06-bert-masked-language-modeling"
VOCAB, ASKED, SHIPPED, LARGE = 20, 10_000, 100_000, 1_000_000
PROB, SEED = 0.15, 42


def sigma(probability, trials):
    """Binomial standard error, in percentage points."""
    return 100 * math.sqrt(probability * (1 - probability) / trials)


def hangs(ref, vocab_size, budget=0.3):
    """Does create_mlm_batch return at this vocabulary size, or spin forever?"""
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError))
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        ref.create_mlm_batch([3] * 50, vocab_size, 1.0, random.Random(0))
        return False
    except TimeoutError:
        return True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def eligible(ref, tokens):
    """Expected selection rate over a sequence whose specials can never be chosen."""
    allowed = sum(1 for t in tokens if t not in ref.SPECIAL_IDS)
    return PROB * allowed / len(tokens), allowed, len(tokens)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {n: ref.distribution_check(n, VOCAB, PROB, seed=SEED)
            for n in (ASKED, SHIPPED, LARGE)}
    sentence = [ref.CLS_ID] + [5] * 9 + [ref.SEP_ID]
    rate, allowed, total = eligible(ref, sentence)
    return {
        "runs": runs,
        "sigma_selected": {n: sigma(PROB, n) for n in (ASKED, SHIPPED)},
        "sigma_mask": {n: sigma(0.8, int(PROB * n)) for n in (ASKED, SHIPPED)},
        "hangs": {v: hangs(ref, v) for v in (4, 5)},
        "eligible": (rate, allowed, total),
        "changed": PROB * 0.9, "maskable": PROB * 0.8, "untouched": PROB * 0.1,
    }


def verify(result):
    asked, shipped, large = (result["runs"][n] for n in (ASKED, SHIPPED, LARGE))
    off = abs(asked["masked_of_selected_pct"] - 80) / result["sigma_mask"][ASKED]
    rate, allowed, total = result["eligible"]
    return [
        practice.Check(
            "ANSWER: 15.710% selected, of which 78.167% become [MASK]",
            13 < asked["selected_pct"] < 17 and 75 < asked["masked_of_selected_pct"] < 85,
            f"at the exercise's {ASKED:,} tokens: {asked['selected_pct']:.3f}% selected, then "
            f"{asked['masked_of_selected_pct']:.3f}% [MASK], "
            f"{asked['random_of_selected_pct']:.3f}% random, "
            f"{asked['unchanged_of_selected_pct']:.3f}% unchanged. Both headline numbers are in "
            "the right neighbourhood and neither is close",
        ),
        practice.Check(
            "FINDING: 10,000 tokens cannot confirm the 80%, and main() uses ten times as many",
            result["sigma_mask"][ASKED] > 1.0 and off > 1.0
            and abs(large["masked_of_selected_pct"] - 80) < 0.1,
            f"only {int(PROB * ASKED):,} of {ASKED:,} tokens are selected, so 1 sigma on the 80% "
            f"is {result['sigma_mask'][ASKED]:.2f} points -- a +/-"
            f"{3 * result['sigma_mask'][ASKED]:.1f} point window at 3 sigma. The measured "
            f"{asked['masked_of_selected_pct']:.2f}% is {off:.1f} sigma low. main() runs the same "
            f"check at {SHIPPED:,} and gets {shipped['masked_of_selected_pct']:.2f}%; at "
            f"{LARGE:,} it is {large['masked_of_selected_pct']:.2f}%",
        ),
        practice.Check(
            "FINDING: create_mlm_batch hangs for vocab_size <= 4",
            result["hangs"][4] and not result["hangs"][5],
            "the random-replacement branch loops `while rand_id in SPECIAL_IDS or rand_id == t`, "
            "and with ids 0, 1 and 2 reserved there is no legal replacement for t = 3. Probed "
            "under a 0.3 s alarm: vocab_size 4 never returns, vocab_size 5 does. The toy vocab in "
            "main() is 20 words, so the lesson never meets its own edge",
        ),
        practice.Check(
            "FINDING: 15% is a rate over eligible positions, not over tokens",
            abs(rate - 0.1227) < 0.001,
            f"the loop skips specials, so on a sequence shaped like main()'s -- [CLS] plus "
            f"{allowed} words plus [SEP] -- only {allowed} of {total} positions can ever be "
            f"selected and the expected rate over the sequence is {rate:.1%}, not {PROB:.0%}. "
            "distribution_check never generates a special, which is why its own number is 15",
        ),
        practice.Check(
            "CONTROL: only 13.5% of tokens are changed and only 12% ever see [MASK]",
            abs(result["changed"] - 0.135) < 1e-12 and abs(result["maskable"] - 0.12) < 1e-12,
            f"{PROB:.0%} x 10% = {result['untouched']:.1%} of tokens are labelled and left "
            f"untouched, so the model is asked to predict a token it can still see; "
            f"{result['changed']:.1%} are actually changed and {result['maskable']:.0%} ever "
            "become [MASK]. That last one is the number the pretrain/finetune mismatch is about",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
