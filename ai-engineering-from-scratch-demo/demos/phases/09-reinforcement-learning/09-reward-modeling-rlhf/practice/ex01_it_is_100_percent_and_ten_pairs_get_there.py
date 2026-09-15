"""Exercise 1 — it is 100%, and ten pairs get there.

    **Easy.** Train the Bradley-Terry reward model in `code/main.py` on 500
    synthetic preference pairs. Measure pairwise accuracy on a held-out 100 pairs.
    Should exceed 90%.

Reading of the exercise: the stated bar invites a yes, so the useful answer is how
far past it the result sits and why. `train_rm` and `rm_accuracy` are the lesson's
own; 20 seeds are run because a single held-out sample of 100 cannot distinguish
94% from 100%, and the training size is swept down to find where the bar is
actually crossed.

**ANSWER: 1.0000, on every one of 20 seeds.** Not "exceeds 90%" -- it does not miss
a single held-out pair, at any seed.

**FINDING: ten pairs get there, and two clear the stated bar.** Accuracy by training
size: 0.79 at 1 pair, **0.95 at 2**, 0.99 at 3, 1.00 at 10 and everything above. The
exercise asks for 500, which is 50x more than the task needs.

**MECHANISM: `GOOD` and `BAD` share no token.** `sample_pair` draws the preferred
response from `GOOD²` and the rejected one from `BAD²`, and the two vocabularies are
disjoint, so a bag-of-words scorer separates them as soon as it has seen each token
once -- 12 tokens, one gradient step each. There is no ambiguous example anywhere in
the distribution for the model to be wrong about.

**FINDING: the held-out set is not held out from anything that matters.** It is drawn
from the same generator, and while 3,888 distinct (prompt, preferred, rejected)
triples exist, the model only has 12 parameters and the labelling rule is a partition
of them. Train and test differ in sample, not in structure.

**FINDING: every `GOOD` weight ends positive and every `BAD` weight negative.** The
learned separation is total, and the margin is what accuracy 1.0 is measuring.

Structure: `curve` sweeps the training size through the lesson's own `train_rm`;
`sign_split` reads the learned weights back against the two vocabularies.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "09-reward-modeling-rlhf"
PAIRS, HOLDOUT, SEEDS = 500, 100, 20
SIZES = (1, 2, 3, 5, 10, 20, 50)


def accuracy(ref, n_pairs, seed):
    """Train on `n_pairs` and score the lesson's own held-out measurement."""
    model = ref.train_rm(n_pairs=n_pairs, rng=random.Random(seed))
    return ref.rm_accuracy(model, n_pairs=HOLDOUT, rng=random.Random(1000 + seed))


def curve(ref):
    """Mean held-out accuracy at each training size, over the seeds."""
    return {n: statistics.fmean(accuracy(ref, n, s) for s in range(SEEDS)) for n in SIZES}


def sign_split(ref, model):
    """(good tokens with positive weight, bad tokens with negative weight)."""
    return (sum(1 for t in ref.GOOD if model.get(t, 0.0) > 0),
            sum(1 for t in ref.BAD if model.get(t, 0.0) < 0))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    full = [accuracy(ref, PAIRS, s) for s in range(SEEDS)]
    model = ref.train_rm(n_pairs=PAIRS, rng=random.Random(0))
    return {"full": full, "curve": curve(ref), "split": sign_split(ref, model),
            "overlap": sorted(set(ref.GOOD) & set(ref.BAD)),
            "vocab": len(ref.VOCAB), "good": len(ref.GOOD), "bad": len(ref.BAD),
            "triples": len(ref.GOOD) ** 2 * len(ref.BAD) ** 2 * len(ref.PROMPTS)}


def verify(result):
    full, rows = result["full"], result["curve"]
    good, bad = result["split"]
    enough = next(n for n in SIZES if rows[n] >= 1.0)
    clears = next(n for n in SIZES if rows[n] > 0.90)
    return [
        practice.Check(
            "ANSWER: 1.0000, on every one of 20 seeds",
            min(full) == 1.0,
            f"training on {PAIRS} pairs and scoring the lesson's own held-out {HOLDOUT}, "
            f"accuracy is {statistics.fmean(full):.4f} with a minimum of {min(full):.4f} across "
            f"{SEEDS} seeds. The exercise's bar is 90%; the model does not miss a single "
            "held-out pair at any seed, so 'should exceed 90%' understates it by every margin "
            "the measurement can express",
        ),
        practice.Check(
            "FINDING: ten pairs get there, and two clear the stated bar",
            enough <= 10 and clears <= 2,
            "mean held-out accuracy by training size -- "
            + ", ".join(f"{n}: {rows[n]:.4f}" for n in SIZES)
            + f". {clears} pairs already exceed the 90% the exercise sets, and {enough} reach "
            f"1.0000. The {PAIRS} the exercise specifies is {PAIRS // enough}x more than the "
            "task requires",
        ),
        practice.Check(
            "MECHANISM: GOOD and BAD share no token",
            result["overlap"] == [],
            f"`sample_pair` draws the preferred response from GOOD^2 and the rejected one from "
            f"BAD^2, and the two vocabularies intersect in {result['overlap']} -- nothing. A "
            f"bag-of-words scorer separates them as soon as it has seen each of the "
            f"{result['vocab']} tokens once, one gradient step each. No example in the "
            "distribution is ambiguous, so there is nothing for the model to be wrong about",
        ),
        practice.Check(
            "FINDING: the held-out set is not held out from anything that matters",
            result["triples"] > 1000 and result["vocab"] < 20,
            f"it is drawn from the same generator as the training data. {result['triples']:,} "
            f"distinct (prompt, preferred, rejected) triples exist, but the model carries "
            f"{result['vocab']} parameters and the labelling rule is a partition of them into "
            f"{result['good']} and {result['bad']}. Train and test differ in sample, not in "
            "structure, so the holdout measures memorisation of nothing",
        ),
        practice.Check(
            "FINDING: every GOOD weight ends positive and every BAD weight negative",
            (good, bad) == (result["good"], result["bad"]),
            f"{good} of {result['good']} GOOD tokens carry positive weight and {bad} of "
            f"{result['bad']} BAD tokens carry negative weight after {PAIRS} pairs. The "
            "separation is total rather than merely accurate, which is why no amount of extra "
            "held-out data would find an error: the sign pattern already decides every pair the "
            "generator can produce",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
