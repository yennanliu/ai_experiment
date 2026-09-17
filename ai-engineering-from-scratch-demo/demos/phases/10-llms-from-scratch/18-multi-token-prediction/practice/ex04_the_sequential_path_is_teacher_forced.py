"""Exercise 4 — sequential wins 32 of 40 seeds with every module untrained, because it is fed the answer.

    Switch the toy to parallel MTP (Gloeckle-style): add D output heads on top of
    the main hidden state, each predicting a different offset. Measure how the
    losses per depth compare to the sequential version on the same synthetic
    signal. The sequential version should produce lower depth-k loss for k > 1
    because it conditions on the intermediate predictions.

Reading of the exercise: parallel MTP is built as the exercise describes -- `D`
independent output matrices over the *same* backbone hidden state, each scored
against a different offset -- and compared with the lesson's own `mtp_loss` on
the same hidden states and the same tokens. It is run on 40 seeds, because both
arms are untrained here and a single draw cannot separate a structural effect
from an initialisation.

**ANSWER: sequential's depth-2 loss is lower in 32 of 40 seeds, mean gap
+0.155 with a standard deviation of 0.191.** On seed 23 it is **3.374 against
3.844**. The exercise's prediction holds, four times in five, with every
parameter in both arms drawn from a random number generator and never trained.

**MECHANISM: the sequential path is fed the true next token, not its own
prediction.** `mtp_loss` advances with
`h_prev = mtp_forward(h_prev, E[tokens[i + k]], modules[k - 1])` -- `tokens[i+k]`
is the ground-truth token whose *loss was just computed*. So the depth-2
prediction is made after being shown the answer to depth 1. That is teacher
forcing, and it is why an untrained module beats an untrained head.

**FINDING: the exercise's stated reason is the one thing the toy does not do.**
"Because it conditions on the intermediate predictions" describes feeding depth 1
its own output; the code feeds it the label. At inference there is no label, so
the advantage measured here is not the advantage the deployed model has --
which is Lesson 15's exposure bias arriving in a different lesson.

**FINDING: depth 1 is identical in both schemes, and has to be.** Parallel MTP
with the tied head scores `shared_head_logits(h_i, E)` against `tokens[i+1]`,
which is exactly what `mtp_loss` does at `k=1`. The two schemes differ only from
depth 2 onward, so a D=1 model is both at once and the comparison the exercise
asks for needs D >= 2 to exist.

Structure: `parallel` scores `D` independent heads on the backbone hidden;
`trial` runs both schemes on one seed and returns the depth-2 gap.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "18-multi-token-prediction"
VOCAB, HIDDEN, FF, SEQ, DEPTHS, LAM = 32, 8, 16, 12, 2, 0.3
TRIALS, NOISE, LESSON_SEED = 40, 0.15, 23


def build(ref, seed):
    """Embeddings, tokens, modules and backbone hidden states for one seed."""
    rng = random.Random(seed)
    embeddings = ref.rand_matrix(VOCAB, HIDDEN, rng, scale=0.2)
    tokens = [rng.randrange(VOCAB) for _ in range(SEQ)]
    modules = [ref.make_mtp_module(HIDDEN, FF, rng) for _ in range(DEPTHS)]
    noise = random.Random(seed + 100)
    hidden = [ref.rms_norm(ref.add(embeddings[tokens[i]],
                                   [noise.gauss(0, NOISE) for _ in range(HIDDEN)]))
              for i in range(SEQ)]
    return embeddings, tokens, modules, hidden


def parallel(ref, hidden, tokens, heads):
    """D independent output matrices over the same backbone hidden, one per offset."""
    return [statistics.fmean(
        ref.cross_entropy(ref.shared_head_logits(hidden[i], heads[k - 1]), tokens[i + k])
        for i in range(len(hidden) - DEPTHS)) for k in range(1, DEPTHS + 1)]


def trial(ref, seed):
    embeddings, tokens, modules, hidden = build(ref, seed)
    heads = [ref.rand_matrix(VOCAB, HIDDEN, random.Random(300 + seed * 10 + k), scale=0.2)
             for k in range(DEPTHS)]
    sequential = ref.mtp_loss(hidden, tokens, modules, embeddings, LAM)[1]
    tied = parallel(ref, hidden, tokens, [embeddings] * DEPTHS)
    return {"sequential": sequential, "parallel": parallel(ref, hidden, tokens, heads),
            "tied": tied}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [trial(ref, seed) for seed in range(TRIALS)]
    gaps = [row["parallel"][1] - row["sequential"][1] for row in rows]
    return {
        "lesson": trial(ref, LESSON_SEED),
        "wins": sum(gap > 0 for gap in gaps),
        "trials": TRIALS,
        "gap_mean": statistics.fmean(gaps),
        "gap_sd": statistics.stdev(gaps),
        "depth_one_identical": all(abs(row["sequential"][0] - row["tied"][0]) < 1e-12
                                   for row in rows),
    }


def verify(result):
    lesson = result["lesson"]
    sequential, parallel_, tied = lesson["sequential"], lesson["parallel"], lesson["tied"]
    return [
        practice.Check(
            "ANSWER: sequential's depth-2 loss is lower in 32 of 40 seeds, mean gap +0.155",
            result["wins"] > 0.7 * result["trials"] and sequential[1] < parallel_[1],
            f"on seed {LESSON_SEED} the per-depth losses are "
            f"{sequential[0]:.3f}, {sequential[1]:.3f} sequential against "
            f"{parallel_[0]:.3f}, {parallel_[1]:.3f} parallel. Across {result['trials']} seeds "
            f"sequential wins at depth 2 in {result['wins']}, with a mean gap of "
            f"{result['gap_mean']:+.3f} and a standard deviation of {result['gap_sd']:.3f}. The "
            "exercise's prediction holds, with every parameter in both arms drawn from a random "
            "number generator and never trained",
        ),
        practice.Check(
            "MECHANISM: the sequential path is fed the true next token, not its own prediction",
            sequential[1] < parallel_[1],
            "mtp_loss advances with h_prev = mtp_forward(h_prev, E[tokens[i + k]], "
            "modules[k - 1]), and tokens[i+k] is the ground-truth token whose loss was just "
            "computed. So the depth-2 prediction is made after being shown the answer to depth 1 "
            "-- teacher forcing, which is why an untrained module beats an untrained head and why "
            "the margin does not need training to appear",
        ),
        practice.Check(
            "FINDING: the exercise's stated reason is the one thing the toy does not do",
            result["gap_mean"] > 0,
            "'because it conditions on the intermediate predictions' describes feeding depth 1 "
            "its own output; the code feeds it the label. At inference there is no label, so the "
            f"{result['gap_mean']:+.3f} measured here is not the advantage the deployed model "
            "has -- it is Lesson 15's exposure bias arriving in a different lesson, and with the "
            "sign that flatters the method",
        ),
        practice.Check(
            "FINDING: depth 1 is identical in both schemes, and has to be",
            result["depth_one_identical"],
            f"parallel MTP with the tied head scores shared_head_logits(h_i, E) against "
            f"tokens[i+1], which is exactly what mtp_loss does at k=1: {sequential[0]:.4f} "
            f"against {tied[0]:.4f} on seed {LESSON_SEED}, and identical to twelve decimals "
            f"on all {result['trials']}. The two schemes differ only from depth 2 onward, so a "
            "D=1 model is both at once and the comparison the exercise asks for needs D >= 2 to "
            "exist at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
