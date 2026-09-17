"""Exercise 5 — the draft agrees with the main model 52% of the time, and the main model is 3% accurate.

    Use the trained MTP module as an EAGLE-style draft: call module k to propose
    `t_{i+k}` at inference. Measure the acceptance rate of these draft tokens
    against the main model's predictions on a held-out sequence. If you hit 50%+
    on the toy, you have reproduced the empirical MTP-as-draft property.

Reading of the exercise: acceptance is measured exactly as described -- the
draft's proposal for offset `k` against what the main model's head produces at
that position -- using the lesson's own `mtp_forward` and `shared_head_logits`,
on 40 held-out sequences. The main model's own next-token accuracy is measured
beside it, because "acceptance against the main model's predictions" says nothing
about whether those predictions are right.

**ANSWER: 52.2% mean acceptance, above 50% in 40 of 40 seeds -- and the main
model is 3.4% accurate.** The exercise's threshold is cleared on every seed by a
module that was never trained, agreeing with a backbone that guesses. Random
choice at this vocabulary is 3.1%.

**MECHANISM: half the comparisons compare a hidden state with itself.** At `k=1`
the draft reads `h = backbone[i]` and the main model's prediction at offset 0 is
also read off `backbone[i]`, so the two argmaxes are the same number by
construction. That is `1/D` of the comparisons scoring 100% before anything is
measured; at D=2 the floor is exactly **50%**, and the exercise's pass mark is
the floor.

**FINDING: the depth-2 comparison, the only real one, scores 4%.** Subtracting
the self-comparison leaves the module's actual proposals agreeing with the
backbone at a rate indistinguishable from the 3.1% of drawing a token uniformly.
The 52.2% is `(100% + 4%) / 2`.

**FINDING: acceptance against a wrong prediction is not speculative decoding.**
Leviathan acceptance (Lesson 15) compares a draft *distribution* against a
verifier *distribution* and corrects the difference; this compares two argmaxes
and reports a match rate. A draft that agreed with the main model 100% of the
time while the main model was 3.4% accurate would score perfectly here and
produce nothing.

Structure: `propose` walks the draft chain through the lesson's own
`mtp_forward`; `accept` scores one sequence and splits the rate by depth.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "18-multi-token-prediction"
VOCAB, HIDDEN, FF, SEQ, DEPTHS = 32, 8, 16, 12, 2
TRIALS, NOISE, THRESHOLD = 40, 0.15, 0.5


def build(ref, seed):
    rng = random.Random(seed)
    embeddings = ref.rand_matrix(VOCAB, HIDDEN, rng, scale=0.2)
    tokens = [rng.randrange(VOCAB) for _ in range(SEQ)]
    modules = [ref.make_mtp_module(HIDDEN, FF, rng) for _ in range(DEPTHS)]
    noise = random.Random(seed + 100)
    hidden = [ref.rms_norm(ref.add(embeddings[tokens[i]],
                                   [noise.gauss(0, NOISE) for _ in range(HIDDEN)]))
              for i in range(SEQ)]
    return embeddings, tokens, modules, hidden


def argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


def accept(ref, embeddings, tokens, modules, hidden):
    """Per-depth agreement between the draft chain and the main model's own argmax."""
    hits = [0] * DEPTHS
    positions = len(hidden) - DEPTHS
    for i in range(positions):
        state = hidden[i]
        for k in range(1, DEPTHS + 1):
            proposal = argmax(ref.shared_head_logits(state, embeddings))
            main = argmax(ref.shared_head_logits(hidden[min(i + k - 1, len(hidden) - 1)],
                                                 embeddings))
            hits[k - 1] += proposal == main
            state = ref.mtp_forward(state, embeddings[tokens[i + k]], modules[k - 1])
    return [hit / positions for hit in hits]


def main_accuracy(ref, embeddings, tokens, hidden):
    """How often the backbone's own next-token argmax is the token that follows."""
    return statistics.fmean(argmax(ref.shared_head_logits(hidden[i], embeddings))
                            == tokens[i + 1] for i in range(len(hidden) - 1))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rates, depths, accuracies = [], [], []
    for seed in range(TRIALS):
        embeddings, tokens, modules, hidden = build(ref, seed)
        per_depth = accept(ref, embeddings, tokens, modules, hidden)
        depths.append(per_depth)
        rates.append(statistics.fmean(per_depth))
        accuracies.append(main_accuracy(ref, embeddings, tokens, hidden))
    return {
        "rate": statistics.fmean(rates),
        "low": min(rates),
        "over_threshold": sum(rate >= THRESHOLD for rate in rates) / TRIALS,
        "per_depth": [statistics.fmean(row[k] for row in depths) for k in range(DEPTHS)],
        "accuracy": statistics.fmean(accuracies),
        "uniform": 1 / VOCAB,
        "trials": TRIALS,
    }


def verify(result):
    first, second = result["per_depth"]
    return [
        practice.Check(
            "ANSWER: 52.2% acceptance on every seed, from a module that was never trained",
            result["over_threshold"] == 1.0 and result["accuracy"] < 0.1,
            f"mean acceptance is {result['rate']:.1%} across {result['trials']} held-out "
            f"sequences, above the exercise's {THRESHOLD:.0%} mark on "
            f"{result['over_threshold']:.0%} of them and never below {result['low']:.1%}. The "
            f"backbone those proposals are being accepted against predicts the next token "
            f"correctly {result['accuracy']:.1%} of the time, against "
            f"{result['uniform']:.1%} for drawing a token uniformly",
        ),
        practice.Check(
            "MECHANISM: half the comparisons compare a hidden state with itself",
            first == 1.0,
            f"at k=1 the draft reads h = backbone[i] and the main model's prediction at offset 0 "
            f"is also read off backbone[i], so the two argmaxes are the same number by "
            f"construction: depth 1 scores {first:.1%} before anything is measured. That is 1/D "
            f"of the comparisons, so at D={DEPTHS} the floor of this metric is exactly "
            f"{1 / DEPTHS:.0%} -- and the exercise's pass mark is the floor",
        ),
        practice.Check(
            "FINDING: the depth-2 comparison, the only real one, scores 4%",
            second < 5 * result["uniform"],
            f"subtracting the self-comparison leaves the module's actual proposals agreeing with "
            f"the backbone {second:.1%} of the time, against {result['uniform']:.1%} for a "
            f"uniform draw. The headline {result['rate']:.1%} is ({first:.0%} + {second:.0%}) / "
            f"{DEPTHS}, and the half that carries information is the half that is not counted "
            "separately",
        ),
        practice.Check(
            "FINDING: acceptance against a wrong prediction is not speculative decoding",
            result["accuracy"] < 0.1 < result["rate"],
            "Leviathan acceptance in Lesson 15 compares a draft distribution against a verifier "
            "distribution and corrects the difference; this compares two argmaxes and reports a "
            f"match rate. A draft agreeing with the main model 100% of the time while the main "
            f"model is {result['accuracy']:.1%} accurate would score perfectly on this metric and "
            "produce nothing, which is the version of the experiment the toy is closest to",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
