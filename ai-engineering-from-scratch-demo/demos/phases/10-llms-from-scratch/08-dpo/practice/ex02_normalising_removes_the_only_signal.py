"""Exercise 2 — length normalisation removes the bias and the accuracy with it, 4 of 6 to 1 of 6.

    Implement length-normalized DPO. Instead of raw log-probabilities, divide by
    the number of response tokens: `normalized_logprob = total_logprob /
    num_tokens`. This prevents the model from favoring shorter responses (which
    have higher total log-prob). Compare the implicit reward margins with and
    without normalization.

Reading of the exercise: "the implicit reward margins" are the ones
`dpo_loss` returns, so both arms are built from the reference's own
`compute_sequence_log_prob` with only the denominator changed. The margins
themselves are identically zero while the reference is a copy of the policy
(Exercise 1), so the comparison is made on the *policy log-probabilities* that
feed them, which is where the length effect lives and where normalisation acts.

**ANSWER: the premise is right and the fix costs accuracy.** Total log-probability
correlates **-0.987** with response length across the twelve responses -- longer
is lower, almost deterministically -- so ranking by raw total picks the shorter
response essentially every time. On the lesson's own data that is 4 of 6 correct,
because preferred is shorter in 4 of 6 pairs. Divide by token count and it falls
to **1 of 6**.

**MECHANISM: per-token log-probability is nearly constant, so normalising leaves
noise.** At this model the twelve responses span **-5.5443 to -4.3577** nats per
token -- a 1.19-nat window on a scale where `ln(256) = 5.5452` is the uniform
baseline. Dividing by length removes the one quantity that varies a lot and
leaves the one that barely varies at all.

**FINDING: on this dataset the length bias *is* the signal.** The exercise
describes favouring shorter responses as a defect to be prevented. Here it is
67% accurate and the corrected version is 17%. The bias is wrong in general and
right on a dataset built by padding the rejected response.

**FINDING: neither number is a property of DPO.** Both arms score a model whose
reference is itself, so the implicit margins they are supposed to compare are
0.0000 either way -- the ranking comes entirely from the policy term, which is
what a *reference-free* method would use. What the exercise measures is a length
heuristic, twice.

Structure: `logprobs` is the reference's own scorer; `ranking` counts preferences
under a chosen denominator.
"""

from __future__ import annotations

import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "08-dpo"
SEED, BETA = 3, 0.1
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=128, ff_dim=256)


def models(ref):
    np.random.seed(SEED)
    policy = ref.MiniGPT(**SHAPE)
    np.random.seed(SEED + 96)
    reference = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, reference)
    return policy, reference


def scored(ref, model, pair, key):
    """(total log-probability, response token count) for one response."""
    tokens = ref.tokenize_sequence(pair[key])
    total = ref.compute_sequence_log_prob(model, ref.tokenize_sequence(pair["prompt"]), tokens)
    return float(total), len(tokens)


def ranking(ref, model, normalise):
    """How many preferences the log-probability ranking reproduces, raw or per token."""
    correct = 0
    for pair in ref.PREFERENCE_DATA:
        good, good_n = scored(ref, model, pair, "preferred")
        bad, bad_n = scored(ref, model, pair, "rejected")
        if normalise:
            correct += good / good_n > bad / bad_n
        else:
            correct += good > bad
    return correct


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policy, reference = models(ref)
    rows = [scored(ref, policy, pair, key)
            for pair in ref.PREFERENCE_DATA for key in ("preferred", "rejected")]
    margins = []
    for pair in ref.PREFERENCE_DATA:
        good, _ = scored(ref, policy, pair, "preferred")
        bad, _ = scored(ref, policy, pair, "rejected")
        ref_good, _ = scored(ref, reference, pair, "preferred")
        ref_bad, _ = scored(ref, reference, pair, "rejected")
        _, metrics = ref.dpo_loss(good, bad, ref_good, ref_bad, BETA)
        margins.append(metrics["reward_margin"])
    per_token = [total / count for total, count in rows]
    return {
        "pairs": len(ref.PREFERENCE_DATA),
        "raw": ranking(ref, policy, False),
        "normalised": ranking(ref, policy, True),
        "correlation": statistics.correlation([count for _, count in rows],
                                              [total for total, _ in rows]),
        "per_token": (min(per_token), max(per_token)),
        "uniform": float(np.log(256)),
        "shorter": sum(len(ref.tokenize_sequence(p["preferred"]))
                       < len(ref.tokenize_sequence(p["rejected"]))
                       for p in ref.PREFERENCE_DATA),
        "margins": margins,
    }


def verify(result):
    raw, normalised, pairs = result["raw"], result["normalised"], result["pairs"]
    low, high = result["per_token"]
    return [
        practice.Check(
            f"ANSWER: raw ranks {raw} of {pairs} right, normalised {normalised}",
            raw > normalised and result["correlation"] < -0.9,
            f"total log-probability correlates {result['correlation']:.3f} with response length "
            f"across the twelve responses -- longer is lower, almost deterministically -- so "
            f"ranking by raw total picks the shorter response nearly every time. That is {raw} "
            f"of {pairs} here, because preferred is shorter in {result['shorter']} of {pairs} "
            f"pairs. Dividing by token count takes it to {normalised}",
        ),
        practice.Check(
            "MECHANISM: per-token log-probability barely varies, so normalising leaves noise",
            high - low < 1.5 and low > -result["uniform"] - 0.1,
            f"the twelve responses span {low:.4f} to {high:.4f} nats per token -- a "
            f"{high - low:.2f}-nat window on a scale where ln(256) = {result['uniform']:.4f} is "
            "the uniform baseline. Dividing by length removes the one quantity that varies a "
            "lot and leaves the one that hardly varies at all",
        ),
        practice.Check(
            "FINDING: on this dataset the length bias is the signal",
            result["shorter"] == raw > pairs / 2 > normalised,
            f"the exercise describes favouring shorter responses as a defect to prevent. Here it "
            f"is {raw / pairs:.0%} accurate and the corrected version is {normalised / pairs:.0%}. "
            "The bias is wrong in general and right on a dataset built by padding the rejected "
            f"response -- the {raw} pairs raw ranking gets right are exactly the "
            f"{result['shorter']} where preferred is shorter",
        ),
        practice.Check(
            "FINDING: neither number is a property of DPO -- the implicit margins are 0.0000",
            all(margin == 0.0 for margin in result["margins"]),
            f"both arms score a model whose reference is itself, so the implicit reward margins "
            f"the exercise asks to compare are {result['margins'][0]:.4f} for every pair, either "
            "way. The ranking comes entirely from the policy term, which is what a "
            "reference-free method would use. What the exercise measures here is a length "
            "heuristic, twice",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
