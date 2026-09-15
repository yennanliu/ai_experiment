"""Exercise 1 — KTO beats DPO 0.500 to 0.000 before either has learned anything.

    Implement KTO (Kahneman-Tversky Optimization). KTO doesn't need pairs --
    just label each response as "good" or "bad." The loss for a good response is
    `-log(sigmoid(beta * log_ratio))` and for a bad response is `-log(1 -
    sigmoid(beta * log_ratio))` with a loss aversion multiplier (typically 1.5x)
    on the bad response loss. Train on the same data (treat preferred as "good"
    and rejected as "bad" independently) and compare accuracy against DPO.

Reading of the exercise: both losses are computed on the same model and the same
reference, at the state DPO actually starts from -- `copy_model_weights` makes
the reference a copy of the policy, which is what the lesson's own demo does.
"Compare accuracy against DPO" uses the reference's own
`evaluate_preference_accuracy` for the DPO arm and the natural KTO analogue --
is `beta * log_ratio > 0` -- for the other.

**FINDING: every log-ratio is exactly zero, so both losses are constants.** The
reference is the policy, so `pi_logprob - ref_logprob` is 0.0 for all 12
responses, not approximately. KTO's loss is then fixed at
`-log(0.5) + 1.5 * -log(0.5)` = **1.7329** per pair, and DPO's at **0.6931**,
whatever the data says.

**ANSWER: KTO scores 0.500 and DPO scores 0.000.** KTO calls a response good
when `beta * ratio > 0`, which `0 > 0` makes false, so it labels all 12 bad --
getting every rejected response right and every preferred one wrong, for exactly
half. DPO's accuracy compares `preferred_reward > rejected_reward`, which is
`0 > 0`, false for every pair, for zero.

**FINDING: the comparison is decided by how each metric breaks a tie.** Neither
method has seen a gradient. KTO's 0.500 is the base rate of a constant "bad"
classifier and DPO's 0.000 is the floor of a strict inequality on two equal
numbers. The exercise asks which method is more accurate and the answer is which
one's tie-break is luckier.

**FINDING: `evaluate_preference_accuracy`'s floor is 0, not 0.5.** A model that
is exactly indifferent -- the state DPO is initialised into, by construction --
scores 0% rather than the 50% a preference metric should give it. The first
number the lesson's own training loop can print is the worst one the metric has.

Structure: `kto_loss` is the loss the exercise specifies; `log_ratios` returns
the twelve policy-minus-reference values both methods are built on.
"""

from __future__ import annotations

import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "08-dpo"
SEED, BETA, AVERSION = 3, 0.1, 1.5
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=128, ff_dim=256)


def pair_of_models(ref):
    """A policy and the reference `copy_model_weights` makes of it."""
    np.random.seed(SEED)
    policy = ref.MiniGPT(**SHAPE)
    np.random.seed(SEED + 96)
    reference = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, reference)
    return policy, reference


def log_ratio(ref, policy, reference, prompt, text):
    prompt_tokens = ref.tokenize_sequence(prompt)
    response_tokens = ref.tokenize_sequence(text)
    return (ref.compute_sequence_log_prob(policy, prompt_tokens, response_tokens)
            - ref.compute_sequence_log_prob(reference, prompt_tokens, response_tokens))


def log_ratios(ref, policy, reference):
    """The twelve policy-minus-reference log-probabilities, preferred then rejected."""
    return [(key, log_ratio(ref, policy, reference, pair["prompt"], pair[key]))
            for pair in ref.PREFERENCE_DATA for key in ("preferred", "rejected")]


def kto_loss(ref, ratio, is_good):
    """The loss the exercise specifies, with loss aversion on the bad side."""
    value = ref.sigmoid(BETA * ratio)
    if is_good:
        return float(-np.log(value + 1e-8))
    return float(AVERSION * -np.log(1 - value + 1e-8))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policy, reference = pair_of_models(ref)
    ratios = log_ratios(ref, policy, reference)
    losses = [kto_loss(ref, ratio, key == "preferred") for key, ratio in ratios]
    good_calls = [BETA * ratio > 0 for _, ratio in ratios]
    dpo, _ = ref.dpo_loss(0.0, 0.0, 0.0, 0.0, BETA)
    return {
        "max_ratio": max(abs(ratio) for _, ratio in ratios),
        "responses": len(ratios),
        "kto_good": losses[0],
        "kto_bad": losses[1],
        "kto_total": losses[0] + losses[1],
        "dpo_loss": float(dpo),
        "called_good": sum(good_calls),
        "kto_accuracy": statistics.fmean(
            float(call == (key == "preferred")) for call, (key, _) in zip(good_calls, ratios)),
        "dpo_accuracy": ref.evaluate_preference_accuracy(policy, reference,
                                                         ref.PREFERENCE_DATA, BETA),
    }


def verify(result):
    kto, dpo = result["kto_accuracy"], result["dpo_accuracy"]
    return [
        practice.Check(
            "FINDING: every log-ratio is exactly zero, so both losses are constants",
            result["max_ratio"] == 0.0,
            f"copy_model_weights makes the reference the policy, so pi_logprob - ref_logprob is "
            f"0.0 for all {result['responses']} responses -- not approximately, exactly. KTO's "
            f"loss is then fixed at {result['kto_good']:.4f} + {AVERSION} * "
            f"{result['kto_bad'] / AVERSION:.4f} = {result['kto_total']:.4f} per pair and DPO's "
            f"at {result['dpo_loss']:.4f}, whatever the data says",
        ),
        practice.Check(
            "ANSWER: KTO scores 0.500 and DPO scores 0.000",
            kto == 0.5 and dpo == 0.0,
            f"KTO calls a response good when beta * ratio > 0, and 0 > 0 is false, so it labels "
            f"all {result['responses']} bad: every rejected response right and every preferred "
            f"one wrong, for exactly {kto:.3f}. DPO compares preferred_reward > rejected_reward, "
            f"which is 0 > 0, false for every pair, for {dpo:.3f}",
        ),
        practice.Check(
            "FINDING: the comparison is decided by how each metric breaks a tie",
            result["called_good"] == 0 and kto > dpo,
            f"neither method has seen a gradient yet. KTO's {kto:.3f} is the base rate of a "
            f"constant 'bad' classifier on a balanced set, and DPO's {dpo:.3f} is the floor of a "
            "strict inequality on two equal numbers. The exercise asks which method is more "
            "accurate; the answer at this state is which one's tie-break is luckier",
        ),
        practice.Check(
            "FINDING: evaluate_preference_accuracy's floor is 0, not 0.5",
            dpo == 0.0,
            "a model that is exactly indifferent -- the state DPO is initialised into, by "
            "construction, since the reference is a copy of the policy -- scores 0% where a "
            "preference metric should give it 50%. The first number the lesson's own training "
            "loop can print is the worst one the metric is capable of",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
