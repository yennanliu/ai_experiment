"""Exercise 1 — each method scores better under the other's metric than under its own.

    Implement KTO (Kahneman-Tversky Optimization). KTO doesn't need pairs --
    just label each response as "good" or "bad." The loss for a good response is
    `-log(sigmoid(beta * log_ratio))` and for a bad response is `-log(1 -
    sigmoid(beta * log_ratio))` with a loss aversion multiplier (typically 1.5x)
    on the bad response loss. Train on the same data (treat preferred as "good"
    and rejected as "bad" independently) and compare accuracy against DPO.

Reading of the exercise: both arms are trained, as the exercise says to -- DPO
through the lesson's own `dpo_train`, KTO through the same update rule with the
KTO loss deciding the direction -- from one initialisation and one RNG stream,
and both are then scored under *both* metrics, because "compare accuracy against
DPO" does not say which metric the comparison uses and the two disagree.

**FINDING: at the state both methods start from, every log-ratio is exactly
zero.** `copy_model_weights` makes the reference the policy, so
`pi_logprob - ref_logprob` is 0.0 for all 12 responses -- not approximately.
KTO's loss is fixed at `-log(0.5) + 1.5 * -log(0.5)` = **1.7329** per pair and
DPO's at **0.6931**, whatever the data says, and the untrained accuracies are
**0.500** for KTO and **0.000** for DPO: two different tie-breaks on the same
indifference.

**ANSWER: after training, both score 0.333 on their own metric -- and each
scores higher under the other's.**

    trained with   KTO metric   DPO metric
    DPO              0.417        0.333
    KTO              0.333        0.667

KTO training produces the best DPO accuracy in the table and the worst KTO
accuracy. "Compare accuracy against DPO" has four answers and they do not agree
on an ordering.

**MECHANISM: neither trainer uses a gradient.** `dpo_train`'s update is
`lr * (1.0 if logit < 0 else -0.1) * np.random.randn(...) * 0.01` -- a random
direction whose only tie to the loss is a sign, at `lr = 5e-6`. After five
epochs the largest weight has moved **8e-07** under DPO and **1e-06** under KTO,
so every accuracy above is the sign of noise, read on 6 pairs and quantised to
sixths.

**FINDING: `evaluate_preference_accuracy`'s floor is 0, not 0.5.** A model that
is exactly indifferent -- the state DPO is initialised into, by construction --
scores 0% rather than the 50% a preference metric should give it. The first
number the lesson's own training loop can print is the worst one the metric has.

Structure: `kto_loss` is the loss the exercise specifies; `kto_train` mirrors
`dpo_train`'s update rule with that loss deciding the direction; `accuracies`
scores one policy under both metrics.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "08-dpo"
SEED, BETA, AVERSION = 3, 0.1, 1.5
EPOCHS, LR = 5, 5e-6
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
    return (float(-np.log(value + 1e-8)) if is_good
            else float(AVERSION * -np.log(1 - value + 1e-8)))


def step(policy, direction):
    """`dpo_train`'s own update: a random direction scaled by the sign of the loss."""
    for block in policy.blocks:
        block.ffn.W1 += LR * direction * np.random.randn(*block.ffn.W1.shape) * 0.01
        block.ffn.W2 += LR * direction * np.random.randn(*block.ffn.W2.shape) * 0.01


def kto_train(ref, policy, reference, data):
    """That same update, with the KTO label deciding the direction."""
    for _ in range(EPOCHS):
        for index in np.random.permutation(len(data)):
            for key, good in (("preferred", True), ("rejected", False)):
                ratio = log_ratio(ref, policy, reference, data[index]["prompt"],
                                  data[index][key])
                step(policy, 1.0 if (ratio < 0) == good else -0.1)


def accuracies(ref, policy, reference):
    """One policy under both metrics: KTO asks each response's side of zero, DPO each pair's."""
    calls = [float((BETA * ratio > 0) == (key == "preferred"))
             for key, ratio in log_ratios(ref, policy, reference)]
    return (statistics.fmean(calls),
            ref.evaluate_preference_accuracy(policy, reference, ref.PREFERENCE_DATA, BETA))


def drift(policy, reference):
    return float(np.abs(policy.blocks[0].ffn.W1 - reference.blocks[0].ffn.W1).max())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policy, reference = pair_of_models(ref)
    ratios = log_ratios(ref, policy, reference)
    losses = [kto_loss(ref, ratio, key == "preferred") for key, ratio in ratios]
    dpo, _ = ref.dpo_loss(0.0, 0.0, 0.0, 0.0, BETA)
    dpo_policy, dpo_reference = pair_of_models(ref)
    np.random.seed(SEED + 7)
    with contextlib.redirect_stdout(io.StringIO()):
        dpo_policy, _, _ = ref.dpo_train(dpo_policy, dpo_reference, ref.PREFERENCE_DATA,
                                         num_epochs=EPOCHS, beta=BETA)
    kto_policy, kto_reference = pair_of_models(ref)
    np.random.seed(SEED + 7)
    kto_train(ref, kto_policy, kto_reference, ref.PREFERENCE_DATA)
    return {"max_ratio": max(abs(ratio) for _, ratio in ratios), "responses": len(ratios),
            "kto_good": losses[0], "kto_bad": losses[1], "kto_total": losses[0] + losses[1],
            "dpo_loss": float(dpo),
            "untrained": accuracies(ref, policy, reference),
            "after_dpo": accuracies(ref, dpo_policy, dpo_reference),
            "after_kto": accuracies(ref, kto_policy, kto_reference),
            "drift": (drift(dpo_policy, dpo_reference), drift(kto_policy, kto_reference))}


def verify(result):
    kto, dpo = result["untrained"]
    dpo_kto, dpo_dpo = result["after_dpo"]
    kto_kto, kto_dpo = result["after_kto"]
    dpo_drift, kto_drift = result["drift"]
    return [
        practice.Check(
            "FINDING: both methods start from an exact tie, and break it differently",
            result["max_ratio"] == 0.0 and kto == 0.5 and dpo == 0.0,
            f"copy_model_weights makes the reference the policy, so pi_logprob - ref_logprob is "
            f"0.0 for all {result['responses']} responses -- exactly. KTO's loss is fixed at "
            f"{result['kto_total']:.4f} per pair and DPO's at {result['dpo_loss']:.4f}, whatever "
            f"the data says. Untrained, KTO scores {kto:.3f} -- the base rate of a constant 'bad' "
            f"classifier, since 0 > 0 is false -- and DPO {dpo:.3f}, the floor of a strict "
            "inequality on two equal numbers",
        ),
        practice.Check(
            "ANSWER: each method scores better under the other's metric than under its own",
            dpo_kto > dpo_dpo and kto_dpo > kto_kto,
            f"after {EPOCHS} epochs the DPO-trained policy scores {dpo_dpo:.3f} on DPO's own "
            f"metric and {dpo_kto:.3f} on KTO's; the KTO-trained policy scores {kto_kto:.3f} on "
            f"KTO's and {kto_dpo:.3f} on DPO's -- KTO training produces the best DPO accuracy in "
            "the table and the worst KTO accuracy. 'Compare accuracy against DPO' has four "
            "answers that do not agree on an ordering",
        ),
        practice.Check(
            "MECHANISM: neither trainer uses a gradient, so every number is the sign of noise",
            dpo_drift < 1e-5 and kto_drift < 1e-5,
            f"dpo_train's update is lr * (1.0 if logit < 0 else -0.1) * np.random.randn(...) * "
            f"0.01 -- a random direction whose only tie to the loss is a sign -- and the KTO arm "
            f"mirrors it. After {EPOCHS} epochs at lr = {LR} the largest weight has moved "
            f"{dpo_drift:.1e} under DPO and {kto_drift:.1e} under KTO, against weights of scale "
            f"0.02, so each accuracy above is the sign of noise on "
            f"{result['responses'] // 2} pairs",
        ),
        practice.Check(
            "FINDING: evaluate_preference_accuracy's floor is 0, not 0.5",
            dpo == 0.0,
            "a model that is exactly indifferent -- the state DPO is initialised into, since "
            "the reference is a copy of the policy -- scores 0% where a preference metric should "
            "give it 50%. The first number the lesson's training loop prints is the metric's "
            "worst",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
