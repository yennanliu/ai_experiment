"""Exercise 3 — at alpha=1.0 the DPO term is 11.4% of the loss, and it is a constant.

    Build an ORPO-style combined loss. Add a standard next-token prediction loss
    on the preferred response to the DPO loss: `L = L_sft(preferred) + alpha *
    L_dpo`. Try alpha values of 0.1, 0.5, and 1.0. The combined loss should
    produce a model that both follows instructions (from the SFT term) and
    prefers better responses (from the DPO term), eliminating the need for a
    separate SFT stage.

Reading of the exercise: `L_sft(preferred)` is the mean next-token
cross-entropy over the preferred response -- the per-token form, which is what
the lesson's own pre-training loss returns and the only form that is comparable
to `L_dpo` across responses of different lengths. Both terms come from the
reference's own `compute_sequence_log_prob` and `dpo_loss`.

**ANSWER: 1.27%, 6.06% and 11.43%.** The DPO term's share of the combined loss
at alpha = 0.1, 0.5 and 1.0. Even at the largest alpha the exercise names, the
combined objective is **88.6% SFT**.

**MECHANISM: the two terms are not on the same scale.** `L_sft` is 5.3734 nats
per token against `ln(256) = 5.5452` for a uniform model -- the SFT term starts
near its ceiling and has the whole range below it to move through. `L_dpo` is
bounded by its own construction: `-log(sigmoid(beta * margin))` with beta = 0.1
sits at 0.6931 whenever the margin is small, and the margin is small by default.

**FINDING: the DPO term's *value* is constant, and its gradient is not zero.**
The reference is a copy of the policy, so every log-ratio is exactly zero
(Exercise 1) and `L_dpo` is **0.6931** for all six pairs at every alpha. That is
one value repeated, not a flat function: differentiating `-log(sigmoid(beta *
m))` at `m = 0` through the lesson's own `dpo_loss` gives **-0.0500**, which is
`-beta/2`, so the term does supply a preference gradient and alpha does scale
it. What makes it inert here is `dpo_train`, which never computes that gradient
-- its update is `lr * (1.0 if logit < 0 else -0.1) * np.random.randn(...)`. The
preference half of "eliminating the need for a separate SFT stage" is live on
paper and unused in the loop.

**FINDING: alpha cannot buy the DPO term influence.** Reaching an even split
needs alpha = **7.8**, nearly 8x the exercise's largest value -- and at that
point the term it is weighting is still 0.6931 for every pair. The knob the
exercise offers moves the ratio between the two terms and not the amount of
information in either.

Structure: `sft_loss` is the per-token cross-entropy on the preferred response;
`dpo_term` is the reference's own DPO loss on one pair; `margin_slope` reads its
derivative at margin zero.
"""

from __future__ import annotations

import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "08-dpo"
SEED, BETA = 3, 0.1
ALPHAS = (0.1, 0.5, 1.0)
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=128, ff_dim=256)


def models(ref):
    np.random.seed(SEED)
    policy = ref.MiniGPT(**SHAPE)
    np.random.seed(SEED + 96)
    reference = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, reference)
    return policy, reference


def logprob(ref, model, pair, key):
    return float(ref.compute_sequence_log_prob(
        model, ref.tokenize_sequence(pair["prompt"]), ref.tokenize_sequence(pair[key])))


def sft_loss(ref, policy, pair):
    """Mean next-token cross-entropy over the preferred response, in nats per token."""
    tokens = ref.tokenize_sequence(pair["preferred"])
    return -logprob(ref, policy, pair, "preferred") / len(tokens)


def dpo_term(ref, policy, reference, pair):
    loss, metrics = ref.dpo_loss(logprob(ref, policy, pair, "preferred"),
                                 logprob(ref, policy, pair, "rejected"),
                                 logprob(ref, reference, pair, "preferred"),
                                 logprob(ref, reference, pair, "rejected"), BETA)
    return float(loss), metrics["reward_margin"]


def margin_slope(ref, step=1e-6):
    """d/dmargin of the lesson's own dpo_loss at margin 0, by central difference."""
    up, _ = ref.dpo_loss(step, 0.0, 0.0, 0.0, BETA)
    down, _ = ref.dpo_loss(-step, 0.0, 0.0, 0.0, BETA)
    return float((up - down) / (2 * step))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policy, reference = models(ref)
    sfts = [sft_loss(ref, policy, pair) for pair in ref.PREFERENCE_DATA]
    dpos = [dpo_term(ref, policy, reference, pair) for pair in ref.PREFERENCE_DATA]
    sft, dpo = statistics.fmean(sfts), statistics.fmean(loss for loss, _ in dpos)
    return {
        "sft": sft,
        "dpo": dpo,
        "uniform": float(np.log(256)),
        "share": {a: a * dpo / (sft + a * dpo) for a in ALPHAS},
        "constant": len({round(loss, 12) for loss, _ in dpos}) == 1,
        "margins": [margin for _, margin in dpos],
        "even_split": sft / dpo,
        "slope": margin_slope(ref),
    }


def verify(result):
    share, sft, dpo = result["share"], result["sft"], result["dpo"]
    return [
        practice.Check(
            "ANSWER: the DPO term is 1.27%, 6.06% and 11.43% of the combined loss",
            max(share.values()) < 0.2,
            "at alpha = "
            + ", ".join(f"{a} it is {100 * s:.2f}%" for a, s in share.items())
            + f". Even at the largest alpha the exercise names, the combined objective is "
            f"{100 * (1 - share[1.0]):.1f}% SFT -- the term the exercise treats as the addition "
            "is the term that dominates",
        ),
        practice.Check(
            "MECHANISM: the two terms are not on the same scale",
            sft > 7 * dpo and sft < result["uniform"],
            f"L_sft is {sft:.4f} nats per token against ln(256) = {result['uniform']:.4f} for a "
            f"uniform model, so it starts near its ceiling with the whole range below it to move "
            f"through. L_dpo is bounded by its own construction: -log(sigmoid(beta * margin)) "
            f"with beta = {BETA} sits at {dpo:.4f} whenever the margin is small, and the margin "
            f"is small by default. The two differ by {sft / dpo:.1f}x before alpha is applied",
        ),
        practice.Check(
            "FINDING: the DPO term's value is constant, and its gradient is not zero",
            result["constant"] and all(m == 0.0 for m in result["margins"])
            and abs(result["slope"] + BETA / 2) < 1e-6,
            f"the reference is a copy of the policy, so every implicit margin is "
            f"{result['margins'][0]:.4f} for all six pairs and L_dpo is {dpo:.4f} everywhere, at "
            f"every alpha. That is one value repeated, not a flat function: differentiating the "
            f"lesson's own dpo_loss at margin 0 gives {result['slope']:.4f}, which is -beta/2, so "
            f"the term does supply a preference gradient and alpha does scale it. What makes it "
            f"inert here is dpo_train, which never computes that gradient -- its update is "
            "lr * (1.0 if logit < 0 else -0.1) * np.random.randn(...). The preference half of "
            "'eliminating the need for a separate SFT stage' is live on paper and unused in the "
            "loop",
        ),
        practice.Check(
            "FINDING: alpha cannot buy the DPO term influence -- an even split needs 7.8",
            result["even_split"] > 5 * max(ALPHAS),
            f"reaching a 50/50 split needs alpha = {result['even_split']:.1f}, "
            f"{result['even_split'] / max(ALPHAS):.0f}x the exercise's largest value -- and at "
            f"that point the term it is weighting is still {dpo:.4f} for every pair. The knob "
            "the exercise offers moves the ratio between the two terms and not the amount of "
            "information in either",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
