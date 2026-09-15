"""Exercise 5 — the most stable reference is the one whose margins are identically zero.

    Compare DPO with different reference models. Instead of using the SFT
    checkpoint as the reference, try: (a) the base model (pre-SFT), (b) a
    checkpoint from epoch 1 of DPO, (c) an exponential moving average of the
    policy model. Report which reference produces the highest preference accuracy
    and the most stable training curve.

Reading of the exercise: all three references score the *same* policy -- a model
trained by the lesson's own `dpo_train` -- so the comparison isolates the
reference and nothing else. "Most stable training curve" is read as the spread
of the implicit reward margins across the six pairs, since a curve whose points
all coincide is as stable as a curve can be and that turns out to be the
interesting case.

**ANSWER: the SFT checkpoint and the EMA tie for the most stable curve, at a
margin spread below 1e-5 -- because their margins are all essentially zero.**
`copy_model_weights` makes the SFT reference a copy of the policy, and an EMA
initialised from the policy that then tracks it stays a copy to within the
policy's own 1e-6 of drift. A reference that equals the policy gives
`pi_logprob - ref_logprob = 0` for every response, so every margin is 0 and the
curve is a flat line at zero.

**FINDING: "most stable" and "most informative" point in opposite directions
here.** The base model -- a different random initialisation -- gives margins with
a spread of **0.1343**, four orders of magnitude larger than the other two. It is
the least stable arm and the only one whose margins carry any information about
the pairs at all.

**ANSWER on accuracy: all three are at or below one pair in six.** The SFT copy
scores **0.000**, the base model **0.167**, the EMA **0.000**. The differences
are not learning -- they are whether a margin of 1e-7 lands on the right side of
`preferred_reward > rejected_reward`, a strict inequality on two nearly equal
numbers. The exercise asks which reference gives the highest accuracy, and none
of them gives an accuracy.

**FINDING: option (b) cannot differ from option (a) here.** "A checkpoint from
epoch 1 of DPO" is the policy after one epoch, and the policy moves 1e-6 across
a whole run, so that checkpoint is the SFT checkpoint to six decimal places. Two
of the exercise's three options are the same model.

Structure: `margins` returns the six implicit reward margins against a given
reference; `ema` builds option (c) by copying the policy, which is what an
exponential moving average initialised from it computes.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "08-dpo"
SEED, BETA, EPOCHS = 3, 0.1, 3
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


def margins(ref, policy, reference):
    """The six implicit reward margins `dpo_loss` reports against this reference."""
    out = []
    for pair in ref.PREFERENCE_DATA:
        _, metrics = ref.dpo_loss(logprob(ref, policy, pair, "preferred"),
                                  logprob(ref, policy, pair, "rejected"),
                                  logprob(ref, reference, pair, "preferred"),
                                  logprob(ref, reference, pair, "rejected"), BETA)
        out.append(metrics["reward_margin"])
    return out


def ema(ref, policy):
    """Option (c): an EMA initialised from the policy is a copy of it."""
    np.random.seed(SEED + 400)
    shadow = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, shadow)
    return shadow


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policy, sft = models(ref)
    np.random.seed(SEED + 2)
    with contextlib.redirect_stdout(io.StringIO()):
        trained, _, _ = ref.dpo_train(policy, sft, ref.PREFERENCE_DATA,
                                      num_epochs=EPOCHS, beta=BETA)
    np.random.seed(SEED + 39)
    arms = {"sft": sft, "base": ref.MiniGPT(**SHAPE), "ema": ema(ref, trained)}
    return {
        "arms": {name: {
            "accuracy": ref.evaluate_preference_accuracy(trained, model,
                                                         ref.PREFERENCE_DATA, BETA),
            "mean": statistics.fmean(margins(ref, trained, model)),
            "spread": statistics.pstdev(margins(ref, trained, model)),
        } for name, model in arms.items()},
        "drift": float(np.abs(trained.blocks[0].ffn.W1 - sft.blocks[0].ffn.W1).max()),
        "pairs": len(ref.PREFERENCE_DATA),
    }


def verify(result):
    arms = result["arms"]
    sft, base, shadow = arms["sft"], arms["base"], arms["ema"]
    return [
        practice.Check(
            "ANSWER: the SFT copy and the EMA tie for most stable, at a spread under 1e-5",
            max(sft["spread"], shadow["spread"]) < 1e-5,
            f"the SFT reference's margins have a spread of {sft['spread']:.2e} and the EMA's "
            f"{shadow['spread']:.2e}. copy_model_weights makes the SFT reference a copy of the "
            f"policy, and an EMA initialised from the policy stays one to within the "
            f"{result['drift']:.1e} the policy itself moves -- so pi_logprob - ref_logprob is 0 "
            "for every response and the curve is a flat line at zero",
        ),
        practice.Check(
            "FINDING: most stable and most informative point in opposite directions",
            base["spread"] > 1000 * max(sft["spread"], shadow["spread"]),
            f"the base model -- a different random initialisation -- gives margins with a spread "
            f"of {base['spread']:.4f} and a mean of {base['mean']:+.4f}, against "
            f"{sft['spread']:.4f} and {sft['mean']:+.4f} for the SFT copy. It is the least "
            "stable arm and the only one whose margins carry any information about the pairs. "
            "The exercise asks for the most stable curve, and the most stable curve is the one "
            "with nothing in it",
        ),
        practice.Check(
            "ANSWER on accuracy: all three are at or below one pair in six",
            max(a["accuracy"] for a in arms.values()) <= 1 / result["pairs"],
            f"the SFT copy scores {sft['accuracy']:.3f}, the base model {base['accuracy']:.3f} "
            f"and the EMA {shadow['accuracy']:.3f} over {result['pairs']} pairs -- at most one "
            "pair right, by any reference. The differences between them are not learning: they "
            "are whether a margin of 1e-7 lands on the right side of "
            "preferred_reward > rejected_reward, a strict inequality on two nearly equal "
            "numbers. The exercise asks which reference gives the highest accuracy, and none "
            "of them gives an accuracy",
        ),
        practice.Check(
            "FINDING: option (b) is option (a) to six decimal places",
            result["drift"] < 1e-5,
            f"'a checkpoint from epoch 1 of DPO' is the policy after one epoch, and the policy "
            f"moves {result['drift']:.1e} across all {EPOCHS}. That checkpoint and the SFT "
            "checkpoint are the same weights to six decimal places, so two of the exercise's "
            "three options name one model",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
