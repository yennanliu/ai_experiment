"""Exercise 4 — self-play flips the one relation the model can see, and accuracy stays 0.000.

    Implement iterative DPO. Run DPO for 3 epochs, then generate new responses
    from the trained model, pair them with the original preferred responses as
    new preference pairs, and run DPO again. Two rounds of this "self-play"
    process. Compare preference accuracy after round 1 and round 2 to see if
    iterative refinement helps.

Reading of the exercise: both rounds use the lesson's own `dpo_train`, and
round 2's pairs are built exactly as described -- the original preferred
response against a fresh sample from the round-1 policy. Accuracy is the
reference's own `evaluate_preference_accuracy`, reported on the original pairs
so the two rounds are scored against the same question.

**ANSWER: 0.000 after round 1 and 0.000 after round 2.** Not one of the six
pairs changes sign. The exercise asks whether iterative refinement helps and the
measured answer is that nothing happens at all -- the three findings below
account for it.

**FINDING: self-play inverts the length relation the data was built on.** The
original rejected responses average **105 bytes** -- they are the padded ones --
while the round-1 policy samples **28 bytes** at 30 new tokens. So round 2 trains
on pairs where the preferred response is the *longer* one, which is the opposite
of every pair in the original set. Exercise 2 shows that response length is the
only property this model's log-probabilities track, so self-play does not add
signal, it reverses it.

**FINDING: the generated responses are not responses.** They are 30 bytes
sampled from a model at random initialisation -- Exercise 5 shows the policy
moves 1e-6 across a whole DPO run -- so the "new preference pairs" are
`(human-written answer, random bytes)`. Two rounds of this is one round of DPO
followed by one round of DPO against noise.

**FINDING: 0.000 is the metric's floor, not a measurement.** The reference is a
copy of the policy, so every implicit margin is exactly 0 and
`preferred_reward > rejected_reward` is false for all six pairs (Exercise 1) --
before either round and after both. Two rounds move the largest weight 1.7e-06,
nowhere near enough to break a tie resolved by a strict inequality.

Structure: `sample` draws a response from a policy the way the lesson's own
generation does; `round_two` builds the self-play pairs the exercise specifies.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "08-dpo"
SEED, BETA, EPOCHS, NEW_TOKENS = 3, 0.1, 3, 30
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=128, ff_dim=256)


def models(ref):
    np.random.seed(SEED)
    policy = ref.MiniGPT(**SHAPE)
    np.random.seed(SEED + 96)
    reference = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, reference)
    return policy, reference


def train(ref, policy, reference, pairs, seed):
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        trained, _, _ = ref.dpo_train(policy, reference, pairs, num_epochs=EPOCHS, beta=BETA)
    return trained


def sample(ref, policy, prompt):
    """One response drawn from the policy, the way the lesson's generation loop does."""
    tokens = ref.tokenize_sequence(prompt)
    start = len(tokens)
    for _ in range(NEW_TOKENS):
        logits = policy.forward(np.array(tokens[-SHAPE["max_seq_len"]:]).reshape(1, -1))[0, -1, :]
        probs = np.exp(logits - logits.max())
        tokens.append(int(np.random.choice(len(probs), p=probs / probs.sum())))
    return bytes(tokens[start:]).decode("utf-8", "replace")


def round_two(ref, policy, seed):
    """The self-play pairs: the original preferred response against a fresh sample."""
    np.random.seed(seed)
    return [{"prompt": pair["prompt"], "preferred": pair["preferred"],
             "rejected": sample(ref, policy, pair["prompt"])}
            for pair in ref.PREFERENCE_DATA]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policy, reference = models(ref)
    first = train(ref, policy, reference, ref.PREFERENCE_DATA, SEED + 2)
    accuracy_one = ref.evaluate_preference_accuracy(first, reference, ref.PREFERENCE_DATA, BETA)
    pairs = round_two(ref, first, SEED + 14)
    second = train(ref, first, reference, pairs, SEED + 2)
    return {
        "accuracy": (accuracy_one,
                     ref.evaluate_preference_accuracy(second, reference,
                                                      ref.PREFERENCE_DATA, BETA)),
        "pairs": len(ref.PREFERENCE_DATA),
        "original_rejected": statistics.fmean(len(p["rejected"]) for p in ref.PREFERENCE_DATA),
        "selfplay_rejected": statistics.fmean(len(p["rejected"]) for p in pairs),
        "preferred": statistics.fmean(len(p["preferred"]) for p in ref.PREFERENCE_DATA),
        "original_shorter": sum(len(p["preferred"]) < len(p["rejected"])
                                for p in ref.PREFERENCE_DATA),
        "selfplay_shorter": sum(len(p["preferred"]) < len(p["rejected"]) for p in pairs),
        "drift": float(np.abs(second.blocks[0].ffn.W1 - reference.blocks[0].ffn.W1).max()),
    }


def verify(result):
    one, two = result["accuracy"]
    pairs = result["pairs"]
    return [
        practice.Check(
            f"ANSWER: {one:.3f} after round 1 and {two:.3f} after round 2 -- no pair moves",
            two == one == 0.0,
            f"not one of the {pairs} pairs changes sign between the rounds. The exercise asks "
            "whether iterative refinement helps and the measured answer is that nothing "
            "happens at all, which the next three findings account for",
        ),
        practice.Check(
            "FINDING: self-play inverts the length relation the data was built on",
            result["selfplay_shorter"] < result["original_shorter"]
            and result["selfplay_rejected"] < result["preferred"],
            f"the original rejected responses average {result['original_rejected']:.0f} bytes -- "
            f"they are the padded ones, and preferred is shorter in "
            f"{result['original_shorter']} of {pairs}. The round-1 policy samples "
            f"{result['selfplay_rejected']:.0f} bytes at {NEW_TOKENS} new tokens, so round 2 "
            f"trains on pairs where preferred is shorter in only {result['selfplay_shorter']}. "
            "Exercise 2 shows length is the only property this model's log-probabilities track, "
            "so self-play does not add signal -- it reverses it",
        ),
        practice.Check(
            "FINDING: the generated responses are not responses",
            result["drift"] < 1e-5,
            f"they are {NEW_TOKENS} bytes sampled from a model that has moved "
            f"{result['drift']:.1e} across two whole DPO runs. The new preference pairs are "
            "(human-written answer, random bytes), so two rounds of this is one round of DPO "
            "followed by one round of DPO against noise",
        ),
        practice.Check(
            "FINDING: 0.000 is the metric's floor, not a measurement",
            one == two == 0.0,
            "the reference is a copy of the policy, so every implicit margin is exactly 0 and "
            f"preferred_reward > rejected_reward is false for all {pairs} pairs -- before either "
            f"round and after both. Two rounds of DPO move the largest weight "
            f"{result['drift']:.1e}, nowhere near enough to break a tie the metric resolves with "
            "a strict inequality",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
