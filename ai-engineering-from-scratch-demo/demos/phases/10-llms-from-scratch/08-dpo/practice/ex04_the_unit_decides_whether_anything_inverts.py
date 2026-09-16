"""Exercise 4 — the unit decides whether self-play inverts anything, and accuracy stays 0.000.

    Implement iterative DPO. Run DPO for 3 epochs, then generate new responses
    from the trained model, pair them with the original preferred responses as
    new preference pairs, and run DPO again. Two rounds of this "self-play"
    process. Compare preference accuracy after round 1 and round 2 to see if
    iterative refinement helps.

Reading of the exercise: both rounds use the lesson's own `dpo_train`, and
round 2's pairs are built exactly as described -- the original preferred
response against a fresh sample from the round-1 policy. Accuracy is the
reference's own `evaluate_preference_accuracy`, reported on the original pairs
so the two rounds are scored against the same question, and every length is
counted with `tokenize_sequence`, which is the unit DPO scores in.

**ANSWER: 0.000 after round 1 and 0.000 after round 2.** Not one of the six
pairs changes sign. The exercise asks whether iterative refinement helps and the
measured answer is that nothing happens at all -- the findings below account for
it.

**FINDING: self-play does not invert the length relation.** The original
rejected responses average **105 tokens** -- they are the padded ones -- and the
round-1 policy samples **56**, still above preferred's 50.3. Preferred is the
shorter response in **4 of 6** pairs before self-play and in **4 of 6** after.
The one property Exercise 2 shows these log-probabilities track is not reversed
by round 2; it is halved and left pointing the same way.

**FINDING: counted in characters the same data says 2 of 6.** `sample` decodes
the drawn bytes with `errors="replace"`, and 11 to 15 of each sample's ~29
characters are U+FFFD. `len(str)` counts each of those once; `tokenize_sequence`
re-encodes each as three bytes. So the self-play responses are 28.5 characters
and 56.3 tokens, and the two units disagree about the direction of the
comparison in two of the six pairs. The unit, not the self-play, produces the
inversion.

**FINDING: the generated responses are not responses.** They are 30 tokens
sampled from a model at random initialisation -- the policy moves 1.7e-06 across
two whole DPO runs -- so the "new preference pairs" are `(human-written answer,
undecodable bytes)`. Two rounds of this is one round of DPO followed by one
round of DPO against noise.

**FINDING: 0.000 is the metric's floor, not a measurement.** The reference is a
copy of the policy, so every implicit margin is exactly 0 and
`preferred_reward > rejected_reward` is false for all six pairs (Exercise 1) --
before either round and after both. Two rounds move the largest weight 1.7e-06,
nowhere near enough to break a tie resolved by a strict inequality.

Structure: `sample` draws a response from a policy the way the lesson's own
generation does; `round_two` builds the self-play pairs the exercise specifies;
`length` counts in the tokenizer's units and `len` is kept beside it for the
comparison.
"""

from __future__ import annotations

import contextlib
import functools
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


def length(ref, text):
    """Length in the units DPO scores: the lesson's own tokens, not Python characters."""
    return len(ref.tokenize_sequence(text))


def tally(ref, pairs, measure):
    """Mean preferred length, mean rejected length, and how often preferred is shorter."""
    return (statistics.fmean(measure(p["preferred"]) for p in pairs),
            statistics.fmean(measure(p["rejected"]) for p in pairs),
            sum(measure(p["preferred"]) < measure(p["rejected"]) for p in pairs))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policy, reference = models(ref)
    first = train(ref, policy, reference, ref.PREFERENCE_DATA, SEED + 2)
    accuracy_one = ref.evaluate_preference_accuracy(first, reference, ref.PREFERENCE_DATA, BETA)
    pairs = round_two(ref, first, SEED + 14)
    second = train(ref, first, reference, pairs, SEED + 2)
    tokens = functools.partial(length, ref)
    return {
        "accuracy": (accuracy_one,
                     ref.evaluate_preference_accuracy(second, reference,
                                                      ref.PREFERENCE_DATA, BETA)),
        "pairs": len(ref.PREFERENCE_DATA),
        "original": tally(ref, ref.PREFERENCE_DATA, tokens),
        "selfplay": tally(ref, pairs, tokens),
        "selfplay_chars": tally(ref, pairs, len),
        "replacements": [pair["rejected"].count("\ufffd") for pair in pairs],
        "drift": float(np.abs(second.blocks[0].ffn.W1 - reference.blocks[0].ffn.W1).max()),
    }


def verify(result):
    one, two = result["accuracy"]
    pairs = result["pairs"]
    was_pref, was_rej, was_shorter = result["original"]
    now_pref, now_rej, now_shorter = result["selfplay"]
    _, char_rej, char_shorter = result["selfplay_chars"]
    marks = result["replacements"]
    return [
        practice.Check(
            f"ANSWER: {one:.3f} after round 1 and {two:.3f} after round 2 -- no pair moves",
            two == one == 0.0,
            f"not one of the {pairs} pairs changes sign between the rounds. The exercise asks "
            "whether iterative refinement helps and the measured answer is that nothing "
            "happens at all, which the next three findings account for",
        ),
        practice.Check(
            "FINDING: self-play does not invert the length relation -- 4 of 6 either way",
            now_shorter == was_shorter and now_rej > now_pref,
            f"the original rejected responses average {was_rej:.0f} tokens -- they are the padded "
            f"ones -- and the round-1 policy samples {now_rej:.0f}, still above preferred's "
            f"{now_pref:.1f}. Preferred is the shorter response in {was_shorter} of {pairs} pairs "
            f"before self-play and {now_shorter} of {pairs} after. The one property Exercise 2 shows "
            "these log-probabilities track is halved by round 2 and left pointing the same way",
        ),
        practice.Check(
            "FINDING: counted in characters the same data says 2 of 6 -- the unit inverts it",
            char_shorter < now_shorter and min(marks) > 5,
            f"sample decodes the drawn bytes with errors='replace', and {min(marks)} to "
            f"{max(marks)} of each sample's characters are U+FFFD. len(str) counts each of those "
            f"once and tokenize_sequence re-encodes each as three bytes, so the self-play "
            f"responses are {char_rej:.1f} characters and {now_rej:.1f} tokens, and the two units "
            f"disagree about the direction of the comparison in {now_shorter - char_shorter} of "
            f"the {pairs} pairs. The unit, not the self-play, produces the inversion",
        ),
        practice.Check(
            "FINDING: the generated responses are not responses",
            result["drift"] < 1e-5,
            f"they are {NEW_TOKENS} tokens sampled from a model that has moved "
            f"{result['drift']:.1e} across two whole DPO runs. The new preference pairs are "
            "(human-written answer, undecodable bytes), so two rounds is one round of DPO "
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
