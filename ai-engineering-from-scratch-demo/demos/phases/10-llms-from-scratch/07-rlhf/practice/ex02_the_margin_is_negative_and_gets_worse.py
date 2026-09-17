"""Exercise 2 — the margin is -0.0383 on training data and -0.1106 on held-out.

    Implement reward model calibration. After training, run all preference pairs
    through the reward model and compute: (a) the average reward for preferred
    responses, (b) the average reward for rejected responses, (c) the margin
    (preferred minus rejected). A well-calibrated model should have a clear
    margin. Then add 4 new preference pairs and check if the margin holds on
    unseen data.

Reading of the exercise: the four new pairs are written in the shape of the
lesson's own six -- a factual question, a short correct answer preferred, a
padded correct answer rejected -- so the held-out set tests generalisation and
not a change of task. All three quantities are computed with the reference's own
`RewardModel.forward` and `tokenize_for_reward`.

**ANSWER: (a) +0.1752, (b) +0.2135, (c) -0.0383.** The margin is negative: the
model scores the *rejected* response higher on average, on the six pairs it was
trained on, and it gets 2 of 6 right.

**ANSWER to the second half: the margin holds, in the sense that it stays
negative.** On four unseen pairs it is **-0.1106** -- nearly three times worse --
with 1 of 4 correct. The exercise asks whether a clear margin survives unseen
data; what survives is the sign, and it is the wrong sign.

**MECHANISM: the reward tracks length, and the rejected responses are the long
ones.** Reward correlates **+0.268** with response length across the twelve
training responses, and the lesson's dataset is built so that the rejected
response is the padded one -- longer in 4 of the 6 pairs. A model that has
learned nothing but "longer scores higher" gets this dataset backwards by
construction.

**FINDING: "a well-calibrated model should have a clear margin" is not a
calibration test.** Calibration is whether a score maps to a probability -- a
0.7 margin meaning the preferred wins 70% of the time. A margin is a
*discrimination* statistic, and this model's is negative, so there is nothing to
calibrate yet. The exercise's own metric would pass a model that ranked every
pair correctly by 1e-9 and fail one that ranked them correctly by 1e9 with
inconsistent confidence.

Structure: `margins` returns the three quantities the exercise asks for on any
pair list; `HELD_OUT` is the four new pairs.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "07-rlhf"
SEED, EPOCHS, MAX_LEN = 1, 10, 128
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=MAX_LEN, ff_dim=256)
HELD_OUT = [
    {"prompt": "What is the boiling point of water?",
     "preferred": "Water boils at 100 degrees Celsius at sea level.",
     "rejected": "Water is a liquid. When you heat it enough it turns to steam, which "
                 "happens around one hundred degrees."},
    {"prompt": "Who wrote Hamlet?",
     "preferred": "William Shakespeare wrote Hamlet.",
     "rejected": "Hamlet is a famous play. It was written a long time ago by an English "
                 "playwright named Shakespeare."},
    {"prompt": "What is the largest planet?",
     "preferred": "Jupiter is the largest planet.",
     "rejected": "The solar system has eight planets. Some are small and rocky, others are "
                 "gas giants. Jupiter is the biggest."},
    {"prompt": "Convert 2 kilometres to metres.",
     "preferred": "2 kilometres is 2000 metres.",
     "rejected": "A kilometre is a unit of distance. There are one thousand metres in a "
                 "kilometre, so two would be two thousand."},
]


def trained(ref):
    np.random.seed(SEED)
    model = ref.RewardModel(**SHAPE)
    np.random.seed(SEED + 50)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, _ = ref.train_reward_model(model, ref.PREFERENCE_DATA, num_epochs=EPOCHS)
    return model


def reward(model, ref, prompt, text):
    ids = np.array(ref.tokenize_for_reward(prompt, text)[:MAX_LEN]).reshape(1, -1)
    return float(model.forward(ids)[0])


def margins(model, ref, pairs):
    """(a) mean preferred reward, (b) mean rejected reward, (c) mean margin, and accuracy."""
    preferred = [reward(model, ref, p["prompt"], p["preferred"]) for p in pairs]
    rejected = [reward(model, ref, p["prompt"], p["rejected"]) for p in pairs]
    gaps = [a - b for a, b in zip(preferred, rejected)]
    return {"preferred": statistics.fmean(preferred), "rejected": statistics.fmean(rejected),
            "margin": statistics.fmean(gaps), "correct": sum(g > 0 for g in gaps),
            "pairs": len(pairs)}


def length_bias(model, ref):
    """Correlation of reward with response length over every training response."""
    texts = ([(p["prompt"], p["preferred"]) for p in ref.PREFERENCE_DATA]
             + [(p["prompt"], p["rejected"]) for p in ref.PREFERENCE_DATA])
    return statistics.correlation([len(text) for _, text in texts],
                                  [reward(model, ref, prompt, text) for prompt, text in texts])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model = trained(ref)
    longer = sum(len(p["preferred"]) < len(p["rejected"]) for p in ref.PREFERENCE_DATA)
    return {
        "train": margins(model, ref, ref.PREFERENCE_DATA),
        "held": margins(model, ref, HELD_OUT),
        "correlation": length_bias(model, ref),
        "preferred_shorter": longer,
    }


def verify(result):
    train, held = result["train"], result["held"]
    return [
        practice.Check(
            "ANSWER: (a) +0.1752, (b) +0.2135, (c) -0.0383 -- the margin is negative",
            train["margin"] < 0 and train["correct"] < train["pairs"] / 2,
            f"on the {train['pairs']} pairs it was trained on, the average preferred reward is "
            f"{train['preferred']:+.4f}, the average rejected reward is {train['rejected']:+.4f}, "
            f"and the margin is {train['margin']:+.4f}. The model scores the rejected response "
            f"higher on average and gets {train['correct']} of {train['pairs']} right",
        ),
        practice.Check(
            "ANSWER: on four unseen pairs the margin holds its sign and triples in size",
            held["margin"] < train["margin"] < 0,
            f"the four new pairs give {held['margin']:+.4f} against {train['margin']:+.4f} on "
            f"training data -- {held['margin'] / train['margin']:.1f}x worse -- with "
            f"{held['correct']} of {held['pairs']} correct. The exercise asks whether a clear "
            "margin survives unseen data; what survives is the sign, and it is the wrong one",
        ),
        practice.Check(
            "MECHANISM: reward tracks length, and the rejected responses are the long ones",
            result["correlation"] > 0.2
            and result["preferred_shorter"] > len(HELD_OUT) // 2,
            f"reward correlates {result['correlation']:+.3f} with response length across the "
            f"twelve training responses, and the lesson's dataset makes the rejected response "
            f"the padded one: preferred is shorter in {result['preferred_shorter']} of "
            f"{train['pairs']} pairs. A model that has learned nothing except 'longer scores "
            "higher' gets this dataset backwards by construction",
        ),
        practice.Check(
            "FINDING: a margin is a discrimination statistic, not a calibration one",
            train["margin"] != 0,
            "calibration is whether a score maps to a probability -- a 0.7 margin meaning the "
            "preferred response wins 70% of the time. The exercise's metric would pass a model "
            "that ranked every pair correctly by 1e-9 and fail one that ranked them correctly by "
            f"1e9 with inconsistent confidence. This model's margin is {train['margin']:+.4f}, "
            "so the question of calibration does not arise yet",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
