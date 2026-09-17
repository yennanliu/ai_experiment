"""Exercise 4 — the combined objective scores 4 of 6, and so does conciseness alone.

    Implement a multi-objective reward. Train two reward models -- one for
    helpfulness and one for conciseness. Combine them as R = 0.7 * R_helpful +
    0.3 * R_concise. Show that the combined objective produces responses that are
    both helpful and concise, avoiding the verbosity trap of a single
    helpfulness reward.

Reading of the exercise: the helpfulness model is the lesson's own, trained by
its own `train_reward_model` on its own six pairs. Conciseness is
`-len(response) / 100`, the sign-flipped version of the length reward Exercise 3
asks for, so the two exercises measure the same quantity in opposite directions.
"Show that the combined objective produces responses that are both" is scored as
agreement with the six human preferences, since `ppo_training` cannot move a
policy (Exercise 3) and generated responses would carry no information.

**ANSWER: the combination gets 4 of 6, up from 2 of 6 for helpfulness alone.**
That is the improvement the exercise predicts.

**FINDING: conciseness alone also gets 4 of 6.** Dropping the helpfulness term
entirely -- `R = 0.0 * R_helpful + 1.0 * R_concise` -- scores exactly the same.
The 0.7 weight on the trained model contributes nothing the length term was not
already supplying, and the two pairs the combination still gets wrong are the
two where the preferred response is the *longer* one.

**FINDING: the verbosity trap is in the helpfulness model, not opposite it.**
The exercise frames conciseness as a counterweight to a helpfulness reward that
over-rewards length. But this helpfulness model *correlates +0.268 with response
length* and scores 2 of 6 -- below chance. The conciseness term is not balancing
a helpful-but-verbose signal; it is overriding a signal that is verbose and not
helpful.

**FINDING: 0.7 sits on a plateau, and 0.9 reaches 5 of 6.** Sweeping the weight
gives 4, 4, 4, 4, 4, **5**, 2 from 0.0 to 1.0. Everything up to 0.7 is tied with
pure conciseness; the only weight that beats it leans **9:1 toward the component
that scores 2 of 6 alone**. The two signals do combine non-trivially -- just not
at the weight the exercise picks, and the direction that helps is *less*
conciseness, not more.

Structure: `combined` is the weighted objective; `agreement` counts how many of
the six preferences it reproduces at a given weight.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "07-rlhf"
SEED, EPOCHS, MAX_LEN = 1, 10, 128
HELPFUL_WEIGHT = 0.7
WEIGHTS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=MAX_LEN, ff_dim=256)


def trained(ref):
    np.random.seed(SEED)
    model = ref.RewardModel(**SHAPE)
    np.random.seed(SEED + 50)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, _ = ref.train_reward_model(model, ref.PREFERENCE_DATA, num_epochs=EPOCHS)
    return model


def helpful(model, ref, prompt, text):
    ids = np.array(ref.tokenize_for_reward(prompt, text)[:MAX_LEN]).reshape(1, -1)
    return float(model.forward(ids)[0])


def concise(text):
    """The conciseness reward: Exercise 3's length reward with the sign flipped."""
    return -len(text) / 100.0


def combined(model, ref, prompt, text, weight):
    return weight * helpful(model, ref, prompt, text) + (1 - weight) * concise(text)


def agreement(model, ref, weight):
    """How many of the six human preferences this weighting reproduces."""
    return sum(combined(model, ref, p["prompt"], p["preferred"], weight)
               > combined(model, ref, p["prompt"], p["rejected"], weight)
               for p in ref.PREFERENCE_DATA)


def length_bias(model, ref):
    texts = ([(p["prompt"], p["preferred"]) for p in ref.PREFERENCE_DATA]
             + [(p["prompt"], p["rejected"]) for p in ref.PREFERENCE_DATA])
    return statistics.correlation([len(text) for _, text in texts],
                                  [helpful(model, ref, prompt, text) for prompt, text in texts])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model = trained(ref)
    data = ref.PREFERENCE_DATA
    return {
        "sweep": {w: agreement(model, ref, w) for w in WEIGHTS},
        "pairs": len(data),
        "correlation": length_bias(model, ref),
        "preferred_shorter": sum(len(p["preferred"]) < len(p["rejected"]) for p in data),
        "longer_pairs": [p["prompt"] for p in data
                         if len(p["preferred"]) >= len(p["rejected"])],
    }


def verify(result):
    sweep, pairs = result["sweep"], result["pairs"]
    mixed, only_helpful, only_concise = sweep[HELPFUL_WEIGHT], sweep[1.0], sweep[0.0]
    return [
        practice.Check(
            f"ANSWER: R = {HELPFUL_WEIGHT} helpful + {1 - HELPFUL_WEIGHT:.1f} concise gets "
            f"{mixed} of {pairs}, up from {only_helpful}",
            mixed > only_helpful,
            f"the combined objective reproduces {mixed} of the {pairs} human preferences against "
            f"{only_helpful} for the trained helpfulness model alone. That is the improvement the "
            "exercise predicts, and it is real",
        ),
        practice.Check(
            f"FINDING: conciseness alone also gets {only_concise} of {pairs}",
            only_concise == mixed,
            f"dropping the trained model entirely -- R = 0.0 * helpful + 1.0 * concise -- scores "
            f"exactly the same {only_concise}. The {HELPFUL_WEIGHT} weight on the helpfulness "
            "model contributes nothing the length term was not already supplying, and the pairs "
            f"the combination still gets wrong are {result['longer_pairs']}, the two where the "
            "preferred response is the longer one",
        ),
        practice.Check(
            "FINDING: the verbosity trap is inside the helpfulness model, not opposite it",
            result["correlation"] > 0.2 and only_helpful < pairs / 2,
            f"the exercise frames conciseness as a counterweight to a helpfulness reward that "
            f"over-rewards length. This helpfulness model correlates "
            f"{result['correlation']:+.3f} with response length and scores {only_helpful} of "
            f"{pairs}, below chance. The conciseness term is not balancing a helpful-but-verbose "
            "signal; it is overriding a signal that is verbose and not helpful",
        ),
        practice.Check(
            f"FINDING: {HELPFUL_WEIGHT} sits on a plateau -- 0.9 reaches {sweep[0.9]} of {pairs}",
            sweep[0.9] > mixed == only_concise,
            "sweeping the weight from 0 to 1 gives "
            + ", ".join(f"{w:.1f} -> {n}" for w, n in sweep.items())
            + f". Everything from 0.0 to {HELPFUL_WEIGHT} scores {mixed}, tied with pure "
            f"conciseness, and the only weight that does better is 0.9 at {sweep[0.9]} -- a 9:1 "
            f"mix in favour of the component that scores {only_helpful} of {pairs} on its own. "
            "The two signals do combine non-trivially, just not at the weight the exercise "
            "picks, and the direction that helps is less conciseness rather than more",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
