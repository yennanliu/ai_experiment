"""Exercise 3 — comparing on RM score rewards whichever method hacks hardest.

    **Hard.** Implement DPO (closed-form preference-likelihood loss) on the same
    preference data and compare to the RLHF-PPO pipeline in compute used and final
    RM score achieved.

Reading of the exercise: the comparison it asks for is compute and RM score, and
exercise 2 established that the RM score is maximised by collapsing onto a single
token -- so the honest version reports KL-to-reference alongside, because otherwise
the winner is whichever method hacks the reward model hardest. DPO is written here;
the reference policy is uniform, matching `rlhf_loop`'s zero-initialised reference,
and a response's log-probability is the sum over its two tokens.

**ANSWER: PPO scores higher and DPO stays closer.** At 2,400 gradient steps each,
PPO-RLHF reaches an RM score of **1.178** at **91%** of the maximum KL, and DPO
reaches **0.893** at **16%**. PPO reaches 99.2% of the reward model's ceiling by
becoming the one token that ceiling belongs to.

**FINDING: DPO uses no policy samples and no reward model at all.** PPO-RLHF spends
600 gradient steps fitting the reward model, then draws **2,400** samples from the
policy and queries the reward model 2,400 times. DPO reads the same preference pairs
directly: 0 samples, 0 reward-model queries, and stage 1 never has to happen.

**FINDING: DPO's score rises with data and its KL rises with it.** 0.295, 0.893,
1.086 at 600, 2,400 and 10,000 pairs, against KL at 2%, 16% and 28% of maximum. It
is on the same trajectory PPO ran to the end, just much further back along it.

**FINDING: the exercise's two axes disagree, and only one of them is stated.** Ranked
by RM score PPO wins by 0.285; ranked by KL-to-reference at equal compute DPO wins by
a factor of 5.7. "Final RM score achieved" is the axis that reward hacking is defined
to maximise.

Structure: `dpo` is the closed-form loss; `evaluate` scores any policy the same way
for both arms, so the comparison is not between two different measurements.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "09-reward-modeling-rlhf"
RM_PAIRS, UPDATES, ROLLOUTS, BETA, LR = 600, 150, 16, 0.1, 0.1
MATCHED = UPDATES * ROLLOUTS
SIZES = (600, MATCHED, 10_000)


def log_response(ref, theta, prompt, tokens, index):
    """`log π(response | prompt)` as the sum over the response's tokens."""
    probs = ref.policy_probs(theta, prompt)
    return sum(math.log(max(probs[index[t]], 1e-12)) for t in tokens)


def apply_step(ref, theta, p, index, pairs, step):
    """Push probability toward the preferred tokens and away from the rejected ones."""
    probs = ref.policy_probs(theta, p)
    for tokens, sign in pairs:
        for token in tokens:
            for i in range(len(ref.VOCAB)):
                theta[p][i] += sign * step * ((1.0 if i == index[token] else 0.0) - probs[i])


def dpo(ref, n_pairs, beta=BETA, lr=LR, seed=0):
    """Closed-form DPO: -log sigmoid(beta * (preferred log-ratio - rejected log-ratio))."""
    rng = random.Random(seed)
    index = {t: i for i, t in enumerate(ref.VOCAB)}
    theta = [[0.0 for _ in ref.VOCAB] for _ in ref.PROMPTS]
    reference = [[0.0 for _ in ref.VOCAB] for _ in ref.PROMPTS]
    for _ in range(n_pairs):
        prompt, preferred, rejected = ref.sample_pair(rng)
        p = ref.PROMPTS.index(prompt)
        margin = beta * ((log_response(ref, theta, p, preferred, index)
                          - log_response(ref, reference, p, preferred, index))
                         - (log_response(ref, theta, p, rejected, index)
                            - log_response(ref, reference, p, rejected, index)))
        apply_step(ref, theta, p, index, ((preferred, 1.0), (rejected, -1.0)),
                   lr * ref.sigmoid(-margin) * beta)
    return theta


def evaluate(ref, model, theta, samples=2000, seed=5):
    """Mean RM score of a sampled token and mean KL to the uniform reference."""
    rng = random.Random(seed)
    reference = [[0.0 for _ in ref.VOCAB] for _ in ref.PROMPTS]
    scores, kls = [], []
    for _ in range(samples):
        prompt = rng.randrange(len(ref.PROMPTS))
        probs = ref.policy_probs(theta, prompt)
        scores.append(model.get(ref.VOCAB[ref.sample_token(probs, rng)], 0.0))
        kls.append(ref.kl(probs, ref.policy_probs(reference, prompt)))
    return {"rm": statistics.fmean(scores), "kl": statistics.fmean(kls)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model = ref.train_rm(n_pairs=RM_PAIRS, rng=random.Random(42))
    ppo_theta, _ = ref.rlhf_loop(model, updates=UPDATES, beta=BETA, rng=random.Random(0))
    return {"ppo": evaluate(ref, model, ppo_theta),
            "dpo": {n: evaluate(ref, model, dpo(ref, n)) for n in SIZES},
            "maxkl": math.log(len(ref.VOCAB)), "ceiling": max(model.values())}


def verify(result):
    ppo, dpo_rows, maxkl = result["ppo"], result["dpo"], result["maxkl"]
    matched = dpo_rows[MATCHED]
    return [
        practice.Check(
            "ANSWER: PPO scores higher and DPO stays closer, at equal gradient steps",
            ppo["rm"] > matched["rm"] and ppo["kl"] > 3 * matched["kl"],
            f"at {MATCHED:,} gradient steps each, PPO-RLHF reaches RM {ppo['rm']:.3f} at "
            f"{100 * ppo['kl'] / maxkl:.0f}% of the maximum KL ({maxkl:.4f}), and DPO reaches "
            f"{matched['rm']:.3f} at {100 * matched['kl'] / maxkl:.0f}%. PPO is at "
            f"{100 * ppo['rm'] / result['ceiling']:.1f}% of the reward model's ceiling "
            f"({result['ceiling']:.4f}) -- it became the single token that ceiling belongs to",
        ),
        practice.Check(
            "FINDING: DPO uses no policy samples and no reward model at all",
            MATCHED == UPDATES * ROLLOUTS,
            f"PPO-RLHF spends {RM_PAIRS} gradient steps fitting the reward model, then draws "
            f"{UPDATES} x {ROLLOUTS} = {MATCHED:,} samples from the policy and queries the "
            f"reward model once per sample. DPO reads the same preference pairs directly: 0 "
            f"policy samples, 0 reward-model queries, and stage 1 never has to happen -- so the "
            f"{RM_PAIRS} pairs the reward model consumed are the same pairs DPO trains on",
        ),
        practice.Check(
            "FINDING: DPO's score rises with data and its KL rises with it",
            all(dpo_rows[a]["rm"] < dpo_rows[b]["rm"] for a, b in zip(SIZES, SIZES[1:]))
            and all(dpo_rows[a]["kl"] < dpo_rows[b]["kl"] for a, b in zip(SIZES, SIZES[1:])),
            "RM score and KL share by pair count -- "
            + ", ".join(f"{n:,}: {dpo_rows[n]['rm']:.3f} at "
                        f"{100 * dpo_rows[n]['kl'] / maxkl:.0f}%" for n in SIZES)
            + ". DPO is not on a different trajectory from PPO, it is further back along the "
            "same one: every gain in reward-model score is bought with distance from the "
            "reference, for both methods",
        ),
        practice.Check(
            "FINDING: the exercise's two axes disagree, and only one of them is stated",
            ppo["rm"] > matched["rm"] and matched["kl"] < ppo["kl"],
            f"ranked by the stated axis, final RM score, PPO wins by "
            f"{ppo['rm'] - matched['rm']:.3f}. Ranked by KL-to-reference at the same compute, "
            f"DPO wins by a factor of {ppo['kl'] / matched['kl']:.1f}. Exercise 2 established "
            "that RM score is maximised by collapsing onto one token, so 'final RM score "
            "achieved' is the axis reward hacking is defined to maximise -- comparing on it "
            "alone scores the methods by how hard each one hacks",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
