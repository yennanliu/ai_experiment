"""Exercise 3 — the harder task is the one GRPO learns faster.

    **Hard.** Extend to a length-2 "reasoning chain": the agent emits two tokens and
    the verifier rewards the pair. Measure how GRPO handles the credit assignment
    across two-step sequences. (Hint: compute group advantage per *full sequence*,
    propagate to both token positions.)

Reading of the exercise: the hint is followed exactly -- one advantage per sequence,
applied unchanged at both positions -- and then tested against the thing it is a hint
about, by re-running with a verifier that scores each token separately. If sequence-
level credit were the bottleneck, the dense verifier would pull ahead; the comparison
is what turns the hint into a measurement.

**ANSWER: GRPO handles it, and handles it better than the single-token task.**
Sequence accuracy reaches 0.989 at 2,000 updates and 0.996 at 5,000, against 0.970 for
the length-1 bandit at 2,000 -- on a task whose random success rate is 1/16 rather than
1/4.

**FINDING: per-token credit buys almost nothing.** A verifier scoring each position
separately reaches 0.990 against the sequence-level 0.989 at the same budget. The
credit assignment the exercise asks about is not where the difficulty is.

**MECHANISM: the harder task keeps GRPO's groups disagreeing.** Exercise 1 measured
that GRPO's gradient vanishes when all `G` samples earn the same reward. At 1/16 the
group disagrees far longer: degenerate updates run 23.5% over the first 500 against
39.8% for the 1/4 task, so more of the budget carries signal.

**FINDING: and the advantage still reaches the right token.** The two positions are
independent in the policy, and a sequence is correct only when both are, so a
sequence-level advantage pushes both positions of a correct pair up and both of a
wrong one down -- which is the correct direction for each, even though neither
position is scored on its own.

Structure: `chain_step` is the hint, implemented literally; `train` runs it at either
verifier and reports the degenerate-group rate alongside the accuracy.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "12-rl-for-games"
GROUP, WIDTH, LR, SEEDS = 8, 4, 0.1, 10
CHECKPOINTS = (500, 2_000, 5_000)


def targets(ref):
    """The correct two-token answer for each prompt: the shipped token, then the next."""
    return tuple((q["correct"], (q["correct"] + 1) % WIDTH) for q in ref.QUESTIONS)


def score(seq, target, per_token):
    """The verifier: the whole pair, or each position separately."""
    if per_token:
        return 0.5 * (seq[0] == target[0]) + 0.5 * (seq[1] == target[1])
    return 1.0 if seq == target else 0.0


def apply_advantage(theta, prompt, probs, seq, adv):
    """One sequence advantage, applied unchanged at both positions."""
    for pos in (0, 1):
        for i in range(WIDTH):
            theta[prompt][pos][i] += (LR / GROUP) * adv * (
                (1.0 if i == seq[pos] else 0.0) - probs[pos][i])


def chain_step(ref, theta, pairs, rng, per_token):
    """One GRPO update on a length-2 chain: one advantage per sequence, both positions."""
    prompt = rng.randrange(ref.N_PROMPTS)
    probs = [ref.softmax(theta[prompt][pos]) for pos in (0, 1)]
    seqs = [tuple(ref.sample(probs[pos], rng) for pos in (0, 1)) for _ in range(GROUP)]
    rewards = [score(s, pairs[prompt], per_token) for s in seqs]
    mean, sd = statistics.fmean(rewards), statistics.pstdev(rewards) + 1e-8
    for seq, adv in zip(seqs, [(r - mean) / sd for r in rewards]):
        apply_advantage(theta, prompt, probs, seq, adv)
    return len(set(rewards)) == 1


def train(ref, updates, seed, per_token=False):
    """`updates` chain steps; returns exact sequence accuracy and the degenerate rate."""
    rng, pairs = random.Random(seed), targets(ref)
    theta = [[[0.0] * WIDTH for _ in (0, 1)] for _ in range(ref.N_PROMPTS)]
    flags = [chain_step(ref, theta, pairs, rng, per_token) for _ in range(updates)]
    accuracy = statistics.fmean(
        ref.softmax(theta[p][0])[pairs[p][0]] * ref.softmax(theta[p][1])[pairs[p][1]]
        for p in range(ref.N_PROMPTS))
    return accuracy, statistics.fmean(flags[:500])


def flat_accuracy(ref, theta):
    """The length-1 bandit's accuracy, for the reference column."""
    return statistics.fmean(ref.policy_probs(theta, p)[ref.QUESTIONS[p]["correct"]]
                            for p in range(ref.N_PROMPTS))


def arm(ref, per_token):
    """Accuracy and degenerate rate at each checkpoint, averaged over the seeds."""
    rows = {n: [train(ref, n, s, per_token) for s in range(SEEDS)] for n in CHECKPOINTS}
    return ({n: statistics.fmean(a for a, _d in v) for n, v in rows.items()},
            {n: statistics.fmean(d for _a, d in v) for n, v in rows.items()})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    chain, degenerate = arm(ref, False)
    dense, _ = arm(ref, True)
    flat = statistics.fmean(flat_accuracy(ref, ref.train_grpo(2_000, rng=random.Random(s))[0])
                            for s in range(SEEDS))
    return {"chain": chain, "degenerate": degenerate, "dense": dense, "flat": flat,
            "chance": 1 / WIDTH ** 2, "flat_chance": 1 / WIDTH}


def verify(result):
    chain, dense, flat = result["chain"], result["dense"], result["flat"]
    degenerate = result["degenerate"][CHECKPOINTS[1]]
    return [
        practice.Check(
            "ANSWER: GRPO handles it, and handles it better than the single-token task",
            chain[2_000] > flat and chain[CHECKPOINTS[-1]] > 0.99,
            "exact sequence accuracy at " + f"{CHECKPOINTS} updates: "
            + ", ".join(f"{chain[n]:.4f}" for n in CHECKPOINTS)
            + f", against {flat:.4f} for the length-1 bandit at 2,000. The chain's random "
            f"success rate is {result['chance']:.4f} against {result['flat_chance']:.2f}, so the "
            "harder task is the one that ends closer to solved",
        ),
        practice.Check(
            "FINDING: per-token credit buys almost nothing",
            abs(dense[2_000] - chain[2_000]) < 0.02,
            "re-running with a verifier that scores each position separately gives "
            + ", ".join(f"{dense[n]:.4f}" for n in CHECKPOINTS)
            + f" against the sequence-level {chain[2_000]:.4f} at 2,000 -- a difference of "
            f"{abs(dense[2_000] - chain[2_000]):.4f}. If sequence-level credit were the "
            "bottleneck the dense verifier would pull ahead; it does not, so the credit "
            "assignment the exercise asks about is not where the difficulty is",
        ),
        practice.Check(
            "MECHANISM: the harder task keeps GRPO's groups disagreeing",
            degenerate < 0.35,
            f"exercise 1 measured that GRPO's gradient vanishes when all {GROUP} samples earn "
            f"the same reward, and that the rate climbs to three quarters as the policy "
            f"improves. At a {result['chance']:.4f} success rate the group disagrees far longer: "
            f"degenerate updates run {100 * degenerate:.1f}% over the first 500 against 39.8% "
            "for the 1/4 task, so more of the budget carries signal",
        ),
        practice.Check(
            "FINDING: and the advantage still reaches the right token",
            chain[CHECKPOINTS[0]] > 0.8,
            f"the two positions are independent in the policy and a sequence is correct only "
            f"when both are, so one sequence-level advantage pushes both positions of a correct "
            f"pair up and both of a wrong pair down -- the right direction for each, even though "
            f"neither position is scored on its own. Accuracy is already {chain[500]:.4f} at 500 "
            "updates, from a start of one correct pair in sixteen",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
