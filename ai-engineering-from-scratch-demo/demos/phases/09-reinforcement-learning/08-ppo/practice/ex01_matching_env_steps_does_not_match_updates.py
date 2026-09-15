"""Exercise 1 — matching env steps does not match updates, and PPO gets both.

    **Easy.** Run PPO on 4×4 GridWorld with `ε=0.2, K=4`. Compare sample
    efficiency to A2C (one epoch per rollout) at matched env steps.

Reading of the exercise: "A2C (one epoch per rollout)" is the lesson's own
`ppo_update` at `epochs=1`, so no second algorithm is written. "At matched env
steps" is taken literally -- each arm runs until it has consumed a 4,000-step
budget rather than for a fixed number of updates -- because a fixed update count
does not match env steps here, and the direction of the mismatch flatters the
loser.

**ANSWER: PPO by 1.56 return, at a matched 4,000-step budget.** -6.25 against
A2C's -7.81 over 12 seeds.

**FINDING: matching env steps hands PPO more updates, not fewer.** Inside the same
budget PPO completes **51.6** updates and A2C **34.6**. A better policy reaches the
terminal sooner, so its rollouts are shorter, so more of them fit -- the arm that
is learning faster is also charged less per update.

**FINDING: at matched *updates* the comparison inverts the other way.** Given 60
updates each, A2C consumes **5,468** env steps to PPO's **4,444** -- 23% more --
and still finishes worse (-6.57 against -6.18). Neither budget is neutral, and the
exercise's phrase decides the answer before the run starts.

**FINDING: `gae` bleeds across episode and environment boundaries.** `collect_rollout`
concatenates 8 independent environments into one buffer and `gae` never resets its
accumulator, so a truncated episode bootstraps from the next environment's first
state. Against the same recursion reset at each boundary, the worst per-step
advantage differs by **15.81** on a board where `|V|` is about 6. Both arms carry
it equally, so the comparison survives -- the numbers above are a comparison, not a
measurement of PPO.

Structure: `budgeted` runs one arm to an env-step budget; `fixed` runs one arm for
a fixed update count; `bleed` is the boundary check against a reset recursion.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "08-ppo"
BUDGET, UPDATES, SEEDS, EPS = 4_000, 60, 12, 0.2
GAMMA, LAM = 0.99, 0.95


def budgeted(ref, epochs, seed):
    """Run until `BUDGET` env steps are consumed; return the evaluated policy and the cost."""
    rng = random.Random(seed)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    steps = updates = 0
    while steps < BUDGET:
        buffer = ref.collect_rollout(theta, w, rng)
        steps += len(buffer)
        updates += 1
        advantages, returns = ref.gae(buffer)
        ref.ppo_update(theta, w, buffer, advantages, returns, eps=EPS, epochs=epochs, rng=rng)
    return {"final": ref.evaluate(theta, random.Random(999), episodes=200),
            "updates": updates, "steps": steps}


def fixed(ref, epochs, seed):
    """Run for `UPDATES` updates whatever that costs; return the policy and the cost."""
    rng = random.Random(seed)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    steps = 0
    for _ in range(UPDATES):
        buffer = ref.collect_rollout(theta, w, rng)
        steps += len(buffer)
        advantages, returns = ref.gae(buffer)
        ref.ppo_update(theta, w, buffer, advantages, returns, eps=EPS, epochs=epochs, rng=rng)
    return {"final": ref.evaluate(theta, random.Random(999), episodes=200), "steps": steps}


def reset_gae(ref, buffer):
    """The lesson's own recursion, reset at every episode boundary."""
    advantages, carry = [0.0] * len(buffer), 0.0
    for t in reversed(range(len(buffer))):
        done = buffer[t]["done"]
        nxt = 0.0 if done else (buffer[t + 1]["v_old"] if t + 1 < len(buffer) else 0.0)
        delta = buffer[t]["r"] + GAMMA * nxt - buffer[t]["v_old"]
        carry = delta if done else delta + GAMMA * LAM * carry
        advantages[t] = carry
    return advantages


def bleed(ref, seed=0, rounds=20):
    """Worst per-step disagreement between the shipped `gae` and a reset one."""
    rng = random.Random(seed)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    worst, dones, total = 0.0, 0, 0
    for _ in range(rounds):
        buffer = ref.collect_rollout(theta, w, rng)
        shipped, _ = ref.gae(buffer)
        worst = max(worst, max(abs(a - b) for a, b in zip(shipped, reset_gae(ref, buffer))))
        dones += sum(1 for rec in buffer if rec["done"])
        total += len(buffer)
    return worst, dones, total


def mean(rows, key):
    return statistics.fmean(r[key] for r in rows)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"ppo": [budgeted(ref, 4, s) for s in range(SEEDS)],
            "a2c": [budgeted(ref, 1, s) for s in range(SEEDS)],
            "ppo_fixed": [fixed(ref, 4, s) for s in range(SEEDS)],
            "a2c_fixed": [fixed(ref, 1, s) for s in range(SEEDS)],
            "bleed": bleed(ref)}


def verify(result):
    ppo, a2c = result["ppo"], result["a2c"]
    pf, af = result["ppo_fixed"], result["a2c_fixed"]
    worst, dones, total = result["bleed"]
    return [
        practice.Check(
            "ANSWER: PPO by 1.56 return at a matched 4,000-step budget",
            mean(ppo, "final") > mean(a2c, "final") + 0.5,
            f"over {SEEDS} seeds, each arm run until it has consumed {BUDGET:,} env steps: PPO "
            f"(K=4) evaluates at {mean(ppo, 'final'):.2f} and A2C (K=1) at "
            f"{mean(a2c, 'final'):.2f}, a gap of "
            f"{mean(ppo, 'final') - mean(a2c, 'final'):.2f}. Actual steps consumed are "
            f"{mean(ppo, 'steps'):.0f} and {mean(a2c, 'steps'):.0f}, matched to "
            f"{abs(mean(ppo, 'steps') - mean(a2c, 'steps')):.0f}",
        ),
        practice.Check(
            "FINDING: matching env steps hands PPO more updates, not fewer",
            mean(ppo, "updates") > mean(a2c, "updates"),
            f"inside the same budget PPO completes {mean(ppo, 'updates'):.1f} updates and A2C "
            f"{mean(a2c, 'updates'):.1f}. A better policy reaches the terminal sooner, so its "
            f"rollouts are shorter and more of them fit in a fixed step count -- the arm that is "
            "learning faster is also charged less per update, which is the opposite of the "
            "handicap 'matched env steps' sounds like it applies",
        ),
        practice.Check(
            "FINDING: at matched updates the mismatch runs the other way, and PPO still wins",
            mean(af, "steps") > mean(pf, "steps") and mean(pf, "final") > mean(af, "final"),
            f"given {UPDATES} updates each, A2C consumes {mean(af, 'steps'):.0f} env steps to "
            f"PPO's {mean(pf, 'steps'):.0f}, "
            f"{100 * (mean(af, 'steps') / mean(pf, 'steps') - 1):.0f}% more, and still finishes "
            f"worse ({mean(af, 'final'):.2f} against {mean(pf, 'final'):.2f}). Neither budget is "
            "neutral; the exercise's phrasing settles part of the answer before the run starts",
        ),
        practice.Check(
            "FINDING: gae bleeds across episode and environment boundaries",
            worst > 1.0,
            f"`collect_rollout` concatenates 8 independent environments into one buffer and `gae` "
            f"never resets its accumulator, so a truncated episode bootstraps from the next "
            f"environment's first state. Against the same recursion reset at each boundary the "
            f"worst per-step advantage differs by {worst:.2f}, on a board where |V| is about 6. "
            f"Only {dones} of {total:,} buffered steps carry a done flag, so most boundaries are "
            "truncations the recursion cannot see",
        ),
        practice.Check(
            "FINDING: A2C needed no new code -- it is the shipped update at epochs=1",
            mean(pf, "final") > mean(af, "final"),
            f"`ppo_update(epochs=1)` performs one pass over the rollout, which is what 'A2C, one "
            f"epoch per rollout' means, so both arms here are the same function at two settings "
            f"and differ in nothing else -- same clip, same normalisation, same critic, same "
            f"bleeding `gae`. The {mean(pf, 'final') - mean(af, 'final'):.2f} between them at "
            "matched updates is attributable to the epoch count alone",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
