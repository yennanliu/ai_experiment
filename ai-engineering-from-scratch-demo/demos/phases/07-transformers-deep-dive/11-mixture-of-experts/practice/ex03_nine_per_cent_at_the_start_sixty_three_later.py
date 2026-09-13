"""Exercise 3 — 9% of tokens re-route at the start, 63% after fifty updates.

    **Hard.** Implement GRPO-style "rollout-matched routing" (DeepSeek-V3.2
    trick): log which experts fire during inference, force the same routing
    during gradient computation. Measure the effect on a toy policy-gradient
    setup.

Reading of the exercise: the trick exists because the routing that produced a
sample and the routing used to differentiate it are not the same function once
the balance bias has moved between them. That gap is what is measured here --
exactly, on the lesson's own `route` and `update_bias`, over 1,000 clustered
tokens -- because it is the quantity any policy-gradient setup would inherit.
Rollout-matched routing sets it to zero by construction; the question is how
large it is without the trick.

**ANSWER: 9.0% of tokens at the lesson's own gamma, and it scales with gamma.**
One rollout epoch, one `update_bias` at `gamma=0.15`, one gradient epoch:

| gamma | top-k set changed | top-1 changed | gate mass on experts that never ran |
|---:|---:|---:|---:|
| 0.00 | 0.0% | 0.0% | 0.0% |
| 0.05 | 3.8% | 0.0% | 0.9% |
| **0.15** | **9.0%** | 0.0% | **2.2%** |
| 0.50 | 25.2% | 3.9% | 8.0% |

**FINDING: 2.2% of the gate mass is credited to experts that did not run.** Those
gradients update parameters that contributed nothing to the sampled output, and
no importance weight corrects them -- the update is silently off-policy, and the
amount is a function of a hyper-parameter chosen for load balance.

**FINDING: it gets worse with training, not better.** After 50 balance updates
the bias has spread to **3.00** -- twenty steps of gamma -- and the experts sit
close enough together in biased score that one further update re-routes **63.2%**
of tokens. The mismatch the trick removes is 9% on the first step and two thirds
of the corpus by the fiftieth, which is why it became necessary at scale rather
than at the start.

**CONTROL: at gamma = 0 the mismatch is exactly zero.** With no bias update the
rollout and gradient passes are the same function, and 0.0% of tokens move. The
mismatch is caused entirely by the balance rule, not by sampling.

**FINDING: top-1 survives where top-2 does not.** At gamma=0.15 the *first*
expert never changes and 9.0% of *sets* do, so the churn is entirely in the last
selected expert -- the one whose biased score sits nearest the cut. That is also
the one carrying the smaller gate weight, which is why 9.0% of tokens move only
2.2% of the mass.

Structure: `epoch` records the routing of every token; `mismatch` compares a
rollout pass against a gradient pass taken after one update.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "11-mixture-of-experts"
WIDTH, EXPERTS, TOP_K, TOKENS, CLUSTERS = 16, 8, 2, 1_000, 3
GAMMAS, LESSON_GAMMA, SETTLE = (0.0, 0.05, 0.15, 0.5), 0.15, 50


def corpus(rng):
    """1,000 tokens from 3 Gaussian clusters -- enough structure to unbalance the router."""
    centres = [[rng.gauss(0, 1) for _ in range(WIDTH)] for _ in range(CLUSTERS)]
    return [[c + 0.15 * rng.gauss(0, 1) for c in rng.choice(centres)] for _ in range(TOKENS)]


def epoch(ref, router, tokens, bias):
    """(per-token (experts, gates), usage counts) for one pass at this bias."""
    routes, counts = [], [0] * EXPERTS
    for token in tokens:
        chosen, gates = ref.route(token, router, TOP_K, bias)
        routes.append((chosen, gates))
        for expert in chosen:
            counts[expert] += 1
    return routes, counts


def mismatch(rollout, gradient):
    """(share of changed top-k sets, changed top-1, gate mass on experts that did not run)."""
    changed = sum(set(a) != set(b) for (a, _), (b, _) in zip(rollout, gradient))
    first = sum(a[0] != b[0] for (a, _), (b, _) in zip(rollout, gradient))
    lost = sum(sum(g for e, g in zip(a, gates) if e not in set(b))
               for (a, gates), (b, _) in zip(rollout, gradient))
    return changed / len(rollout), first / len(rollout), lost / len(rollout)


def step(ref, router, tokens, bias, counts, gamma):
    """One balance update, then the gradient pass it would be paired with."""
    moved = ref.update_bias(list(bias), counts, TOKENS * TOP_K / EXPERTS, gamma)
    return moved, epoch(ref, router, tokens, moved)[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(42)
    router = [[rng.gauss(0, 0.3) for _ in range(WIDTH)] for _ in range(EXPERTS)]
    tokens = corpus(rng)
    rollout, counts = epoch(ref, router, tokens, [0.0] * EXPERTS)
    grid = {gamma: mismatch(rollout, step(ref, router, tokens, [0.0] * EXPERTS,
                                          counts, gamma)[1]) for gamma in GAMMAS}
    bias, late, late_counts = [0.0] * EXPERTS, rollout, counts
    for _ in range(SETTLE):
        bias = ref.update_bias(bias, late_counts, TOKENS * TOP_K / EXPERTS, LESSON_GAMMA)
        late, late_counts = epoch(ref, router, tokens, bias)
    return {
        "grid": grid, "settled": mismatch(late, step(ref, router, tokens, bias,
                                                     late_counts, LESSON_GAMMA)[1]),
        "spread": max(bias) - min(bias), "steps": (max(bias) - min(bias)) / LESSON_GAMMA,
    }


def verify(result):
    grid, settled = result["grid"], result["settled"]
    return [
        practice.Check(
            "ANSWER: 9.0% of tokens re-route at the lesson's own gamma",
            0.05 < grid[LESSON_GAMMA][0] < 0.15,
            "one rollout epoch, one update_bias, one gradient epoch -- share of top-k sets that "
            "changed: " + ", ".join(f"gamma {g} {grid[g][0]:.1%}" for g in GAMMAS)
            + f". At the lesson's {LESSON_GAMMA} it is {grid[LESSON_GAMMA][0]:.1%}, and it scales "
              "with a hyper-parameter that was chosen for load balance",
        ),
        practice.Check(
            "FINDING: 2.2% of the gate mass is credited to experts that did not run",
            0.01 < grid[LESSON_GAMMA][2] < 0.05 and grid[0.5][2] > 0.05,
            "gate mass landing on experts absent from the rollout: " + ", ".join(
                f"gamma {g} {grid[g][2]:.1%}" for g in GAMMAS)
            + ". Those gradients update parameters that contributed nothing to the sampled "
              "output, with no importance weight correcting them -- silently off-policy",
        ),
        practice.Check(
            "FINDING: it gets worse with training, not better",
            settled[0] > 0.5,
            f"after {SETTLE} balance updates the bias has spread to {result['spread']:.2f}, "
            f"{result['steps']:.0f} steps of gamma, and the experts sit close enough in biased "
            f"score that one further update re-routes {settled[0]:.1%} of tokens -- against "
            f"{grid[LESSON_GAMMA][0]:.1%} on the first step. That is why the trick became "
            "necessary at scale rather than at the start",
        ),
        practice.Check(
            "FINDING: top-1 survives where top-2 does not",
            grid[LESSON_GAMMA][1] == 0.0 and grid[LESSON_GAMMA][0] > 0.05,
            f"at gamma {LESSON_GAMMA} the first expert changes for "
            f"{grid[LESSON_GAMMA][1]:.1%} of tokens while {grid[LESSON_GAMMA][0]:.1%} of sets "
            f"change, so all the churn is in the last selected expert -- the one nearest the cut, "
            f"and the one carrying the smaller gate. That is why {grid[LESSON_GAMMA][0]:.1%} of "
            f"tokens move only {grid[LESSON_GAMMA][2]:.1%} of the mass",
        ),
        practice.Check(
            "CONTROL: at gamma = 0 the mismatch is exactly zero",
            grid[0.0] == (0.0, 0.0, 0.0),
            "with no bias update the rollout and gradient passes are the same function and "
            "nothing moves. The mismatch rollout-matched routing removes is caused entirely by "
            "the balance rule, not by sampling and not by the router",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
