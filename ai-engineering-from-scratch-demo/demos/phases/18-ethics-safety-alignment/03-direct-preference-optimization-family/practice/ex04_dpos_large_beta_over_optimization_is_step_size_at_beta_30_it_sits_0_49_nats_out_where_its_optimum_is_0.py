"""Exercise 4 — DPO's large-beta over-optimization is step size: at beta 30 it sits 0.49 nats out where its optimum is 0.

    Rafailov et al. (NeurIPS 2024) claim DAAs over-optimize. Reproduce a
    single-point version: plot chosen-minus-rejected KL divergence and observe
    over-optimization in DPO at large beta.

Reading of the exercise: "chosen-minus-rejected KL" is read as the mean over
pairs of log(pi/pi_ref)(y_w) - log(pi/pi_ref)(y_l), the per-sample log-ratio gap
that DPO's implicit reward scales by beta. The policy's KL(pi || pi_ref) is
reported beside it. Gold is E_pi[TRUE_UTILITY], which is the reward the
preferences were sampled from. Over-optimization means more KL for less gold.
Each beta is run with the shipped `train_dpo` (2,000 steps, lr 0.05) on 20
seeds of 500 fresh pairs. The "plot" is the table.

**ANSWER: the shape appears, at large beta.**

| beta | KL | log-ratio gap | gold | runs with pi(best) below ref | KL at lr x beta = 0.005 | closed-form KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.339 | 0.743 | 0.646 | 0/20 | 0.339 | 1.196 |
| 0.3 | 0.684 | 1.294 | 0.844 | 0/20 | 0.265 | 0.934 |
| 1 | 0.236 | 0.585 | 0.559 | 0/20 | 0.122 | 0.209 |
| 3 | 0.048 | 0.213 | 0.293 | 0/20 | 0.027 | 0.026 |
| 10 | 0.060 | 0.175 | 0.250 | 4/20 | 0.003 | 0.002 |
| 30 | 0.489 | 0.534 | 0.432 | 7/20 | 0.000 | 0.000 |

KL falls with beta up to 3 and then climbs again. At beta 30 the policy sits
0.489 nats from the reference, further than at beta 0.1 (0.339), yet with less
gold (0.432 against 0.646). In 7 of 20 runs it ends with less mass on the best
action than the reference started with.

**FINDING: it is the optimizer, not Goodhart.** `train_dpo` multiplies its
gradient by beta, so beta is also a learning rate: lr x beta is 1.5 at beta 30.
If lr is scaled so that lr x beta stays at the shipped 0.005, KL falls
monotonically with beta, to 0.000 at beta 30, and no run drops below the
reference. The closed-form optimum pi* ~ pi_ref exp(u / beta) agrees: its KL
is 0.000 at beta 30. Its gold only falls as beta rises (1.000 at beta 0.1 to
0.121 at beta 30), so in this toy gold never peaks and collapses.

**FINDING: large beta is the wrong end for this claim.** In DPO beta is the KL
coefficient, so a large beta *restrains* the policy. Rafailov et al. find
over-optimization at large KL budgets, that is small beta. The toy cannot
show that either. Its preferences are Bradley-Terry on the gold utility
itself, so the implicit reward has no gap from gold to exploit.

Structure: `sweep()` trains per seed and averages `measure()`; `closed_form()`
evaluates the RLHF optimum through the reference's own Policy.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "03-direct-preference-optimization-family"
BETAS, SEEDS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0), range(20)


def measure(ref, pi, pairs):
    """(KL(pi || ref), mean chosen-minus-rejected log-ratio, gold E_pi[u], pi(best))."""
    base = ref.logsoftmax(ref.make_policy_and_ref()[1].logits)
    logp, p = ref.logsoftmax(pi.logits), ref.softmax(pi.logits)
    d = [a - b for a, b in zip(logp, base)]
    return (sum(pa * da for pa, da in zip(p, d)),
            sum(d[w] - d[lo] for w, lo, _ in pairs) / len(pairs),
            sum(pa * u for pa, u in zip(p, ref.TRUE_UTILITY)), p[1])


def sweep(ref, beta, lr):
    """Mean of measure() over SEEDS, plus how many runs end with pi(best) below the reference."""
    rows = []
    for seed in SEEDS:
        ref.random = random.Random(seed)
        pairs = [ref.sample_pref_pair() for _ in range(500)]
        rows.append(measure(ref, ref.train_dpo(pairs, beta=beta, lr=lr), pairs))
    start = ref.softmax(ref.make_policy_and_ref()[1].logits)[1]
    means = tuple(round(sum(col) / len(rows), 3) for col in zip(*rows))
    return means + (sum(r[3] < start for r in rows),)


def closed_form(ref, beta):
    """RLHF optimum pi* ~ pi_ref exp(u / beta): (KL, gold)."""
    logits = ref.make_policy_and_ref()[1].logits
    pi = ref.Policy([a + u / beta for a, u in zip(logits, ref.TRUE_UTILITY)])
    kl, _, gold, _ = measure(ref, pi, [(0, 0, 0.0)])
    return round(kl, 3), round(gold, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lr = inspect.signature(ref.train_dpo).parameters["lr"].default
    saved = ref.random
    try:
        ship = {b: sweep(ref, b, lr) for b in BETAS}
        held = {b: sweep(ref, b, lr * BETAS[0] / b) for b in BETAS}
    finally:
        ref.random = saved
    return {"lr": lr, "ship": ship, "held": held, "theory": {b: closed_form(ref, b) for b in BETAS},
            "beta_in_grad": "grad = [beta * (g_margin" in inspect.getsource(ref.train_dpo),
            "bt_on_gold": "TRUE_UTILITY[i] - TRUE_UTILITY[j]" in inspect.getsource(ref.sample_pref_pair)}


SHIP = {0.1: (0.339, 0.743, 0.646, 0.667, 0), 0.3: (0.684, 1.294, 0.844, 0.836, 0),
        1.0: (0.236, 0.585, 0.559, 0.602, 0), 3.0: (0.048, 0.213, 0.293, 0.423, 0),
        10.0: (0.06, 0.175, 0.25, 0.387, 4), 30.0: (0.489, 0.534, 0.432, 0.508, 7)}


def verify(result):
    ship, held, theory = result["ship"], result["held"], result["theory"]
    held_kl = [held[b][0] for b in BETAS]
    return [
        practice.Check(
            "ANSWER: the shape appears, at large beta",
            all([ship == SHIP, ship[30.0][0] > ship[0.1][0], ship[30.0][2] < ship[0.1][2]]),
            f"(KL, log-ratio gap, gold, pi best, runs below ref) by beta at lr {result['lr']}: "
            f"{ship}",
        ),
        practice.Check(
            "FINDING: it is the optimizer, not Goodhart",
            all([result["beta_in_grad"], round(result["lr"] * 30, 6) == 1.5,
                 held_kl == sorted(held_kl, reverse=True) == [0.339, 0.265, 0.122, 0.027, 0.003, 0.0],
                 [held[b][4] for b in BETAS] == [0] * 6, theory[30.0] == (0.0, 0.121),
                 theory[0.1][1] == 1.0]),
            f"KL with lr x beta held at {result['lr'] * BETAS[0]:g}: {held_kl}; closed-form "
            f"(KL, gold): {theory}",
        ),
        practice.Check(
            "FINDING: large beta is the wrong end for this claim",
            all([[t[0] for t in theory.values()] == [1.196, 0.934, 0.209, 0.026, 0.002, 0.0],
                 [t[1] for t in theory.values()] == sorted((t[1] for t in theory.values()),
                                                           reverse=True),
                 result["bt_on_gold"]]),
            "the RLHF optimum's KL and gold both fall monotonically as beta rises; "
            "sample_pref_pair draws preferences from sigmoid(TRUE_UTILITY[i] - TRUE_UTILITY[j])",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
