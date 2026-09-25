"""Exercise 5 — the synergy is all in the no-ToM runs, and it is the collision rule.

    Read Riedl 2025 (arXiv:2510.05174). Implement the higher-order synergy
    statistic on your simulation logs. Is the effect present without the ToM
    prompt condition?

Reading of the exercise: Riedl's statistic is a partial information
decomposition of time-delayed mutual information -- for each agent pair,
I({X_i,t, X_j,t}; T_ij,t+1) split into unique, redundant and synergistic
parts with Williams-Beer I_min redundancy and plug-in probabilities (§2) --
plus a "practical criterion" S_macro = I(V_t; V_t+1) - sum_k I(X_k,t; V_t+1).
Here X_k,t is agent k's committed box (-1 once done), logs are padded to 3
turns, T_ij is the pair's next choices and V_t the number still searching. A
shuffled-target control, not in the paper, sizes the plug-in bias.

**ANSWER: yes -- and only there.** At 3x3 the zeroth-order runs carry 0.480
bits of pairwise synergy out of 0.965 bits of TDMI, against 0.078 with the
target shuffled. The primed ToM runs carry exactly 0: every agent takes box
i on turn 0 and is done, so every variable is constant and there is no
information to decompose. Unprimed ToM is the zeroth-order condition trial
for trial (exercise 1), so it scores the same 0.480. By this statistic the
condition with coordination has none, and the one without has plenty.

**FINDING: the synergy is the collision rule, not coordination.** It sits
entirely in the turn-0 -> turn-1 step (0.885 bits for agents 0 and 1; 0
after). Whether agent 1 is still searching on turn 1 depends on whether it
picked the *same* box as agent 0 -- a function of both choices that neither
predicts alone, XOR-shaped. The environment's first-in-order-wins resolution
manufactures it; no agent models anything.

**FINDING: the practical criterion says no emergence anywhere.** S_macro is
-0.505 at 3x3 zeroth-order, since the agents' own choices predict the
macro signal better than its past does, and exactly 0 under ToM.

**FINDING: at 5x5 the estimate is mostly bias.** Zeroth-order synergy is
0.524 bits against 0.390 shuffled -- with 5 box values plus "done" per
agent, 200 trials of plug-in counts no longer separate signal from noise.
Riedl's groups ran 200 replications per condition too, with N = 10 agents.

Structure: `pid()` is two-source Williams-Beer on plug-in counts;
`logs()` collects padded choice vectors from ex01's `trial()`.
"""

from __future__ import annotations

import collections
import itertools
import math
import pathlib
import random

from harness import practice

HERE = pathlib.Path(__file__).resolve().parent
SIM = practice.load_module(next(HERE.glob("ex01_*.py")))
H = 3


def mi(pairs):
    n, joint = len(pairs), collections.Counter(pairs)
    px, py = collections.Counter(x for x, _ in pairs), collections.Counter(y for _, y in pairs)
    return sum(c / n * math.log2(c * n / (px[x] * py[y])) for (x, y), c in joint.items())


def specific(samples, y, k):
    """I_spec(Y=y; S_k) = sum_s p(s|y) log p(y|s)/p(y)."""
    p_y = sum(t == y for _, t in samples) / len(samples)
    given = collections.Counter(s[k] for s, t in samples if t == y)
    total = collections.Counter(s[k] for s, _ in samples)
    return sum(c / sum(given.values()) * math.log2(c / total[v] / p_y) for v, c in given.items())


def pid(samples):
    """(synergy, joint TDMI) for samples of ((a, b), target)."""
    n, targets = len(samples), collections.Counter(t for _, t in samples)
    red = sum(c / n * min(specific(samples, y, 0), specific(samples, y, 1)) for y, c in targets.items())
    ia, ib = mi([(s[0], t) for s, t in samples]), mi([(s[1], t) for s, t in samples])
    joint = mi(samples)
    return joint - ia - ib + red, joint


def logs(ref, n, tom):
    out = []
    for seed in range(SIM.TRIALS):
        rows = []
        SIM.trial(ref, n, n, tom, seed, log=rows)
        out.append((rows + [[-1] * n] * H)[:H])
    return out


def pair_samples(runs, i, j, steps=range(H - 1)):
    return [((r[t][i], r[t][j]), (r[t + 1][i], r[t + 1][j])) for r in runs for t in steps]


def synergy(runs, n, shuffle=False):
    rng, values = random.Random(0), []
    for i, j in itertools.combinations(range(n), 2):
        samples = pair_samples(runs, i, j)
        if shuffle:
            targets = [t for _, t in samples]
            rng.shuffle(targets)
            samples = [(s, t) for (s, _), t in zip(samples, targets)]
        values.append(pid(samples))
    return tuple(round(sum(v[k] for v in values) / len(values), 3) for k in (0, 1))


def s_macro(runs, n):
    steps = [(r[t], r[t + 1]) for r in runs for t in range(H - 1)]
    searching = lambda row: sum(x >= 0 for x in row)
    whole = mi([(searching(a), searching(b)) for a, b in steps])
    parts = sum(mi([(a[k], searching(b)) for a, b in steps]) for k in range(n))
    return round(whole - parts, 3)


def solve():
    ref = SIM.reference()
    runs = {(n, tom): logs(ref, n, tom) for n in (3, 5) for tom in (False, True)}
    zeroth3 = runs[(3, False)]
    return {
        "syn": {key: synergy(r, key[0]) for key, r in runs.items()},
        "shuffled": {n: synergy(runs[(n, False)], n, shuffle=True)[0] for n in (3, 5)},
        "macro": {key: s_macro(r, key[0]) for key, r in runs.items() if key[0] == 3},
        "by_step": [round(pid(pair_samples(zeroth3, 0, 1, [t]))[0], 3) for t in range(H - 1)],
    }


def verify(result):
    syn, shuffled = result["syn"], result["shuffled"]
    return [
        practice.Check(
            "ANSWER: yes -- and only there",
            all([syn[(3, False)] == (0.48, 0.965), shuffled[3] < 0.1,
                 syn[(3, True)] == (0.0, 0.0)]),
            f"3x3 zeroth-order: synergy {syn[(3, False)][0]} of {syn[(3, False)][1]} bits "
            f"of TDMI, {shuffled[3]} with the target shuffled; primed ToM "
            f"{syn[(3, True)]} -- every variable is constant",
        ),
        practice.Check(
            "FINDING: the synergy is the collision rule, not coordination",
            result["by_step"][0] > 0.8 and result["by_step"][1] == 0.0,
            f"agents 0 and 1: {result['by_step'][0]} bits at turn 0 -> 1 and "
            f"{result['by_step'][1]} after; whether one keeps searching depends on "
            "whether both picked the same box",
        ),
        practice.Check(
            "FINDING: the practical criterion says no emergence anywhere",
            result["macro"][(3, False)] < 0 and result["macro"][(3, True)] == 0.0,
            f"S_macro = {result['macro'][(3, False)]} zeroth-order and "
            f"{result['macro'][(3, True)]} under ToM at 3x3",
        ),
        practice.Check(
            "FINDING: at 5x5 the estimate is mostly bias",
            shuffled[5] > 0.7 * syn[(5, False)][0],
            f"zeroth-order synergy {syn[(5, False)][0]} against {shuffled[5]} shuffled",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
