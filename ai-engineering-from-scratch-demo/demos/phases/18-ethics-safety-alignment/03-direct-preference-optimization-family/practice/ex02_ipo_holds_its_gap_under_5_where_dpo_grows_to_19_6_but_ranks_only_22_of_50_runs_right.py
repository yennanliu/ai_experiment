"""Exercise 2 — IPO holds its gap under 5 where DPO's grows to 19.6, but ranks only 22 of 50 runs right.

    Modify the preference data so that all pairs have equal strength. Which of
    the six methods is most robust? Which degrades? Explain IPO's advantage here.

Reading of the exercise: "equal strength" swaps `sample_pref_pair` for one where
the better action of every pair wins with the same probability p. p = 0.721 is
the mean strength of the 6 action pairs in the original data, so only the
*variation* in strength is removed. `main()` then runs unchanged over 50 seeds.
Robust means the final policy still orders all 4 actions by true utility.
Degrades means pi(best action) falls against the original data. IPO's advantage
is tested at the extreme p = 1, where DPO's closed-form gap is infinite.

**ANSWER: DPO and BPO are the most robust; every pairwise method degrades, DPO
the most.**

| method | ranks right, original | ranks right, equal | pi(best), original | pi(best), equal |
|---|---:|---:|---:|---:|
| DPO | 50/50 | 50/50 | 0.659 | 0.572 |
| IPO | 21/50 | 22/50 | 0.721 | 0.656 |
| BPO | 50/50 | 50/50 | 0.499 | 0.452 |
| SimPO | 40/50 | 44/50 | 0.492 | 0.443 |
| KTO | 49/50 | 49/50 | 0.683 | 0.683 |
| ORPO | 38/50 | 32/50 | 0.399 | 0.360 |

KTO does not change at all: it trains on unpaired labels, so pair data never
reaches it. ORPO loses the most rankings (38 to 32).

**FINDING: no loss reads the strength.** `train_dpo` unpacks `strength` and
never uses it again; SimPO, ORPO and KTO discard it as `_`. Strength enters only
through how often each pair flips. Equal strength just removes information
about action 1, which had the strongest pairs.

**FINDING: IPO's advantage is a bounded gap, and in this toy it is the worst
ranker.** With deterministic pairs (p = 1) the mean chosen-minus-rejected
log-ratio gap after 2,000 / 8,000 / 32,000 steps is 2.53 / 8.25 / 19.64 for
DPO and 12.26 at 32,000 for BPO. IPO stays at 3.95 / 4.36 / 3.76, under its
target 1/(2 beta) = 5. The squared loss has a finite optimum, while
-log sigmoid keeps paying for a wider gap. But IPO orders the actions right
in only 22 of 50 equal-strength runs. Its per-pair gradient scale
2 x (gap - 5) starts at -10, so at lr 0.05 the last few pairs decide the final
policy. At lr 0.005 IPO ranks 46 of 50 original-data runs right.

Structure: `run()` is the shipped `main()` with the RNG and pair sampler
swapped (restored after); `gap_growth()` calls `train_dpo` directly.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "03-direct-preference-optimization-family"
SEEDS, METHODS = range(50), ("DPO", "IPO", "BPO", "SimPO", "KTO", "ORPO")
STEPS = (2000, 8000, 32000)
TABLE = {"DPO": ((50, 0.659), (50, 0.572)), "IPO": ((21, 0.721), (22, 0.656)),
         "BPO": ((50, 0.499), (50, 0.452)), "SimPO": ((40, 0.492), (44, 0.443)),
         "KTO": ((49, 0.683), (49, 0.683)), "ORPO": ((38, 0.399), (32, 0.36))}
GROWTH = {"dpo": [2.53, 8.25, 19.64], "ipo": [3.95, 4.36, 3.76], "bpo": [2.09, 5.81, 12.26]}


def equal_sampler(ref, p):
    """sample_pref_pair with every pair at strength p: same RNG draws, so KTO's labels match."""
    u = ref.TRUE_UTILITY

    def sample():
        i, j = ref.random.sample(range(ref.N_ACTIONS), 2)
        hi, lo = (i, j) if u[i] > u[j] else (j, i)
        return (hi, lo, p) if ref.random.random() < p else (lo, hi, 1 - p)

    return sample


def run(ref, seed, sampler, ipo_lr):
    """Shipped main(), pair sampler and IPO's lr swappable: the final probs of every method."""
    probs, saved = {}, (ref.random, ref.report, ref.sample_pref_pair, ref.train_dpo)
    ref.random, ref.sample_pref_pair = random.Random(seed), sampler or saved[2]
    ref.report = lambda name, pi: probs.__setitem__(name, ref.softmax(pi.logits))
    ref.train_dpo = lambda pairs, variant: saved[3](pairs, variant=variant,
                                                    lr=ipo_lr if variant == "ipo" else 0.05)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.main()
    finally:
        ref.random, ref.report, ref.sample_pref_pair, ref.train_dpo = saved
    return probs


def sweep(ref, sampler, ipo_lr=0.05):
    """Per method over SEEDS: (runs ranking all 4 actions right, mean pi(best action))."""
    order = sorted(range(ref.N_ACTIONS), key=lambda a: -ref.TRUE_UTILITY[a])
    runs = [run(ref, s, sampler, ipo_lr) for s in SEEDS]
    return {m: (sum(sorted(range(4), key=lambda a: -r[m][a]) == order for r in runs),
                round(sum(r[m][order[0]] for r in runs) / len(runs), 3)) for m in METHODS}


def gap_growth(ref, variant):
    """Deterministic pairs (p = 1): mean log-ratio gap chosen minus rejected, by step budget."""
    out, base = [], ref.logsoftmax(ref.make_policy_and_ref()[1].logits)
    for steps in STEPS:
        ref.random = random.Random(1)
        pairs = [equal_sampler(ref, 1.0)() for _ in range(500)]
        pi = ref.train_dpo(pairs, variant=variant, steps=steps)
        d = [a - b for a, b in zip(ref.logsoftmax(pi.logits), base)]
        out.append(round(sum(d[w] - d[lo] for w, lo, _ in pairs) / len(pairs), 2))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    u, saved = ref.TRUE_UTILITY, ref.random
    strengths = [ref.sigmoid(abs(u[i] - u[j])) for i in range(4) for j in range(i + 1, 4)]
    p = round(sum(strengths) / len(strengths), 3)
    try:
        growth = {v: gap_growth(ref, v) for v in GROWTH}
    finally:
        ref.random = saved
    src = inspect.getsource(ref.train_dpo)
    return {"p": p, "orig": sweep(ref, None), "equal": sweep(ref, equal_sampler(ref, p)),
            "growth": growth, "strength_uses": src.count("strength"), "slow_ipo": sweep(ref, None, 0.005)["IPO"][0],
            "ipo_grad": "g_margin = 2 * diff" in src,
            "others": [inspect.getsource(f).count("strength")
                       for f in (ref.train_simpo, ref.train_orpo, ref.train_kto)],
            "target": 1 / (2 * inspect.signature(ref.train_dpo).parameters["beta"].default)}


def verify(result):
    orig, equal, growth = result["orig"], result["equal"], result["growth"]
    drops = {m: round(orig[m][1] - equal[m][1], 3) for m in METHODS}
    ranks = sorted(v[0] for v in equal.values())
    return [
        practice.Check(
            "ANSWER: DPO and BPO are the most robust; every pairwise method degrades, DPO the most",
            all([result["p"] == 0.721, {m: (orig[m], equal[m]) for m in METHODS} == TABLE,
                 max(drops, key=drops.get) == "DPO", min(drops.values()) == drops["KTO"] == 0]),
            f"at p = {result['p']}, (ranks right of {len(SEEDS)}, mean pi(best)) original vs "
            f"equal: { {m: (orig[m], equal[m]) for m in METHODS} }; pi(best) drop {drops}",
        ),
        practice.Check(
            "FINDING: no loss reads the strength",
            (result["strength_uses"], result["others"]) == (1, [0, 0, 0]),
            f"'strength' appears {result['strength_uses']}x in train_dpo (the unpack), "
            f"{result['others']}x in train_simpo / train_orpo / train_kto",
        ),
        practice.Check(
            "FINDING: IPO's advantage is a bounded gap, and in this toy it is the worst ranker",
            all([growth == GROWTH, max(growth["ipo"]) < result["target"] == 5.0,
                 result["ipo_grad"],
                 2 * (0 - result["target"]) == -10, ranks[0] == equal["IPO"][0] < ranks[1],
                 result["slow_ipo"] == 46]),
            f"p = 1 mean log-ratio gap at {STEPS} steps: {growth}; IPO target "
            f"{result['target']}; IPO ranks right {equal['IPO'][0]}/50 equal-strength, "
            f"{result['slow_ipo']}/50 original at lr 0.005",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
