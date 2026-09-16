"""Exercise 3 — the gate's three outputs are identical whether the needle is far back or in the window.

    Implement the gate MLP. It takes the query as input and outputs three
    scalars. Show that the gate behaves sensibly: near-uniform weighting on
    random queries, heavy weight on the selected branch when the query hits a
    far-back block.

Reading of the exercise: the gate is the lesson's own -- `dot(q, Wg[i])` through
a sigmoid, three of them -- and both halves of "behaves sensibly" are measured
rather than asserted. The second half needs two sequences that differ only in
where the needle is, so the same query is run against a needle in block 1 and a
needle in the last block, with everything else held fixed.

**ANSWER: the first half is true and vacuous; the second half is impossible.**
On 60 random queries the three gates average **0.557, 0.563, 0.523** -- uniform,
because a sigmoid of a zero-mean logit averages 0.5 for any fixed `Wg`. With the
needle moved from block 1 to block 15, the gates are **identical to every
decimal**: `0.719, 0.767, 0.733` either way, while the selected blocks change
from `[1, 7, 11, 12]` to `[7, 11, 12, 15]`.

**MECHANISM: `gate(q, Wg)` takes the query and nothing else.** It never sees
`K`, `V`, the compressed scores, the selected blocks or the three branch
outputs. There is no path by which "the query hits a far-back block" can reach
it, because whether a query hits a far-back block is a property of the sequence
and the gate is a function of the query alone.

**FINDING: the three gates are not a distribution, and the output magnitude
rides on that.** Three independent sigmoids sum to **1.643** on average and
range from **0.313 to 2.968** across queries -- so `nsa_step`'s combination is
not a convex mixture, and the same three branch outputs scale by nearly **10x**
depending only on which query asked. A softmax over the three logits would fix
it and is one line away.

**FINDING: "near-uniform on random queries" is a property of the initialisation,
not of the gate.** `Wg` is drawn once and never trained here, so each logit is
`dot(q, w)` with `q` and `w` independent and zero-mean. The uniformity the
exercise asks you to confirm holds for *any* `Wg` at all, and would stop holding
the moment the gate were trained -- which is the only condition under which the
second half of the exercise could become true.

Structure: `random_gates` samples the gate on random queries; `placed` runs the
lesson's own `nsa_step` with the needle in a given block and reports both the
gates and the picks.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "17-native-sparse-attention"
TOKENS, DIM, BLOCK, TOP_K, WINDOW = 1024, 16, 64, 4, 256
QUERIES, SEED = 60, 7


def weights(rng):
    return [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(3)]


def random_gates(ref, gate_weights):
    """The gate on `QUERIES` random queries: per-branch means and the spread of the sum."""
    rows = []
    for seed in range(QUERIES):
        rng = random.Random(seed)
        rows.append(ref.gate([rng.gauss(0, 1) for _ in range(DIM)], gate_weights))
    sums = [sum(row) for row in rows]
    return {"means": [statistics.fmean(row[i] for row in rows) for i in range(3)],
            "sum_mean": statistics.fmean(sums), "sum_low": min(sums), "sum_high": max(sums)}


def placed(ref, gate_weights, needle_block):
    """One `nsa_step` with the needle in a chosen block, everything else held fixed."""
    rng = random.Random(5)
    keys, values, query = ref.synthesize_sequence(TOKENS, DIM, [needle_block], BLOCK, rng)
    config = ref.NSAConfig(l=BLOCK, k=TOP_K, W=WINDOW)
    _, info = ref.nsa_step(query, keys, values, gate_weights, config)
    return {"gates": info["gates"], "picks": info["selected_blocks"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gate_weights = weights(random.Random(SEED))
    last_block = TOKENS // BLOCK - 1
    far, near = placed(ref, gate_weights, 1), placed(ref, gate_weights, last_block)
    other = random_gates(ref, weights(random.Random(SEED + 1)))
    return {
        "random": random_gates(ref, gate_weights),
        "other_init": other,
        "far": far,
        "near": near,
        "identical": far["gates"] == near["gates"],
        "picks_differ": far["picks"] != near["picks"],
        "last_block": last_block,
    }


def means(row):
    return ", ".join(f"{m:.3f}" for m in row["means"])


def verify(result):
    random_row, far, near = result["random"], result["far"], result["near"]
    other = result["other_init"]
    uniform = all(abs(m - 0.5) < 0.1 for m in random_row["means"] + other["means"])
    return [
        practice.Check(
            "ANSWER: the gates are identical whether the needle is far back or in the window",
            result["identical"] and result["picks_differ"],
            "with the needle in block 1 the gates are "
            + ", ".join(f"{g:.3f}" for g in far["gates"])
            + f" and the selected blocks {far['picks']}; with it in block "
            f"{result['last_block']} the gates are "
            + ", ".join(f"{g:.3f}" for g in near["gates"])
            + f" and the selected blocks {near['picks']}. The picks move and the gates do not "
            "change in any decimal place -- the second half of 'behaves sensibly' does not "
            "happen",
        ),
        practice.Check(
            "MECHANISM: gate(q, Wg) takes the query and nothing else",
            far["gates"] == near["gates"],
            "the gate is three dot products of q with rows of Wg through a sigmoid. It never "
            "sees K, V, the compressed attention scores, the selected blocks or the three branch "
            "outputs, so there is no path by which 'the query hits a far-back block' can reach "
            "it: whether a query hits a far-back block is a property of the sequence, and the "
            "gate is a function of the query alone",
        ),
        practice.Check(
            "FINDING: the three gates are not a distribution, and the output magnitude rides on it",
            random_row["sum_high"] / random_row["sum_low"] > 5,
            f"three independent sigmoids sum to {random_row['sum_mean']:.3f} on average and range "
            f"from {random_row['sum_low']:.3f} to {random_row['sum_high']:.3f} across "
            f"{QUERIES} random queries, a factor of "
            f"{random_row['sum_high'] / random_row['sum_low']:.1f}. nsa_step combines the branches "
            "as g[0]*cmp + g[1]*sel + g[2]*win, so the same three branch outputs scale by that "
            "factor depending only on which query asked. A softmax over the three logits is one "
            "line away",
        ),
        practice.Check(
            "FINDING: near-uniform on random queries is a property of the initialisation",
            uniform,
            "the per-branch means are " + means(random_row) + " for one draw of Wg and "
            + means(other) + " for another. Wg is drawn once and never trained, so each logit is dot(q, w) with "
            "q and w independent and zero-mean, and a sigmoid of that averages 0.5 for any Wg at "
            "all. The uniformity the exercise asks you to confirm cannot fail, and would stop "
            "holding the moment the gate were trained -- which is the only condition under which "
            "the other half could become true",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
