"""Exercise 3 — the head's output scale ranges 4.9x with lambda fixed, which is what the RMSNorm was for.

    Read Section 3 of the DIFF V1 paper (arXiv:2410.05258) and Section 2 of the
    DIFF V2 Hugging Face blog. In two sentences, explain why the V1 per-head
    RMSNorm was necessary and why V2 could remove it without causing training
    divergence.

Reading of the exercise: the explanation is written as a measurement against the
lesson's own attention, because the quantity both papers are arguing about --
the magnitude a differential head hands to the residual stream -- is computable
from `diff_attention` directly. Heads are simulated by varying the *correlation*
between the two branches, which is the thing that differs from head to head in a
trained model and which `lambda` does not control.

**ANSWER: at a fixed lambda = 0.8, a head's output norm ranges 4.9x depending on
how correlated its two branches are.**

    branch correlation   |out|   vs standard
          0.00           0.655       0.98x
          0.30           0.615       0.92x
          0.60           0.518       0.77x
          0.90           0.273       0.41x
          0.99           0.149       0.22x
          1.00           0.134       0.20x

A head whose two branches have learned the same thing emits a fifth of what an
uncorrelated head emits, through the same lambda and into the same residual
stream. That spread is the argument for a per-head RMSNorm in one table:
normalise each head's output and the residual stream stops depending on a
quantity nobody is controlling.

**MECHANISM: the attention weights no longer sum to one.** `A1 - lam * A2` has
row sum exactly **1 - lam**, so the output is a `(1 - lam)`-weighted average plus
a contrast term. At lam = 1 the weights sum to **0.000** and the head emits a
pure difference with no mean component at all. Standard attention's row sum is
1.0 by construction and its output scale is fixed; differential attention's is
not fixed by anything.

**FINDING: V2's removal is a width change, not a normalisation change.** V1
halves `d_head` to 64 and concatenates two branches back to 4096; V2 keeps
`d_head` at 128 and hands the output projection **8192** inputs. The projection
that follows is twice as wide and is itself trained, so it can absorb a per-head
scale that V1's fixed-width concatenation could not.

**FINDING: lambda cannot do the RMSNorm's job, because it is the wrong knob.**
Sweeping lambda moves the row sum and the scale together -- there is no setting
at which a correlated head and an uncorrelated one emit the same magnitude, since
the ratio between them is **4.9x** at lam = 0.8 and **1.0x** only at lam = 0,
where there is no differential attention left.

Structure: `head` builds one (branch-1, branch-2) pair at a given correlation;
`scale` runs the lesson's own weighting and reports the output norm.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "16-differential-attention-v2"
TOKENS, SIGNAL_POS, SIGNAL, VALUE_DIM = 256, 100, 4.0, 16
LAMBDA, NOISE, SEEDS = 0.8, 0.5, 20
CORRELATIONS = (0.0, 0.3, 0.6, 0.9, 0.99, 1.0)


def head(ref, rho, seed):
    """One head: branch 2 shares a fraction `rho` of branch 1's logits."""
    rng = random.Random(seed)
    first = [rng.gauss(0, NOISE) for _ in range(TOKENS)]
    first[SIGNAL_POS] = SIGNAL
    second = [rho * x + math.sqrt(max(0.0, 1 - rho * rho)) * rng.gauss(0, NOISE)
              for x in first]
    values = [[rng.gauss(0, 1) for _ in range(VALUE_DIM)] for _ in range(TOKENS)]
    return ref.softmax_row(first), ref.softmax_row(second), values


def apply(weights, values):
    return [sum(weights[j] * values[j][c] for j in range(len(values)))
            for c in range(VALUE_DIM)]


def norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def scale(ref, rho, lam=LAMBDA):
    """Output norm of a differential head at this branch correlation, and its row sum."""
    norms, ratios, sums = [], [], []
    for seed in range(SEEDS):
        first, second, values = head(ref, rho, seed)
        weights = [a - lam * b for a, b in zip(first, second)]
        norms.append(norm(apply(weights, values)))
        ratios.append(norms[-1] / norm(apply(first, values)))
        sums.append(sum(weights))
    return {"norm": statistics.fmean(norms), "vs_standard": statistics.fmean(ratios),
            "row_sum": statistics.fmean(sums)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {rho: scale(ref, rho) for rho in CORRELATIONS}
    lambdas = {lam: scale(ref, 0.0, lam)["row_sum"] for lam in (0.0, 0.5, 0.8, 1.0)}
    at_zero = {rho: scale(ref, rho, 0.0)["vs_standard"] for rho in (0.0, 1.0)}
    hidden, heads, head_dim = 4096, 32, 128
    return {
        "rows": rows,
        "spread": max(r["norm"] for r in rows.values()) / min(r["norm"] for r in rows.values()),
        "row_sums": lambdas,
        "at_zero_spread": max(at_zero.values()) / min(at_zero.values()),
        "widths": {"v1": 2 * heads * (head_dim // 2), "v2": 2 * heads * head_dim,
                   "baseline": hidden},
    }


def column(rows, field, fmt):
    return ", ".join(f"rho={rho} {format(row[field], fmt)}" for rho, row in rows.items())


def verify(result):
    rows, widths = result["rows"], result["widths"]
    return [
        practice.Check(
            "ANSWER: at a fixed lambda the head's output norm ranges 4.9x with branch correlation",
            result["spread"] > 4 and rows[1.0]["vs_standard"] < 0.25 < rows[0.0]["vs_standard"],
            f"at lambda = {LAMBDA} the output norm is " + column(rows, "norm", ".3f")
            + ", which against standard attention's output is " + column(rows, "vs_standard", ".2f")
            + f"x. A head whose two branches have learned the same thing emits "
            f"{rows[1.0]['vs_standard'] / rows[0.0]['vs_standard']:.2f} of what an uncorrelated "
            f"head emits, through the same lambda and into the same residual stream -- a spread "
            f"of {result['spread']:.1f}x that nothing in the architecture controls",
        ),
        practice.Check(
            "MECHANISM: the attention weights no longer sum to one",
            all(abs(s - (1 - lam)) < 1e-9 for lam, s in result["row_sums"].items()),
            "A1 - lam * A2 has row sum exactly 1 - lam: "
            + ", ".join(f"lam={lam} sum {s:.3f}" for lam, s in result["row_sums"].items())
            + ". So the output is a (1 - lam)-weighted average plus a contrast term, and at "
            "lam = 1 the weights sum to zero and the head emits a pure difference with no mean "
            "component at all. Standard attention's row sum is 1.0 by construction and its output "
            "scale is fixed; this one is not fixed by anything",
        ),
        practice.Check(
            "FINDING: V2's removal is a width change, not a normalisation change",
            widths["v1"] == widths["baseline"] and widths["v2"] == 2 * widths["baseline"],
            f"V1 halves d_head to 64 and concatenates two branches back to {widths['v1']:,} -- "
            f"exactly the baseline's hidden size, so the output projection that follows is the "
            f"baseline's and cannot absorb anything. V2 keeps d_head at 128 and hands the "
            f"projection {widths['v2']:,} inputs, twice as wide and itself trained, which is "
            "where a per-head scale can go instead of into an explicit RMSNorm",
        ),
        practice.Check(
            "FINDING: lambda cannot do the RMSNorm's job, because it is the wrong knob",
            abs(result["at_zero_spread"] - 1.0) < 1e-6 < result["spread"] - 1,
            f"sweeping lambda moves the row sum and the scale together. At lambda = 0 every head "
            f"emits exactly the standard output, a spread of {result['at_zero_spread']:.3f}x, and "
            f"there is no differential attention left; at lambda = {LAMBDA} the spread is "
            f"{result['spread']:.1f}x. No setting makes a correlated head and an uncorrelated one "
            "emit the same magnitude, which is what a per-head normalisation is for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
