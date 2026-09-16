"""Exercise 3 — doubling the experts doubles the total and moves the active count by 0.4%.

    Compare DeepSeek-V3's (256 experts, top-8) routing to a hypothetical (512
    experts, top-8) variant. Total parameters grow; active parameters stay the
    same. What does the extra expert capacity buy in theory, and what costs at
    inference?

Reading of the exercise: both variants go through the lesson's own
`compute_totals`, which reports `total`, `active` and `active_ratio` from one
config, so the claim "total grows, active stays the same" is checkable directly.
A 1024-expert variant is run beside them because the exercise's question is about
a direction, and two points do not show whether anything saturates.

**ANSWER: the total grows 1.98x and the active count moves 0.11B -- 0.4%.**

    experts   total      active    active ratio   router / layer
      256     664.5B     30.36B       4.57%          1.84M
      512    1318.6B     30.47B       2.31%          3.67M
     1024    2626.6B     30.68B       1.17%          7.34M

"Active parameters stay the same" is true to within the router, which is the one
per-layer term that grows with the expert *count* rather than with the top-k.

**MECHANISM: what the extra capacity buys is storage, and storage is what it
costs.** Every expert must be resident somewhere for the router to be able to
choose it, so doubling the count doubles the weight memory of the deployment --
**654B more parameters, 0.6 TB at FP8** -- while the FLOPs per token are
unchanged. The trade is memory for capacity at constant compute, which is the
only trade MoE offers and the only one this calculator can price.

**FINDING: the router is the one term that scales with the wrong thing.**
`router_params` is `hidden * n_experts`, so it doubles with the expert count and
appears in the *active* path: 1.84M to 3.67M to 7.34M per layer, and 58 MoE
layers of it. At 1024 experts the routers alone are 425.7M of active parameters,
**1.4%** of the active total, spent deciding which 8 of 1024 to use.

**FINDING: the active ratio is not a measure of sparsity the exercise can
interpret.** It falls 4.57% to 1.17% across the sweep, and every point of that
fall is the denominator growing. The numerator is essentially fixed, so the
ratio measures how many experts were bought, not how selectively they are used.

Structure: `variant` runs the lesson's own `compute_totals` at one expert count;
`router_cost` prices the routing term across the active path.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "20-deepseek-v3-walkthrough"
COUNTS = (256, 512, 1024)
FP8_BYTES, TIB = 1, 1024 ** 4


def variant(ref, cfg, experts):
    report = ref.compute_totals(dict(cfg, num_experts=experts))
    return {"total": report.total, "active": report.active, "ratio": report.active_ratio,
            "router": ref.router_params(cfg["hidden_size"], experts)}


def router_cost(cfg, experts, ref):
    """What routing costs on the active path, across the MoE layers."""
    moe_layers = cfg["num_hidden_layers"] - cfg["first_k_dense_layers"]
    return ref.router_params(cfg["hidden_size"], experts) * moe_layers


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = dict(ref.DEEPSEEK_V3)
    rows = {count: variant(ref, cfg, count) for count in COUNTS}
    base, doubled = rows[256], rows[512]
    return {
        "rows": rows,
        "total_growth": doubled["total"] / base["total"],
        "active_growth": doubled["active"] / base["active"],
        "extra_params": doubled["total"] - base["total"],
        "extra_tb": (doubled["total"] - base["total"]) * FP8_BYTES / TIB,
        "routers": {count: router_cost(cfg, count, ref) for count in COUNTS},
        "router_share": router_cost(cfg, 1024, ref) / rows[1024]["active"],
        "top_k": cfg["num_experts_per_tok"],
    }


def column(rows, field, fmt):
    return ", ".join(f"{count} {format(row[field], fmt)}" for count, row in rows.items())


def verify(result):
    rows, routers = result["rows"], result["routers"]
    base, doubled, quad = rows[256], rows[512], rows[1024]
    return [
        practice.Check(
            "ANSWER: the total grows 1.98x and the active count moves 0.4%",
            abs(result["total_growth"] - 2) < 0.05 and result["active_growth"] < 1.01,
            "totals are " + column(rows, "total", ",.0f")
            + " on active counts of " + column(rows, "active", ",.0f")
            + f", so doubling the experts multiplies the total by "
            f"{result['total_growth']:.2f} and the active count by "
            f"{result['active_growth']:.4f} -- {doubled['active'] - base['active']:,.0f} "
            f"parameters, {result['active_growth'] - 1:.1%}. 'Active parameters stay the same' is "
            "true to within the router",
        ),
        practice.Check(
            "MECHANISM: what the extra capacity buys is storage, and storage is what it costs",
            result["extra_params"] > 600e9,
            f"every expert must be resident somewhere for the router to be able to choose it, so "
            f"doubling the count adds {result['extra_params'] / 1e9:.0f}B parameters -- "
            f"{result['extra_tb']:.1f} TB at FP8 -- while the FLOPs per token are unchanged, "
            f"because top-k is still {result['top_k']}. Memory for capacity at constant compute "
            "is the only trade MoE offers and the only one this calculator can price",
        ),
        practice.Check(
            "FINDING: the router is the one term that scales with the wrong thing",
            routers[1024] == 4 * routers[256] and result["router_share"] > 0.01,
            f"router_params is hidden * n_experts, so it doubles with the expert count and "
            f"appears in the active path: "
            + ", ".join(f"{count} experts {rows[count]['router'] / 1e6:.2f}M per layer"
                        for count in COUNTS)
            + f". Across the MoE layers that is {routers[1024] / 1e6:.1f}M at 1024 experts, "
            f"{result['router_share']:.1%} of the active total, spent deciding which "
            f"{result['top_k']} of 1024 to use",
        ),
        practice.Check(
            "FINDING: the active ratio measures how many experts were bought",
            quad["ratio"] < base["ratio"] / 3,
            "the active ratio falls " + column(rows, "ratio", ".2%")
            + f" across the sweep, and every point of that fall is the denominator growing: the "
            f"numerator moves from {base['active'] / 1e9:.2f}B to {quad['active'] / 1e9:.2f}B "
            "while the total moves 4x. The ratio measures how many experts were bought, not how "
            "selectively they are used",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
