"""Exercise 2 — 3.1% of experts is 4.9% of parameters, and the gap is everything that is not an expert.

    DeepSeek V3 uses 256 experts with top-8 routing. Compute the ratio of
    activated experts to total experts and compare to Mixtral 8x7B's top-2 of 8.
    What does the shift from sparse (25%) to denser sparse (3%) imply about
    capacity per FLOP?

Reading of the exercise: both ratios are computed, and then the quantity the
exercise is really asking about -- what fraction of the *model* is active per
token -- is read off the lesson's own `analyze`, which tracks `total_params` and
`active_params` separately. The two ratios are not the same number, and the
difference is the whole answer to "capacity per FLOP".

**ANSWER: 25.0% of experts against 3.1%, an 8x shift -- and 27.4% of parameters
against 4.9%, a 5.6x shift.**

    mixtral-8x7b     8 experts, top-2    25.00% of experts    46.6B total    12.7B active  (27.4%)
    deepseek-v3    256 experts, top-8     3.12% of experts   665.9B total    32.4B active   (4.9%)

The expert ratio overstates the sparsity in both models, because attention,
embeddings, norms and the shared expert are active on every token whatever the
router does.

**MECHANISM: capacity per FLOP is `total / active`, and it is what the shift
buys.** Mixtral holds 3.7 parameters for every one it spends; DeepSeek holds
**20.6**. Going from top-2-of-8 to top-8-of-256 multiplies stored capacity per
unit of compute by **5.6x** while *increasing* the number of experts consulted
per token from 2 to 8 -- finer experts, more of them, each smaller.

**FINDING: the expert got smaller, which is how the ratio moved.** DeepSeek's
`moe_intermediate_size` is **2048** against Mixtral's 14336 -- a 7x narrower
expert -- so 256 of DeepSeek's experts is **8.0x** Mixtral's 8 in
parameters, not 32x. The headline "256 experts" counts a different unit.

**FINDING: the floor is the dense part, and DeepSeek's is 11.9B.** Attention,
embeddings, norms, the router, the first three dense layers and the always-on
shared expert come to 11.9B active parameters before a single routed expert is
consulted -- **37%** of the 32.4B active. No routing decision can reduce it, so
the sparsity the router controls is bounded by what is left.

Structure: `shares` computes the expert ratio and the parameter ratio for one
model; `floor` is the part of the active count no router touches.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "14-open-models-architecture-walkthroughs"
MODELS = ("mixtral-8x7b", "deepseek-v3")


def shares(ref, name):
    """Both ratios for one MoE model: experts activated, and parameters activated."""
    config = ref.CONFIGS[name]
    breakdown = ref.analyze(name, config)
    return {
        "experts": config["num_experts"],
        "top_k": config["experts_per_token"],
        "expert_share": config["experts_per_token"] / config["num_experts"],
        "total": breakdown.total_params,
        "active": breakdown.active_params,
        "param_share": breakdown.active_params / breakdown.total_params,
        "capacity_per_flop": breakdown.total_params / breakdown.active_params,
        "expert_width": config.get("moe_intermediate_size", config["intermediate_size"]),
        "expert_params": ref.mlp_params(config["hidden_size"],
                                        config.get("moe_intermediate_size",
                                                   config["intermediate_size"]),
                                        config["activation"]),
    }


def floor(ref, name):
    """Active parameters no routing decision can reduce: everything that is not routed."""
    config = ref.CONFIGS[name]
    breakdown = ref.analyze(name, config)
    routed = ref.mlp_params(config["hidden_size"],
                            config.get("moe_intermediate_size", config["intermediate_size"]),
                            config["activation"]) * config["experts_per_token"]
    moe_layers = config["num_hidden_layers"] - config.get("first_dense_layers", 0)
    return breakdown.active_params - routed * moe_layers


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: shares(ref, name) for name in MODELS}
    return {"rows": rows, "floor": {name: floor(ref, name) for name in MODELS}}


def column(rows, field, fmt):
    return ", ".join(f"{name} {format(row[field], fmt)}" for name, row in rows.items())


def verify(result):
    rows = result["rows"]
    mixtral, deepseek = rows["mixtral-8x7b"], rows["deepseek-v3"]
    deep_floor = result["floor"]["deepseek-v3"]
    return [
        practice.Check(
            "ANSWER: 25.0% of experts against 3.1% -- but 27.4% of parameters against 4.9%",
            abs(mixtral["expert_share"] - 0.25) < 1e-9
            and deepseek["param_share"] > deepseek["expert_share"],
            "experts activated is " + column(rows, "expert_share", ".2%")
            + " and parameters activated is " + column(rows, "param_share", ".1%")
            + ", on totals of " + ", ".join(f"{n} {r['total'] / 1e9:.1f}B" for n, r in rows.items())
            + f". The expert ratio overstates the sparsity by "
            f"{deepseek['param_share'] / deepseek['expert_share']:.1f}x on DeepSeek and "
            f"{mixtral['param_share'] / mixtral['expert_share']:.1f}x on Mixtral, because "
            "attention, embeddings, norms and the shared expert are active on every token "
            "whatever the router does",
        ),
        practice.Check(
            "MECHANISM: capacity per FLOP is total/active, and it moves 5.6x",
            deepseek["capacity_per_flop"] > 5 * mixtral["capacity_per_flop"],
            "Mixtral holds " + column(rows, "capacity_per_flop", ".1f")
            + f" parameters for every one it spends, so the shift from top-2-of-8 to top-8-of-256 "
            f"multiplies stored capacity per unit of compute by "
            f"{deepseek['capacity_per_flop'] / mixtral['capacity_per_flop']:.1f}x while "
            f"*increasing* the experts consulted per token from {mixtral['top_k']} to "
            f"{deepseek['top_k']}. Finer experts, more of them, each smaller",
        ),
        practice.Check(
            "FINDING: the expert got smaller, which is how the ratio moved",
            deepseek["expert_width"] < mixtral["expert_width"] / 5,
            f"DeepSeek's moe_intermediate_size is {deepseek['expert_width']:,} against Mixtral's "
            f"{mixtral['expert_width']:,}, a {mixtral['expert_width'] / deepseek['expert_width']:.0f}x "
            f"narrower expert at {deepseek['expert_params'] / 1e6:.0f}M parameters against "
            f"{mixtral['expert_params'] / 1e6:.0f}M. So 256 of DeepSeek's experts is "
            f"{256 * deepseek['expert_params'] / (8 * mixtral['expert_params']):.1f}x Mixtral's 8 "
            "in parameters, not 32x -- the headline '256 experts' counts a different unit",
        ),
        practice.Check(
            "FINDING: the floor is the dense part, and DeepSeek's is 11.9B",
            deep_floor > 0.2 * deepseek["active"],
            f"attention, embeddings, norms, the router, the first three dense layers and the "
            f"always-on shared expert come to {deep_floor / 1e9:.1f}B active parameters before a "
            f"single routed expert is consulted -- {deep_floor / deepseek['active']:.0%} of the "
            f"{deepseek['active'] / 1e9:.1f}B active total. No routing decision can reduce it, so "
            "the sparsity the router controls is bounded by whatever is left over",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
