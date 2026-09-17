"""Exercise 2 — the counter gives 0.65B for DeepSeek's own shape against the 14B it is asked to explain.

    Compute the parameter overhead for a dense 70B model (hidden 8192, 80
    layers) with D=1 MTP module. Compare to the DeepSeek-V3 reported 14B
    overhead. Explain why DeepSeek's number is higher: the MTP transformer block
    inherits the same MoE structure, inflating the per-module parameter count.

Reading of the exercise: the dense number comes from the lesson's own
`count_parameters`, and the explanation is then *checked* rather than repeated --
the same counter is run on DeepSeek's own shape, and the MoE term the exercise
names is added by hand to see whether it accounts for the gap.

**ANSWER: 1.04B for the dense 70B -- 1.3% overhead -- and the counter cannot
produce DeepSeek's number for DeepSeek.**

    model                  main     per MTP module   overhead
    70B dense (8192/28672) 78.9B        1.040B         1.3%
    DeepSeek-V3 shape      37.6B        0.653B         1.7%

Run on DeepSeek's own `hidden=7168, ff=18432`, `count_parameters` reports
**0.653B** against the **14B** the exercise cites -- a factor of **21**. The
explanation the exercise supplies is correct and the counter it supplies cannot
express it.

**MECHANISM: `per_mtp` is `hidden*hidden + 4*hidden*hidden + 3*hidden*ff`, and
none of those terms is an expert count.** Adding DeepSeek's actual MoE MLP -- 256
routed experts plus 1 shared, each `3 * 7168 * 2048` = **44.0M** -- gives a
module of **11.58B**, which is the right order of magnitude. The 257 experts are
**97.8%** of it.

**FINDING: the dense overhead is small because the main model is large, not
because the module is.** A 70B's MTP module is 1.040B against 78.9B of backbone,
1.3%. The same module on the mini-GPT shape is **10.0M against 211.6M** --
**4.7%**, three and a half times the rate, and that is with a 128,000-token
embedding table carrying 98.3M of the mini-GPT's total. "~1-2% parameters for a
dense model" is a statement about 70B-scale models.

**FINDING: the module is one layer, and one layer of a 70B is 1.04B.** `per_mtp`
is exactly `hidden*hidden` more than a single decoder layer's
`4*hidden*hidden + 3*hidden*ff` -- the projection `M_k` that folds the previous
hidden and the next embedding together. So the overhead of D=1 MTP is
`(1 + 1/80)` layers, or **1.25%** of the backbone's layer stack, which is where
the 1.3% comes from.

Structure: `dense` runs the lesson's own counter on one shape; `moe_module`
prices the MTP block the way DeepSeek builds it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "18-multi-token-prediction"
SHAPES = {"70B dense": (8192, 28672, 80, 128000),
          "DeepSeek-V3 shape": (7168, 18432, 61, 129280),
          "mini GPT": (768, 3072, 12, 128000)}
REPORTED_MTP = 14e9
MOE_HIDDEN, MOE_FF, EXPERTS, SHARED = 7168, 2048, 256, 1


def dense(ref, hidden, ff, layers, vocab):
    report = ref.count_parameters(vocab=vocab, hidden=hidden, ff=ff, n_layers=layers, D=1)
    return {"main": report.main_total, "per_mtp": report.per_mtp,
            "overhead": report.mtp_total / report.main_total,
            "layer": report.main_attention_per_layer + report.main_mlp_per_layer,
            "layers": layers}


def moe_module():
    """The MTP block as DeepSeek builds it: the same MoE MLP the backbone layers use."""
    expert = 3 * MOE_HIDDEN * MOE_FF
    return {"expert": expert,
            "experts": expert * (EXPERTS + SHARED),
            "attention": 4 * MOE_HIDDEN * MOE_HIDDEN,
            "projection": MOE_HIDDEN * MOE_HIDDEN,
            "router": MOE_HIDDEN * EXPERTS}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: dense(ref, *shape) for name, shape in SHAPES.items()}
    block = moe_module()
    total = sum(block.values()) - block["expert"]
    return {
        "rows": rows,
        "moe": block,
        "moe_total": total,
        "expert_share": block["experts"] / total,
        "shortfall": REPORTED_MTP / rows["DeepSeek-V3 shape"]["per_mtp"],
        "reported": REPORTED_MTP,
    }


def verify(result):
    rows, moe = result["rows"], result["moe"]
    dense70, deep, mini = rows["70B dense"], rows["DeepSeek-V3 shape"], rows["mini GPT"]
    return [
        practice.Check(
            "ANSWER: 1.04B and 1.3% for the dense 70B, and 0.65B where 14B was expected",
            abs(dense70["overhead"] - 0.013) < 0.002 and result["shortfall"] > 15,
            f"count_parameters gives the 70B a {dense70['main'] / 1e9:.1f}B backbone and a "
            f"{dense70['per_mtp'] / 1e9:.3f}B MTP module -- {dense70['overhead']:.1%} overhead. "
            f"Run on DeepSeek's own hidden={MOE_HIDDEN}, ff=18432 it reports "
            f"{deep['per_mtp'] / 1e9:.3f}B against the {result['reported'] / 1e9:.0f}B the "
            f"exercise cites, a factor of {result['shortfall']:.0f}. The explanation the exercise "
            "supplies is correct and the counter it supplies cannot express it",
        ),
        practice.Check(
            "MECHANISM: per_mtp has no expert count in it",
            result["moe_total"] > 10e9 and result["expert_share"] > 0.9,
            f"per_mtp is hidden*hidden + 4*hidden*hidden + 3*hidden*ff, and none of those terms "
            f"is an expert count. Adding DeepSeek's actual MoE MLP -- {EXPERTS} routed experts "
            f"plus {SHARED} shared, each 3 x {MOE_HIDDEN} x {MOE_FF} = "
            f"{moe['expert'] / 1e6:.1f}M -- gives a module of "
            f"{result['moe_total'] / 1e9:.2f}B, the right order of magnitude, and the "
            f"{EXPERTS + SHARED} experts are {result['expert_share']:.1%} of it",
        ),
        practice.Check(
            "FINDING: the overhead is small because the main model is large",
            mini["overhead"] > 3 * dense70["overhead"],
            f"a 70B's MTP module is {dense70['per_mtp'] / 1e9:.3f}B against "
            f"{dense70['main'] / 1e9:.1f}B of backbone, {dense70['overhead']:.1%}. The same "
            f"module on the mini-GPT shape is {mini['per_mtp'] / 1e6:.1f}M against "
            f"{mini['main'] / 1e6:.1f}M -- {mini['overhead']:.1%}. 'About 1-2% of parameters for "
            "a dense model' is a statement about 70B-scale models and stops holding two orders "
            "of magnitude down",
        ),
        practice.Check(
            "MECHANISM: the module is one decoder layer plus one projection",
            dense70["per_mtp"] - dense70["layer"] == 8192 * 8192,
            f"per_mtp is exactly hidden*hidden more than one decoder layer's "
            f"{dense70['layer'] / 1e9:.3f}B -- the M_k projection that folds the previous hidden "
            f"state and the next embedding together. So D=1 MTP costs "
            f"{1 + 1 / dense70['layers']:.4f} layers out of {dense70['layers']}, which is "
            f"{(1 + 1 / dense70['layers']) / dense70['layers']:.2%} of the layer stack and where "
            f"the {dense70['overhead']:.1%} comes from",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
