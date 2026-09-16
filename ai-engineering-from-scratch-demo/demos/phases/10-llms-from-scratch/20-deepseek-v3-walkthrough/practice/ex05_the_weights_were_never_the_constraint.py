"""Exercise 5 — FP8 saves 619 GB of weights across 2,048 GPUs, which is 0.30 GB each.

    DeepSeek-V3 uses FP8 training for most operations. Compute the memory savings
    of FP8 vs BF16 for storing the 671B weights. How does this intersect with the
    14.8T-token training budget?

Reading of the exercise: the weight figure uses the lesson's own
`compute_totals` total rather than the round 671B, so the saving is measured on
the model the calculator describes. The intersection with the token budget is
then computed the only way it can be -- 6ND against the published GPU-hours --
because weight *storage* and a token *count* do not otherwise meet.

**ANSWER: 1,238 GB at BF16 against 619 GB at FP8, a saving of 619 GB -- and
across 2,048 H800s that is 0.30 GB per GPU.**

    precision   weights      per GPU (2,048)
    BF16        1,238 GB        0.60 GB
    FP8           619 GB        0.30 GB

The weights were never the constraint. What FP8 buys on a 2,048-GPU cluster is
not the 0.3 GB per card -- it is the halved bandwidth on every tensor that
crosses a link, and the halved activation and gradient traffic, neither of which
the calculator models.

**MECHANISM: the token budget meets the precision through compute, not
storage.** `6ND` with 30.36B active parameters over 14.8T tokens is
**2.70e24 FLOPs**, and the published 2.788M H800-hours at 989 TFLOPs implies an
MFU of **27.2%**. FP8 raises the achievable FLOPs per second rather than lowering
the FLOPs required, so it moves the denominator of that fraction and leaves the
numerator alone.

**FINDING: the calculator has no dtype anywhere.** `compute_totals` returns a
parameter count and a `kv_cache_bytes` that hard-codes `* 2` -- BF16 -- with no
argument to change it. The cache the lesson reports is the only byte count in the
module, and it cannot be asked for the FP8 number the exercise is about.

**FINDING: at 14.8T tokens each parameter sees 22,000 tokens.** 14.8e12 over
664.5B total is **22.3**, and over 30.36B active it is **487**. The
Chinchilla-optimal ratio is about 20 tokens per parameter, so DeepSeek-V3 is
trained at roughly Chinchilla on its *total* count and 24x past it on its active
one -- which is the sparse model's actual bargain, and is a token-budget question
the exercise's framing does not reach.

Structure: `weights` prices the model at one dtype; `compute_budget` runs 6ND
against the published hours.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "20-deepseek-v3-walkthrough"
TOKENS, GPUS, GPU_HOURS, PEAK = 14.8e12, 2048, 2.788e6, 989e12
GIB, CHINCHILLA = 1024 ** 3, 20


def weights(total, dtype_bytes):
    return {"gb": total * dtype_bytes / GIB, "per_gpu": total * dtype_bytes / GPUS / GIB}


def compute_budget(active):
    flops = 6 * active * TOKENS
    return {"flops": flops, "mfu": flops / (GPU_HOURS * 3600 * PEAK)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = dict(ref.DEEPSEEK_V3)
    report = ref.compute_totals(cfg)
    arms = {"BF16": weights(report.total, 2), "FP8": weights(report.total, 1)}
    cache_2k = ref.compute_totals(cfg, 2048).kv_cache_bytes
    cache_4k = ref.compute_totals(cfg, 4096).kv_cache_bytes
    return {
        "arms": arms,
        "saving_gb": arms["BF16"]["gb"] - arms["FP8"]["gb"],
        "saving_per_gpu": arms["BF16"]["per_gpu"] - arms["FP8"]["per_gpu"],
        "budget": compute_budget(report.active),
        "bytes_per_token_per_layer": cache_4k / 4096 / cfg["num_hidden_layers"],
        "cache_linear": cache_4k == 2 * cache_2k,
        "tokens_per_param": {"total": TOKENS / report.total, "active": TOKENS / report.active},
        "total": report.total,
        "active": report.active,
    }


def verify(result):
    arms, budget = result["arms"], result["budget"]
    ratio = result["tokens_per_param"]
    return [
        practice.Check(
            "ANSWER: 619 GB saved across 2,048 GPUs is 0.30 GB each",
            abs(result["saving_gb"] / arms["BF16"]["gb"] - 0.5) < 1e-9
            and result["saving_per_gpu"] < 1.0,
            f"the calculator's {result['total'] / 1e9:.1f}B parameters are "
            f"{arms['BF16']['gb']:,.0f} GB at BF16 and {arms['FP8']['gb']:,.0f} GB at FP8, a "
            f"saving of {result['saving_gb']:,.0f} GB -- exactly half, since the count does not "
            f"depend on the dtype. Across {GPUS:,} H800s that is "
            f"{arms['BF16']['per_gpu']:.2f} GB against {arms['FP8']['per_gpu']:.2f}, so the "
            "weights were never the constraint",
        ),
        practice.Check(
            "MECHANISM: the token budget meets the precision through compute, not storage",
            0.2 < budget["mfu"] < 0.5,
            f"6ND with {result['active'] / 1e9:.2f}B active parameters over "
            f"{TOKENS:.1e} tokens is {budget['flops']:.2e} FLOPs, and the published "
            f"{GPU_HOURS:.3e} H800-hours at {PEAK / 1e12:.0f} TFLOPs implies an MFU of "
            f"{budget['mfu']:.1%}. FP8 raises the achievable FLOPs per second rather than "
            "lowering the FLOPs required, so it moves the denominator of that fraction and leaves "
            "the numerator alone",
        ),
        practice.Check(
            "FINDING: the calculator has no dtype anywhere",
            result["cache_linear"] and abs(result["bytes_per_token_per_layer"] - 1024) < 1,
            f"compute_totals returns a parameter count and a kv_cache_bytes that hard-codes a "
            f"factor of 2 for BF16, with no argument to change it: the cache works out to "
            f"{result['bytes_per_token_per_layer']:.0f} bytes per token per layer at every "
            "context length. That is the only byte count in the module, and it cannot be asked "
            "for the FP8 number this exercise is about",
        ),
        practice.Check(
            "FINDING: at 14.8T tokens each parameter sees 22 tokens, or 487 if it is active",
            ratio["total"] < 2 * CHINCHILLA < ratio["active"],
            f"{TOKENS:.1e} over the {result['total'] / 1e9:.1f}B total is "
            f"{ratio['total']:.1f} tokens per parameter and over the "
            f"{result['active'] / 1e9:.2f}B active it is {ratio['active']:.0f}. The "
            f"Chinchilla-optimal ratio is about {CHINCHILLA}, so V3 is trained near Chinchilla on "
            f"its total count and {ratio['active'] / CHINCHILLA:.0f}x past it on its active one "
            "-- which is the sparse model's actual bargain, and a token-budget question the "
            "exercise's framing does not reach",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
