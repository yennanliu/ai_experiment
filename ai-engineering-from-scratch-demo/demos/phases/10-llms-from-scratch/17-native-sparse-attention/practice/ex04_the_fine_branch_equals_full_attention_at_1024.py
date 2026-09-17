"""Exercise 4 — the fine branch is constant at k*l, so it equals full attention at exactly 1024 tokens.

    Compute the KV cache memory budget for an NSA-enabled 70B model at 128k
    context. KV heads are 8, head dim 128, BF16. Compare to full attention and
    to MLA (Phase 10 · 14 showed MLA's numbers). Identify the sequence length
    where NSA's fine-grained branch KV cache equals full attention.

Reading of the exercise: the cache is priced with the same formula Lesson 14
uses -- `2 * layers * kv_heads * head_dim * bytes * tokens` -- so NSA, full
attention and MLA are on one scale, and MLA's number comes from DeepSeek V3's
own `kv_lora_rank` of 512 as Lesson 14's config records it. The crossover is
solved rather than searched, because `count_nsa`'s three terms make it a linear
equation.

**ANSWER: at 128k context, 40.00 GB full against 1.09 GB for NSA -- and the
fine branch equals full attention at exactly N = 1024.**

    branch                    keys at 128k    GB
    full attention              131,072      40.00
    NSA compressed (N/64)         2,048       0.62
    NSA selected (k*l)            1,024       0.31
    NSA sliding window (W)          512       0.16
    NSA total                     3,584       1.09
    MLA latent (512 per layer)   131,072      10.00

The selected branch is `k * l` keys **whatever N is**, so "the sequence length
where NSA's fine-grained branch KV cache equals full attention" is the N at
which `N = k * l`: **1024** for `k=16, l=64`. Below it NSA's fine branch reads
more than the whole sequence.

**FINDING: NSA is 9.2x smaller than MLA at 128k, and 1.6x larger at 4k.** MLA's
cache is `2 * layers * 512 * bytes * N` -- linear in N with a small constant --
while NSA's is `N/l + k*l + W`, mostly constant. They cross at **N = 6,554**.
Two methods the exercise treats as alternatives are not comparable at a single
context length, and it picks one.

**MECHANISM: NSA caches the whole sequence and reads part of it; MLA caches a
compression of the whole sequence.** The 1.09 GB above is what NSA *reads per
query*, not what it stores -- the selected branch picks blocks from the full K
and V, which must still be resident. Stored, NSA is 40.00 GB plus the compressed
summaries: **40.62 GB**, more than full attention. The saving is bandwidth, not
capacity, and the exercise says "memory budget".

**FINDING: the compressed branch is the only term that grows, and it overtakes
the other two at N = 98,304.** At 4k context the fixed terms are 96% of the
per-query cost; at 128k they are 43%. NSA's asymptotic cost is `N/l`, which is
still linear in N -- a 64x smaller slope than full attention, and not a flat one.

Structure: `cache_gb` prices one key count; `branches` splits NSA's three terms
at a given context; `crossover` solves each comparison in closed form.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "17-native-sparse-attention"
LAYERS, KV_HEADS, HEAD_DIM, BYTES = 80, 8, 128, 2
BLOCK, TOP_K, WINDOW = 64, 16, 512
MLA_RANK = 512
CONTEXTS = (4096, 32768, 131072)
GIB = 1024 ** 3


def cache_gb(keys, width=KV_HEADS * HEAD_DIM):
    """The Lesson 14 formula: K and V, per layer, per token, at `width` per head-group."""
    return 2 * LAYERS * width * BYTES * keys / GIB


def branches(ref, context):
    """NSA's three per-query terms, priced, against full attention and MLA."""
    compressed, selected = context // BLOCK, TOP_K * BLOCK
    total = ref.count_nsa(context, BLOCK, TOP_K, WINDOW)
    return {"full": cache_gb(ref.count_full_attention(context)),
            "compressed": cache_gb(compressed), "selected": cache_gb(selected),
            "window": cache_gb(WINDOW), "nsa": cache_gb(total),
            "mla": cache_gb(context, width=MLA_RANK / 2),
            "keys": {"full": context, "compressed": compressed, "selected": selected,
                     "window": WINDOW, "nsa": total},
            "fixed_share": (selected + WINDOW) / total}


def crossovers():
    """Where each pair of curves meets, solved rather than searched."""
    return {"fine_vs_full": TOP_K * BLOCK,
            "nsa_vs_full": (TOP_K * BLOCK + WINDOW) / (1 - 1 / BLOCK),
            "nsa_vs_mla": (TOP_K * BLOCK + WINDOW) / (MLA_RANK / 2 / (KV_HEADS * HEAD_DIM)
                                                      - 1 / BLOCK),
            "compressed_overtakes": BLOCK * (TOP_K * BLOCK + WINDOW)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {context: branches(ref, context) for context in CONTEXTS}
    return {"rows": rows, "crossovers": crossovers(),
            "stored_gb": rows[131072]["full"] + rows[131072]["compressed"]}


def verify(result):
    rows, crossing = result["rows"], result["crossovers"]
    long, short = rows[131072], rows[4096]
    return [
        practice.Check(
            "ANSWER: 40.00 GB full against 1.09 GB for NSA, and the fine branch matches at N=1024",
            crossing["fine_vs_full"] == TOP_K * BLOCK == 1024 and long["nsa"] < 0.05 * long["full"],
            f"at {131072 // 1024}k context full attention is {long['full']:.2f} GB and NSA's "
            f"per-query total is {long['nsa']:.2f} GB -- compressed {long['compressed']:.2f}, "
            f"selected {long['selected']:.2f}, window {long['window']:.2f}, on key counts of "
            f"{long['keys']}. The selected branch is k * l keys whatever N is, so it equals full "
            f"attention at N = {crossing['fine_vs_full']}, and below that it reads more than the "
            "whole sequence",
        ),
        practice.Check(
            "FINDING: NSA is 9.2x smaller than MLA at 128k and 1.6x larger at 4k",
            long["nsa"] < long["mla"] and short["nsa"] > short["mla"],
            f"MLA's cache is 2 x layers x {MLA_RANK} x bytes x N -- linear in N with a small "
            f"constant -- while NSA's is N/l + k*l + W, mostly constant. At 128k they are "
            f"{long['nsa']:.2f} GB against {long['mla']:.2f}; at 4k, {short['nsa']:.2f} against "
            f"{short['mla']:.2f}. They cross at N = {crossing['nsa_vs_mla']:,.0f}, so two methods "
            "the exercise treats as alternatives are not comparable at a single context length",
        ),
        practice.Check(
            "MECHANISM: NSA caches the whole sequence and reads part of it",
            result["stored_gb"] > long["full"],
            f"the {long['nsa']:.2f} GB above is what NSA reads per query, not what it stores: the "
            f"selected branch picks blocks out of the full K and V, which must still be resident. "
            f"Stored, NSA is {long['full']:.2f} GB plus the compressed summaries -- "
            f"{result['stored_gb']:.2f} GB, more than full attention. The saving is bandwidth "
            "rather than capacity, and the exercise asks for a memory budget",
        ),
        practice.Check(
            "FINDING: only the compressed branch grows, and it overtakes the other two at 98,304",
            short["fixed_share"] > 0.9 > long["fixed_share"],
            f"the fixed terms -- k*l selected keys and W window keys -- are "
            f"{short['fixed_share']:.0%} of the per-query cost at 4k and "
            f"{long['fixed_share']:.0%} at 128k, and the compressed branch overtakes them at "
            f"N = {crossing['compressed_overtakes']:,}. NSA's asymptotic cost is N/l, which is "
            f"still linear in N -- a {BLOCK}x smaller slope than full attention, and not a flat "
            "one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
