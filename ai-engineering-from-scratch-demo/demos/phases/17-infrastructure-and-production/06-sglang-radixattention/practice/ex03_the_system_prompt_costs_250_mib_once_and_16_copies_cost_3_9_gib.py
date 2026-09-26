"""Exercise 3 — the system prompt costs 250 MiB once, and 16 copies cost 3.9 GiB.

    Compute the HBM cost of keeping a 2,000-token system prompt resident as
    one radix branch on Llama 3.1 8B. Compare to the cost of a 16-sequence
    batch without prefix reuse.

Reading of the exercise: the cost is the K and V tensors for every layer. The
Llama 3.1 8B shape comes from its published config: 32 layers, 8 KV heads
(grouped-query attention over 32 query heads), head_dim 128, bf16. The weight
count, 8.03B, is used only for scale. The lesson's own BLOCK_TOKENS and budget
convert the result into the toy's units.

**ANSWER: 250 MiB resident once, against 3.91 GiB for 16 private copies.**
Per token, 2 (K, V) x 32 layers x 8 heads x 128 dims x 2 bytes is 128 KiB, so
2000 tokens are 262.1 MB (250 MiB). A 16-sequence batch without reuse holds
the prefix 16 times: 4.19 GB (3.91 GiB). The radix branch saves 15 copies,
3.66 GiB. On an 80 GB H100 the batch's copies take 5.2%, the shared copy 0.33%.
The 16.06 GB of bf16 weights do not change with either choice. Each
sequence's own suffix costs the same with or without sharing, so it drops out
of the comparison.

**FINDING: grouped-query attention already cuts it 4x.** The same arithmetic
with 32 KV heads, as in multi-head attention, gives 1000 MiB for the prefix and
15.6 GiB for the batch. A per-token figure computed from `num_attention_heads`
instead of `num_key_value_heads` overstates the cost 4x.

**FINDING: the lesson's block counts are off by one, and its budget is tiny.**
At 16 tokens a block, a 2,000-token prompt is 125 blocks, which is what
`code/main.py` computes. The concept diagram says 124, and it gives doc A's 500
tokens 31 blocks where the code gives 32. One block is 2 MiB here, so
KV_BUDGET_BLOCKS = 160 is 320 MiB: 1.28x the system prompt, and less than one
2,860-token RAG request.

Structure: `kv_bytes()` is the one formula; everything else is unit
conversion.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "06-sglang-radixattention"
LLAMA_31_8B = {"layers": 32, "kv_heads": 8, "q_heads": 32, "head_dim": 128, "bytes": 2}
PARAMS, HBM, BATCH, MIB, GIB = 8.03e9, 80e9, 16, 2**20, 2**30


def kv_bytes(tokens, kv_heads=LLAMA_31_8B["kv_heads"]):
    """K and V, every layer, every KV head, bf16."""
    m = LLAMA_31_8B
    return 2 * m["layers"] * kv_heads * m["head_dim"] * m["bytes"] * tokens


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prefix = ref.token_count("SYSTEM")
    one, batch = kv_bytes(prefix), BATCH * kv_bytes(prefix)
    request = sum(ref.token_count(s) for s in ref.workload_rag()[0].segments)
    block = kv_bytes(ref.BLOCK_TOKENS)
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "per_token": kv_bytes(1), "one": one, "batch": batch, "saved": batch - one,
        "weights": PARAMS * LLAMA_31_8B["bytes"],
        "mha": (kv_bytes(prefix, LLAMA_31_8B["q_heads"]),
                BATCH * kv_bytes(prefix, LLAMA_31_8B["q_heads"])),
        "blocks": (-(-prefix // ref.BLOCK_TOKENS), -(-ref.token_count("DOC_0") // ref.BLOCK_TOKENS)),
        "doc_blocks": ("(2,000 tokens, 124 KV blocks)" in doc, "(500 tokens, 31 blocks)" in doc),
        "budget": ref.KV_BUDGET_BLOCKS * block, "block": block, "request": request,
        "request_blocks": ref.KV_BUDGET_BLOCKS * ref.BLOCK_TOKENS,
    }


def verify(result):
    one, batch, mha = result["one"], result["batch"], result["mha"]
    return [
        practice.Check(
            "ANSWER: 250 MiB resident once, against 3.91 GiB for 16 private copies",
            all([result["per_token"] == 128 * 1024, one == 250 * MIB,
                 round(batch / GIB, 2) == 3.91, round(result["saved"] / GIB, 2) == 3.66,
                 round(batch / HBM * 100, 1) == 5.2]),
            f"{result['per_token'] // 1024} KiB/token; prefix {one / 1e6:.1f} MB "
            f"({one / MIB:.0f} MiB); batch {batch / 1e9:.2f} GB ({batch / GIB:.2f} GiB); "
            f"saved {result['saved'] / GIB:.2f} GiB; {batch / HBM:.1%} vs {one / HBM:.2%} "
            f"of 80 GB; weights {result['weights'] / 1e9:.2f} GB either way",
        ),
        practice.Check(
            "FINDING: grouped-query attention already cuts it 4x",
            mha[0] == 4 * one and mha[0] == 1000 * MIB and round(mha[1] / GIB, 1) == 15.6,
            f"with 32 KV heads the prefix is {mha[0] / MIB:.0f} MiB and the batch "
            f"{mha[1] / GIB:.1f} GiB",
        ),
        practice.Check(
            "FINDING: the lesson's block counts are off by one, and its budget is tiny",
            all([result["blocks"] == (125, 32), result["doc_blocks"] == (True, True),
                 result["block"] == 2 * MIB, result["budget"] == 320 * MIB,
                 result["request_blocks"] < result["request"]]),
            f"code blocks (SYSTEM, DOC) {result['blocks']} vs the diagram's (124, 31); "
            f"budget {result['budget'] / MIB:.0f} MiB = {result['request_blocks']} tokens, "
            f"under one {result['request']}-token request",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
