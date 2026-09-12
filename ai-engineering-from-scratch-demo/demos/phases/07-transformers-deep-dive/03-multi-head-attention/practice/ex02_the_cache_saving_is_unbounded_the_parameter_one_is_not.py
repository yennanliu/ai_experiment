"""Exercise 2 — the cache saving is unbounded, the parameter saving is capped at half.

    **Medium.** Implement MQA (one KV head shared across all query heads).
    Measure how much parameter count drops vs full MHA. Compute how much the
    KV-cache size shrinks at inference for N=2048.

Reading of the exercise: MQA is `grouped_query_attention(..., n_kv_heads=1)`,
which the lesson already ships -- `repeat = n_heads // 1` makes every query head
index `Kh_small[0]`, so all 16 heads read one K and one V. So "implement" is read
as *exercise the lesson's own path and check it is MQA*, then do the arithmetic
the exercise asks for at the configuration Exercise 1 sweeps to, d_model=64 with
n_heads=16.

**ANSWER: parameters fall 46.875%, the KV cache falls 93.75%.**

| | MHA | MQA | ratio |
|---|---:|---:|---:|
| parameters | 16,384 | **8,704** | 1.88x |
| KV cache at N=2048 | 262,144 | **16,384** | **16x** |

**FINDING: the two savings are not the same kind of number.** MQA keeps `Wq` and
`Wo` at `d_model^2` each and shrinks only `Wk` and `Wv`, so the parameter ratio
is `1/2 + 1/(2h)` -- 0.53125 at h=16, and it converges to **exactly 1/2** as
heads grow. No amount of KV sharing can remove more than half the parameters of
an attention block. The cache ratio is `1/h` with no floor at all: it is 16x at
16 heads and 64x at 64, and it does not depend on N or on d_model. MQA is a
memory-bandwidth change that happens to save some weights, and the exercise's own
pairing of the two questions is what makes that visible.

**FINDING: the cache is where the asymmetry bites.** At N=2048 one layer's MHA
cache is 16x the entire block's parameter count; at N=2048 with MQA it is
*equal* to the parameters saved. Cache grows with N and parameters do not, which
is why every 2023-onward decoder shares KV and none of them shares Wq.

**CONTROL: GQA at `n_kv_heads = n_heads` is MHA, to 0.0.** The same function
reproduces `multi_head_attention` bit for bit when nothing is shared, so the
group count is the only thing MQA changes.

Structure: `pull` reads a Matrix out; `run` drives the lesson's own GQA at a
chosen group count; `counts` is the arithmetic, in closed form.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "03-multi-head-attention"
D_MODEL, N_HEADS, CONTEXT, TOKENS = 64, 16, 2048, 6


def pull(matrix):
    """A lesson Matrix as a list of rows, so results can be compared elementwise."""
    return [matrix.row(i) for i in range(matrix.rows)]


def run(ref, rng, n_heads, n_kv_heads, d_model=D_MODEL, tokens=TOKENS):
    """The lesson's own grouped_query_attention at one group count."""
    head = d_model // n_heads
    x = ref.randn_matrix(tokens, d_model, rng, scale=1.0)
    wq = ref.randn_matrix(d_model, d_model, rng)
    wk = ref.randn_matrix(d_model, head * n_kv_heads, rng)
    wv = ref.randn_matrix(d_model, head * n_kv_heads, rng)
    wo = ref.randn_matrix(d_model, d_model, rng)
    shared = ref.split_heads(ref.matmul(x, wk), n_kv_heads)
    gqa = ref.grouped_query_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads)
    return gqa, shared, (x, wq, wk, wv, wo)


def counts(n_heads, d_model=D_MODEL, context=CONTEXT):
    """Parameters and KV-cache elements for MHA and for MQA, per layer."""
    head = d_model // n_heads
    return {
        "params": (4 * d_model * d_model, 2 * d_model * d_model + 2 * d_model * head),
        "cache": (2 * context * d_model, 2 * context * head),
        "limit": 0.5 + 1 / (2 * n_heads),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(42)
    mqa, shared, _ = run(ref, rng, N_HEADS, 1)
    rng = random.Random(7)
    full, _, args = run(ref, rng, 4, 4)
    plain = ref.multi_head_attention(*args, n_heads=4)[0]
    numbers = counts(N_HEADS)
    params, cache = numbers["params"], numbers["cache"]
    return {
        "shape": (mqa.rows, mqa.cols), "kv_heads": len(shared), "kv_width": shared[0].cols,
        "identical": max(abs(a - b) for ra, rb in zip(pull(full), pull(plain))
                         for a, b in zip(ra, rb)),
        "params": params, "param_drop": 1 - params[1] / params[0],
        "cache": cache, "cache_drop": 1 - cache[1] / cache[0], "cache_ratio": cache[0] / cache[1],
        "limit": numbers["limit"], "asymptote": counts(4096, d_model=4096)["params"],
    }


def verify(result):
    params, cache = result["params"], result["cache"]
    floor = 1 - result["asymptote"][1] / result["asymptote"][0]
    return [
        practice.Check(
            "ANSWER: one KV head, and the lesson already has the code for it",
            result["kv_heads"] == 1 and result["kv_width"] == D_MODEL // N_HEADS,
            f"grouped_query_attention(..., n_kv_heads=1) makes repeat = {N_HEADS}, so every query "
            f"head indexes Kh_small[0]: {result['kv_heads']} KV head of width "
            f"{result['kv_width']} = d_model / n_heads, read by all {N_HEADS} query heads. MQA is "
            f"an argument to a function the lesson ships, not a new implementation",
        ),
        practice.Check(
            "ANSWER: parameters fall 46.875% at d_model=64, n_heads=16",
            abs(result["param_drop"] - 0.46875) < 1e-12,
            f"{params[0]:,} -> {params[1]:,} per layer, {result['param_drop']:.3%}. Wq and Wo stay "
            f"at d_model^2 and only Wk and Wv shrink, so the ratio is 1/2 + 1/(2h) = "
            f"{result['limit']:.5f}",
        ),
        practice.Check(
            "ANSWER: the KV cache at N=2048 falls 93.75%, exactly n_heads",
            result["cache_ratio"] == N_HEADS and abs(result["cache_drop"] - 0.9375) < 1e-12,
            f"{cache[0]:,} -> {cache[1]:,} elements per layer, "
            f"{result['cache_ratio']:.0f}x. The ratio is 1/h with no dependence on N or d_model, "
            f"so it is {N_HEADS}x here and 64x at 64 heads",
        ),
        practice.Check(
            "FINDING: the parameter saving is capped at half, the cache saving is not",
            floor < 0.5 and floor > 0.499,
            f"at d_model = n_heads = 4096, one scalar per KV head, the parameter drop is "
            f"{floor:.4%} and it never reaches 50%, because "
            f"Wq and Wo are untouched by KV sharing. The cache ratio has no such floor. The two "
            "questions the exercise pairs are a bandwidth number and a weight number, and only "
            "one of them is unbounded -- which is why decoders share KV and never share Wq",
        ),
        practice.Check(
            "CONTROL: GQA with n_kv_heads = n_heads is MHA, to 0.0",
            result["identical"] == 0.0 and result["shape"] == (TOKENS, D_MODEL),
            f"grouped_query_attention at repeat=1 reproduces multi_head_attention bit for bit "
            f"({result['identical']}), so the group count is the only thing MQA changes. The MQA "
            f"forward itself returns {result['shape']}, the same shape as MHA",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
