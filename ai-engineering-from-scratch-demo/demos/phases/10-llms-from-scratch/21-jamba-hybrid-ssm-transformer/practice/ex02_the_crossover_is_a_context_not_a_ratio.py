"""Exercise 2 — no integer layer ratio makes them equal; the crossover is at 248 tokens of context.

    Modify the calculator to model a 1:3 hybrid (4 Mamba : 1 Attention) and a
    1:15 hybrid (14 Mamba : 1 Attention). Plot KV cache vs ratio. At what ratio
    does the KV cache equal the SSM state memory?

Reading of the exercise: the two hybrids are built as `HybridConfig`s and swept
with the lesson's own two functions, and the closing question is then solved
rather than searched, because both quantities are linear -- one in `attn_layers`
and the context, the other in `total_layers - attn_layers` and nothing else.

**ANSWER: at no integer ratio. The KV cache is 1,057x the SSM state even at 1
attention layer in 32.**

    attention layers    ratio      KV at 256k    SSM state    KV / SSM
          16            1:1        64.000 GB     2.00 MB       32,768
           8            1:3        32.000 GB     3.00 MB       10,923
           4            1:7        16.000 GB     3.50 MB        4,681
           2            1:15        8.000 GB     3.75 MB        2,185
           1            1:31        4.000 GB     3.88 MB        1,057

Solving `2*A*kv_heads*head_dim*ctx = (32-A)*hidden*state` at 256k gives
**A = 0.00098** attention layers. The question has no answer in the space it is
asked in.

**MECHANISM: the two memories scale in different variables.** KV is linear in
*context* and in the attention-layer count; the SSM state is linear in the Mamba
layer count and **constant in context**. Sweeping the ratio moves both by a
factor of at most 32 and leaves a gap of three orders of magnitude; the only
variable that can close it is the one the exercise holds fixed.

**FINDING: the crossover is a context length, and it is 248 tokens.** With one
attention layer and 31 Mamba layers, the KV cache equals the SSM state at
`ctx = 31*4096*16 / (2*1*32*128)` = **248**. Beyond a two-sentence prompt, the
attention layers dominate the cache at every ratio.

**FINDING: the SSM state *grows* as attention layers are removed, and it does not
matter.** Going from 16 attention layers to 1 doubles the Mamba count and takes
the state from 2.00 MB to 3.88 MB while the KV cache falls 64.00 GB to 4.00. The
term the exercise is asking about moves in the opposite direction from the one
that decides the answer, by a factor 8,000 times smaller.

Structure: `sweep` runs the lesson's own two functions across the attention-layer
counts; `crossover_context` solves for the context at which the two are equal.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "21-jamba-hybrid-ssm-transformer"
CONTEXT, BYTES, GIB, MIB = 262_144, 2, 1024 ** 3, 1024 ** 2
LAYERS, HIDDEN, HEAD_DIM, KV_HEADS, STATE = 32, 4096, 128, 32, 16
ATTENTION = (16, 8, 4, 2, 1)


def shape(ref, attn_layers):
    return ref.HybridConfig(name=f"1:{LAYERS // attn_layers - 1}", total_layers=LAYERS,
                            attn_layers=attn_layers, hidden=HIDDEN, n_q_heads=32,
                            n_kv_heads=KV_HEADS, head_dim=HEAD_DIM, ssm_state_size=STATE)


def sweep(ref):
    rows = {}
    for attn in ATTENTION:
        cfg = shape(ref, attn)
        kv = ref.kv_cache_bytes(cfg, CONTEXT, BYTES)
        state = ref.ssm_state_bytes(cfg, BYTES)
        rows[attn] = {"name": cfg.name, "kv_gb": kv / GIB, "state_mb": state / MIB,
                      "ratio": kv / state}
    return rows


def crossover_context(attn_layers):
    """The context at which one config's KV cache equals its SSM state."""
    return (LAYERS - attn_layers) * HIDDEN * STATE / (2 * attn_layers * KV_HEADS * HEAD_DIM)


def crossover_layers(context):
    """The (fractional) attention-layer count at which they are equal, at a fixed context."""
    return LAYERS * HIDDEN * STATE / (2 * KV_HEADS * HEAD_DIM * context + HIDDEN * STATE)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = sweep(ref)
    return {
        "rows": rows,
        "layers_needed": crossover_layers(CONTEXT),
        "context_needed": crossover_context(1),
        "at_short": {ctx: crossover_layers(ctx) for ctx in (256, 1024, 8192)},
        "state_growth": rows[1]["state_mb"] / rows[16]["state_mb"],
        "kv_drop": rows[16]["kv_gb"] / rows[1]["kv_gb"],
    }


def column(rows, field, fmt):
    return ", ".join(f"{row['name']} {format(row[field], fmt)}" for row in rows.values())


def verify(result):
    rows = result["rows"]
    sparse, dense = rows[1], rows[16]
    return [
        practice.Check(
            "ANSWER: at no integer ratio -- the KV cache is 1,057x the state at 1 layer in 32",
            result["layers_needed"] < 1 and sparse["ratio"] > 1000,
            "at 256k the KV cache is " + column(rows, "kv_gb", ".3f")
            + " GB against SSM states of " + column(rows, "state_mb", ".2f")
            + " MB, ratios of " + column(rows, "ratio", ",.0f")
            + f". Solving 2*A*kv_heads*head_dim*ctx = (32-A)*hidden*state gives "
            f"A = {result['layers_needed']:.5f} attention layers, so the question has no answer "
            "in the space it is asked in",
        ),
        practice.Check(
            "MECHANISM: the two memories scale in different variables",
            abs(result["kv_drop"] - 16) < 1e-9 and result["state_growth"] < 2,
            f"KV is linear in context and in the attention-layer count; the SSM state is linear in "
            f"the Mamba-layer count and constant in context. Sweeping the ratio moves the KV cache "
            f"by {result['kv_drop']:.0f}x and the state by {result['state_growth']:.2f}x, leaving "
            "a gap of three orders of magnitude. The only variable that can close it is the one "
            "the exercise holds fixed",
        ),
        practice.Check(
            "FINDING: the crossover is a context length, and it is 248 tokens",
            240 < result["context_needed"] < 260,
            f"with one attention layer and {LAYERS - 1} Mamba layers, the KV cache equals the SSM "
            f"state at ctx = {LAYERS - 1}*{HIDDEN}*{STATE} / (2*1*{KV_HEADS}*{HEAD_DIM}) = "
            f"{result['context_needed']:.0f} tokens. Beyond a two-sentence prompt the attention "
            "layers dominate at every ratio, and at "
            + ", ".join(f"{ctx} tokens the answer is {value:.2f} layers"
                        for ctx, value in result["at_short"].items())
            + " -- still below one",
        ),
        practice.Check(
            "FINDING: the SSM state grows as attention layers are removed, and it does not matter",
            sparse["state_mb"] > dense["state_mb"] and sparse["kv_gb"] < dense["kv_gb"],
            f"going from {16} attention layers to 1 doubles the Mamba count and takes the state "
            f"from {dense['state_mb']:.2f} MB to {sparse['state_mb']:.2f} while the KV cache falls "
            f"{dense['kv_gb']:.2f} GB to {sparse['kv_gb']:.2f}. The term the exercise is asking "
            f"about moves in the opposite direction from the one that decides the answer, by a "
            f"factor {result['kv_drop'] / result['state_growth'] * 1000:.0f} times smaller",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
