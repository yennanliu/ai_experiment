"""Exercise 4 — 25.0% at 8k, and 50% is the ceiling no matter how long the context gets.

    Gemma 2 alternates full-attention and sliding-window-attention layers. Write
    the math for the KV cache when half the layers use a 4096-token sliding
    window instead of full context. How much memory does that save at 8k total
    context?

Reading of the exercise: the math is written as the lesson's own KV formula with
the per-layer sequence length made a per-layer quantity --
`2 x bytes x sum_layers(kv_heads x head_dim x cached_len)` with
`cached_len = min(window, context)` on the sliding layers. Gemma 2 27B's shape
is used (46 layers, 16 KV heads, 128 head dim, 4096 window) because `CONFIGS`
has no Gemma and the exercise names the model. The 8k answer is then swept
across context lengths, since the saving is a function of the ratio and the
exercise asks for one point on it.

**ANSWER: 2.88 GB becomes 2.16 GB -- a saving of exactly 25.0%.** At 8k
context, half the layers cache 8192 tokens and half cache 4096, so the total is
`(8192 + 4096) / (2 x 8192)` = 0.75 of the full cache. The saving is
`(1 - window/context) / 2` whenever `window < context`, and 0 when it is not.

**FINDING: the ceiling is 50% and the model approaches it, never reaching it.**

    ctx    8k    2.88 GB -> 2.16 GB    25.0%
    ctx   32k   11.50 GB -> 6.47 GB    43.8%
    ctx  128k   46.00 GB -> 23.72 GB   48.4%

Half the layers have no window, so their cache grows linearly with context
forever. The alternation caps the saving at 50% by construction, which is a
different promise from what "sliding window attention" sounds like.

**MECHANISM: the saving is a property of the ratio, not of the window.** At
`context <= window` the sliding layers cache everything and the saving is
**0.0%** -- the 4096-token window does nothing at 4k context and nothing at 2k.
Every number the exercise asks for is `(1 - 4096/context) / 2`, and the 8k point
it picks is the smallest context at which the answer is not zero.

**FINDING: 25% of Gemma's cache is 0.72 GB, and 25% of a model without GQA would
be 1.44 GB.** Gemma 2 27B is already GQA at 16 KV heads against 32 query heads,
so the window is the *second* halving applied to the same tensor. The two
compose multiplicatively and the second one is worth less, because it is halving
a number the first one already halved.

Structure: `cache_gb` is the per-layer-length formula; `sweep` runs it across
contexts with and without the alternation.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "14-open-models-architecture-walkthroughs"
LAYERS, KV_HEADS, HEAD_DIM, WINDOW = 46, 16, 128, 4096
QUERY_HEADS, DTYPE_BYTES = 32, 2
CONTEXTS = (2048, 4096, 8192, 32768, 131072)
GIB = 1024 ** 3


def cache_gb(context, full_layers, swa_layers, window=WINDOW, kv_heads=KV_HEADS):
    """The lesson's KV formula with the cached length made a per-layer quantity."""
    per_layer_token = 2 * kv_heads * HEAD_DIM * DTYPE_BYTES
    cached = full_layers * context + swa_layers * min(window, context)
    return cached * per_layer_token / GIB


def sweep():
    """Full cache against the alternating one, at every context."""
    rows = {}
    for context in CONTEXTS:
        full = cache_gb(context, LAYERS, 0)
        alternating = cache_gb(context, LAYERS // 2, LAYERS - LAYERS // 2)
        rows[context] = {"full": full, "alternating": alternating,
                         "saving": 1 - alternating / full}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = sweep()
    mha = {c: 1 - cache_gb(c, LAYERS // 2, LAYERS - LAYERS // 2, kv_heads=QUERY_HEADS)
           / cache_gb(c, LAYERS, 0, kv_heads=QUERY_HEADS) for c in CONTEXTS}
    gemma = {"hidden_size": 4608, "num_attention_heads": QUERY_HEADS,
             "num_key_value_heads": KV_HEADS}
    return {
        "rows": rows,
        "mha_savings": mha,
        "scheme": ref.attention_scheme(gemma),
        "saved_gb": rows[8192]["full"] - rows[8192]["alternating"],
        "mha_saved_gb": (cache_gb(8192, LAYERS, 0, kv_heads=QUERY_HEADS)
                         - cache_gb(8192, LAYERS // 2, LAYERS - LAYERS // 2,
                                    kv_heads=QUERY_HEADS)),
        "layers": (LAYERS // 2, LAYERS - LAYERS // 2),
    }


def verify(result):
    rows = result["rows"]
    at8k, at128k = rows[8192], rows[131072]
    return [
        practice.Check(
            "ANSWER: 2.88 GB becomes 2.16 GB at 8k -- a saving of exactly 25.0%",
            abs(at8k["saving"] - 0.25) < 1e-9,
            f"at 8k context {result['layers'][0]} layers cache 8192 tokens and "
            f"{result['layers'][1]} cache {WINDOW}, so the total is "
            f"{at8k['alternating']:.2f} GB against {at8k['full']:.2f} -- "
            f"(8192 + {WINDOW}) / (2 x 8192) = 0.75 of the full cache, saving "
            f"{result['saved_gb']:.2f} GB. The formula is (1 - window/context) / 2 whenever the "
            "window is shorter than the context",
        ),
        practice.Check(
            "FINDING: the ceiling is 50%, approached and never reached",
            at128k["saving"] < 0.5 and at8k["saving"] < at128k["saving"],
            ", ".join(f"{c // 1024}k {row['full']:.2f} -> {row['alternating']:.2f} GB "
                      f"({row['saving']:.1%})" for c, row in rows.items() if c >= 8192)
            + f". Half the layers have no window, so their cache grows linearly with context "
            f"forever and the alternation caps the saving at 50% by construction -- "
            f"{at128k['saving']:.1%} at 128k. That is a different promise from what 'sliding "
            "window attention' sounds like",
        ),
        practice.Check(
            "MECHANISM: the saving is a property of the ratio, not of the window",
            rows[2048]["saving"] == rows[4096]["saving"] == 0.0,
            f"at a context no longer than the window the sliding layers cache everything, so the "
            f"saving is {rows[4096]['saving']:.1%} at 4k and {rows[2048]['saving']:.1%} at 2k. "
            f"Every number the exercise asks for is (1 - {WINDOW}/context) / 2, and the 8k point "
            "it picks is the smallest context in this sweep at which the answer is not zero",
        ),
        practice.Check(
            "FINDING: the window is the second halving applied to the same tensor",
            result["scheme"].startswith("GQA")
            and abs(result["mha_saved_gb"] - 2 * result["saved_gb"]) < 1e-6,
            f"Gemma 2 27B is {result['scheme']} before the window is considered, so the "
            f"{at8k['saving']:.0%} at 8k is {result['saved_gb']:.2f} GB where the same alternation "
            f"on a multi-head model would save {result['mha_saved_gb']:.2f} GB -- the same "
            f"{list(result['mha_savings'].values())[2]:.0%} of a tensor twice the size. The two "
            "knobs compose multiplicatively and the second is worth less, because it halves a "
            "number the first already halved",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
