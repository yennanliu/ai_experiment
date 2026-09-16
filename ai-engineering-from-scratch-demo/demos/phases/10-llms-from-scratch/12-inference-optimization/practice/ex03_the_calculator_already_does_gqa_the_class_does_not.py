"""Exercise 3 — `kv_cache_memory` already implements GQA; the `KVCache` class still does not.

    Implement a grouped-query attention (GQA) version of the KV cache where
    `num_kv_heads < num_query_heads`. Llama 3 70B uses 64 query heads but only 8
    KV heads. Compute the memory savings vs full multi-head attention (8x
    reduction in KV cache size).

Reading of the exercise: the thing to implement is the *baseline*, not the
feature. `kv_cache_memory` already takes `num_kv_heads` and
`MODEL_CONFIGS["Llama-3-70B"]` already sets it to 8, so what has to be written to
make the comparison is the multi-head config the lesson does not ship -- the same
shape with `num_kv_heads = 64`. Both arms then go through the lesson's own
calculator and its own `memory_budget`.

**ANSWER: exactly 8.0x, which is the head-count ratio and nothing else.**
`2 x layers x kv_heads x head_dim x bytes` is linear in `kv_heads`, so 64 query
heads over 8 KV heads gives 8.0x at every context length and every dtype:
**320 KB** per token against **2560 KB**. The "8x reduction" the exercise asks
you to compute is `64 / 8` written out.

**FINDING: GQA is not an optimisation here, it is the deployment.** On
4 x A100-80GB at 4K context the 70B seats **126** users with 8 KV heads and
**15** with 64. Without GQA the model does not serve a useful batch on the
hardware the lesson prices it on, so the saving is the difference between a
product and a demo.

**FINDING: the lesson ships two KV-cache models and they disagree.** `KVCache`
takes a single `num_heads` and allocates `k_cache` and `v_cache` at that width,
so a cache built for Llama 3 70B's 64 query heads is **8x** the size the
calculator reports for the same model. The class is MHA, the calculator is GQA,
and the exercise asks you to add GQA to the one that has it.

**MECHANISM: the query heads never appear in the KV term.** Queries are not
cached -- they are consumed in the step that produces them. Only K and V persist,
so the cache size depends on `num_kv_heads` alone, and sharing 8 KV heads across
64 query heads costs attention quality rather than memory.

Structure: `configs` returns the shipped GQA shape and the MHA shape it is
compared against; `profile` runs one shape through the lesson's own calculator
and budget.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "12-inference-optimization"
MODEL, QUERY_HEADS, GPU_GB = "Llama-3-70B", 64, 320
CONTEXTS = (4096, 32768, 131072)


def configs(ref):
    """The shipped GQA shape and the multi-head baseline the lesson does not ship."""
    gqa = ref.MODEL_CONFIGS[MODEL]
    return {"gqa": gqa, "mha": dict(gqa, num_kv_heads=QUERY_HEADS)}


def profile(ref, config):
    """One shape through the lesson's own calculator and its own budget."""
    return {"per_token_kb": ref.kv_cache_memory(config, 1)["per_token_kb"],
            "gb": {n: ref.kv_cache_memory(config, n)["total_gb"] for n in CONTEXTS},
            "users": ref.memory_budget(config, GPU_GB)["max_users_at_4k"],
            "kv_heads": config["num_kv_heads"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shapes = configs(ref)
    arms = {name: profile(ref, config) for name, config in shapes.items()}
    cache = ref.KVCache(2, QUERY_HEADS, shapes["gqa"]["head_dim"], 16)
    return {
        "arms": arms,
        "config_keys": sorted(shapes["gqa"]),
        "ratio": arms["mha"]["per_token_kb"] / arms["gqa"]["per_token_kb"],
        "context_ratios": [arms["mha"]["gb"][n] / arms["gqa"]["gb"][n] for n in CONTEXTS],
        "class_heads": cache.k_cache.shape[1],
        "class_bytes": cache.memory_bytes(),
        "query_heads": QUERY_HEADS,
    }


def verify(result):
    gqa, mha = result["arms"]["gqa"], result["arms"]["mha"]
    ratios = result["context_ratios"]
    return [
        practice.Check(
            "ANSWER: exactly 8.0x, at every context length -- it is the head-count ratio",
            result["ratio"] == QUERY_HEADS / gqa["kv_heads"] and set(ratios) == {result["ratio"]},
            f"{gqa['per_token_kb']:.0f} KB per token with {gqa['kv_heads']} KV heads against "
            f"{mha['per_token_kb']:.0f} KB with {mha['kv_heads']}, a ratio of "
            f"{result['ratio']:.1f}. 2 x layers x kv_heads x head_dim x bytes is linear in "
            "kv_heads, so the ratio is identical at "
            + ", ".join(f"{n // 1024}K ({r:.1f}x)" for n, r in zip(CONTEXTS, ratios))
            + f" -- the 8x the exercise asks you to compute is {result['query_heads']} / "
            f"{gqa['kv_heads']} written out",
        ),
        practice.Check(
            "FINDING: GQA is not an optimisation here, it is the deployment",
            gqa["users"] > 8 * mha["users"],
            f"on 4xA100-80GB at 4K context the profiler seats {gqa['users']} users with "
            f"{gqa['kv_heads']} KV heads and {mha['users']} with {mha['kv_heads']}. Without GQA "
            f"the 70B does not serve a useful batch on the hardware the lesson prices it on: "
            f"at 4K a single MHA user needs {mha['gb'][4096]:.1f} GB of cache against "
            f"{gqa['gb'][4096]:.2f}",
        ),
        practice.Check(
            "FINDING: the lesson ships two KV-cache models and they disagree",
            result["class_heads"] == QUERY_HEADS,
            f"KVCache takes one num_heads and allocates k_cache and v_cache at that width, so "
            f"building one for this model's {result['query_heads']} query heads gives "
            f"k_cache.shape[1] = {result['class_heads']} and {result['class_bytes'] / 1024:.0f} "
            "KB for 16 positions on 2 layers. The class is multi-head and the calculator is "
            "grouped-query, and the exercise asks you to add GQA to the one that already has it",
        ),
        practice.Check(
            "MECHANISM: the query heads are not an input to the calculator at all",
            "num_query_heads" not in result["config_keys"]
            and gqa["per_token_kb"] / gqa["kv_heads"] == mha["per_token_kb"] / mha["kv_heads"],
            f"a model config is {result['config_keys']} -- there is no query-head field, because "
            "queries are not cached: they are consumed in the step that produces them, and only "
            f"K and V persist. Cost per KV head is {gqa['per_token_kb'] / gqa['kv_heads']:.0f} KB "
            f"per token in both arms, so the whole 8x is head count. Sharing {gqa['kv_heads']} KV "
            f"heads across {result['query_heads']} query heads costs attention quality, which no "
            "memory calculator can price, and costs nothing in the quantity this one measures",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
