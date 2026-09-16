"""Exercise 1 — the 1.246B gap is the untied LM head, to 0.0003%.

    Read the Qwen 2.5 72B config from HuggingFace. Compute total parameters from
    scratch. Compare to the HF-reported value and identify where any delta comes
    from (head dim rounding, KV sharing factor, etc.).

Reading of the exercise: "from scratch" is the lesson's own `analyze`, since it
is a from-scratch parameter counter and the config is already in `CONFIGS`. The
comparison is against Qwen's reported **72.706B**, and the delta is then
attributed term by term rather than described -- each candidate the exercise
names is computed and either accounts for the gap or does not.

**ANSWER: `analyze` gives 71.4597B against 72.706B, and the 1.2463B gap is one
term.** Qwen 2.5 72B does not tie its input embedding to its output head, and
`analyze` counts `vocab_size * hidden_size` exactly once. Adding the second copy
-- **1.2457B** -- and Qwen's q/k/v attention biases -- **0.0008B** -- gives
**72.7062B** against a reported 72.706B: a residual of **0.0002B**, or 0.0003%.

**FINDING: neither candidate the exercise suggests contributes anything.** Head
dim is `8192 / 64 = 128` exactly, so there is no rounding to find; the KV
sharing factor is already in `attention_params_per_layer`, which sizes the K and
V projections at `kv_heads * head_dim` rather than `hidden_size`. Turning GQA off
would *add* **9.4B**, not 1.2B -- the sharing factor is the largest single term
in the count and it is the one already correct.

**MECHANISM: an untied head is a whole embedding table, and vocabulary is where
the width is.** At `vocab=152064` and `h=8192` one table is 1.246B parameters --
**1.7%** of the model, and more than any single layer's 0.8B. Tying is a
one-line config flag that moves more parameters than most architecture choices
in the lesson's six-knob table.

**FINDING: the same omission is worth 0.93B on DeepSeek V3**, whose config also
has an untied head, and `analyze` reports 665.9B against a published 671B. The
LM head closes a fifth of that gap; the rest is the multi-token-prediction module
the config does not describe.

Structure: `terms` computes each candidate correction from the config; `closes`
reports how much of the gap each one accounts for.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "14-open-models-architecture-walkthroughs"
MODEL, REPORTED = "qwen2.5-72b", 72.706e9
DEEPSEEK, DEEPSEEK_REPORTED = "deepseek-v3", 671e9


def terms(config):
    """Each correction the exercise's candidates imply, in parameters."""
    h, vocab, layers = config["hidden_size"], config["vocab_size"], config["num_hidden_layers"]
    q_heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = h // q_heads
    return {
        "untied_head": vocab * h,
        "qkv_bias": layers * (h + 2 * kv_heads * head_dim),
        "head_dim_remainder": layers * (h - q_heads * head_dim),
        "gqa_if_disabled": layers * 2 * h * (q_heads - kv_heads) * head_dim,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    config = ref.CONFIGS[MODEL]
    counted = ref.analyze(MODEL, config)
    parts = terms(config)
    closed = counted.total_params + parts["untied_head"] + parts["qkv_bias"]
    deep_config = ref.CONFIGS[DEEPSEEK]
    deep = ref.analyze(DEEPSEEK, deep_config)
    return {
        "counted": counted.total_params,
        "reported": REPORTED,
        "gap": REPORTED - counted.total_params,
        "terms": parts,
        "closed": closed,
        "residual": REPORTED - closed,
        "embedding_once": counted.embedding_params == parts["untied_head"],
        "head_dim": config["hidden_size"] // config["num_attention_heads"],
        "per_layer": counted.attn_params_per_layer + counted.mlp_params_per_layer,
        "deepseek": (deep.total_params, DEEPSEEK_REPORTED,
                     deep_config["vocab_size"] * deep_config["hidden_size"]),
    }


def verify(result):
    parts, gap = result["terms"], result["gap"]
    deep_counted, deep_reported, deep_head = result["deepseek"]
    return [
        practice.Check(
            "ANSWER: the 1.246B gap is the untied LM head, closing to 0.0003%",
            abs(result["residual"]) < 0.001 * result["reported"]
            and abs(parts["untied_head"] - gap) < 0.01 * gap,
            f"analyze counts {result['counted'] / 1e9:.4f}B against Qwen's reported "
            f"{result['reported'] / 1e9:.3f}B, a gap of {gap / 1e9:+.4f}B. Qwen 2.5 72B does not "
            f"tie its input embedding to its output head and analyze counts vocab_size * "
            f"hidden_size exactly once, so adding the second copy "
            f"({parts['untied_head'] / 1e9:.4f}B) and the q/k/v attention biases "
            f"({parts['qkv_bias'] / 1e9:.4f}B) gives {result['closed'] / 1e9:.4f}B -- a residual "
            f"of {result['residual'] / 1e9:+.4f}B, or "
            f"{abs(result['residual']) / result['reported']:.4%}",
        ),
        practice.Check(
            "FINDING: neither candidate the exercise names contributes anything",
            parts["head_dim_remainder"] == 0 and parts["gqa_if_disabled"] > 5 * gap,
            f"head dim is 8192 / 64 = {result['head_dim']} exactly, so the head-dim rounding the "
            f"exercise suggests is {parts['head_dim_remainder']} parameters. The KV sharing "
            f"factor is already in attention_params_per_layer, which sizes K and V at "
            f"kv_heads * head_dim rather than hidden_size: turning GQA off would add "
            f"{parts['gqa_if_disabled'] / 1e9:.1f}B, not {gap / 1e9:.1f}B. The sharing factor is "
            "the largest term in the count and it is the one already correct",
        ),
        practice.Check(
            "MECHANISM: an untied head is a whole embedding table, and vocabulary is the width",
            result["embedding_once"] and parts["untied_head"] > result["per_layer"],
            f"at vocab 152,064 and h 8,192 one table is {parts['untied_head'] / 1e9:.3f}B "
            f"parameters -- {parts['untied_head'] / result['reported']:.1%} of the model, and "
            f"more than any single layer's {result['per_layer'] / 1e9:.2f}B of attention plus "
            "MLP. Tying is a one-line config flag that moves more parameters than most of the "
            "architecture choices in the lesson's six-knob table",
        ),
        practice.Check(
            "FINDING: the same omission is worth 0.93B on DeepSeek V3",
            deep_counted < deep_reported and deep_head > 0.9e9,
            f"analyze reports {deep_counted / 1e9:.1f}B for DeepSeek V3 against a published "
            f"{deep_reported / 1e9:.0f}B, and its config is also untied: the LM head is "
            f"{deep_head / 1e9:.2f}B, which closes "
            f"{deep_head / (deep_reported - deep_counted):.0%} of that gap. The rest is the "
            "multi-token-prediction module, which the config does not describe and the counter "
            "therefore cannot see",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
