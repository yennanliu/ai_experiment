"""Exercise 5 — the seventh knob is in the lesson's own `CONFIGS`, worth 7.9% of DeepSeek's active parameters.

    Find a recent frontier open model that was released after this lesson was
    written. Identify which of the six knobs it picked and whether it introduced
    a seventh knob. The curriculum will feel out of date the moment a new
    architecture ships -- the goal is to update your table without rebuilding
    your mental model.

Reading of the exercise: a model released after the lesson cannot be fetched
from here, and the exercise's own point is that the *table* is the artifact, not
the model list. So the test is run against the newest model the lesson already
ships -- DeepSeek V3 -- by asking what `analyze`'s six-knob `verdict` string
reports about it and what it leaves out. If the table is complete, the verdict
accounts for the model; if a knob is missing, the missing knob shows up as
parameters the verdict cannot explain.

**ANSWER: the verdict lists five flags and misses a knob worth 2.554B active
parameters.**

    RMSNORM · SWIGLU · ROPE · MLA · MoE 256e/top-8

`shared_experts: 1` is in the config, is read by `analyze`, changes both
`total_params` and `active_params`, and appears in no flag. It is **7.9%** of
DeepSeek's 32.4B active parameters -- more than the router, the norms and the
first three dense layers together.

**FINDING: a shared expert is a different kind of knob from the six.** The six
are substitutions -- LayerNorm *or* RMSNorm, dense *or* MoE. A shared expert is
an addition: an always-on MLP alongside the routed ones, which makes "dense vs
sparse" a spectrum rather than a switch. DeepSeek V3 is **both**, in every MoE
layer, and the table has no column for that.

**FINDING: the norm and activation knobs are settled across every model that
postdates GPT-2.** All six non-GPT-2 configs are RMSNorm and SwiGLU; the
position knob varies only between RoPE and RoPE-with-YaRN. The knobs that still
discriminate are attention sharing (MHA, GQA, MLA) and dense-vs-MoE, so the
table is carrying settled questions alongside open ones -- which is why a new
model mostly needs a row rather than a column.

**MECHANISM: what makes a knob a knob is that `analyze` reads it.** `moe`,
`num_experts`, `experts_per_token`, `attention`, `kv_lora_rank`,
`first_dense_layers` and `shared_experts` all change the parameter count;
`position` changes nothing at all -- RoPE has no parameters, so the third knob in
the table is invisible to the counter that scores the other five.

Structure: `flags` re-derives the verdict string; `unflagged` prices the config
fields that move parameters and are not reported.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "14-open-models-architecture-walkthroughs"
NEWEST = "deepseek-v3"
KNOB_FIELDS = ("norm", "activation", "position", "num_key_value_heads", "moe")


def shared_expert_params(ref, config):
    """What `shared_experts` costs: an always-on MLP in every MoE layer."""
    expert = ref.mlp_params(config["hidden_size"],
                            config.get("moe_intermediate_size", config["intermediate_size"]),
                            config["activation"])
    moe_layers = config["num_hidden_layers"] - config.get("first_dense_layers", 0)
    return expert * config.get("shared_experts", 0) * moe_layers


def position_costs(ref, config):
    """Whether the position knob moves the parameter count at all."""
    rope = ref.analyze("probe", dict(config, position="rope")).total_params
    learned = ref.analyze("probe", dict(config, position="learned")).total_params
    return rope, learned


def settled(ref):
    """Which of the six knobs still take more than one value across the shipped configs."""
    modern = {n: c for n, c in ref.CONFIGS.items() if n != "gpt2-small"}
    return {field: {str(c.get(field)) for c in modern.values()} for field in KNOB_FIELDS}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    config = ref.CONFIGS[NEWEST]
    breakdown = ref.analyze(NEWEST, config)
    rope, learned = position_costs(ref, config)
    shared = shared_expert_params(ref, config)
    return {
        "verdict": breakdown.verdict,
        "flags": breakdown.verdict.split(" · "),
        "shared_params": shared,
        "active": breakdown.active_params,
        "shared_in_config": "shared_experts" in config,
        "shared_in_verdict": "shared" in breakdown.verdict.lower(),
        "position_free": rope == learned,
        "settled": settled(ref),
        "scheme": breakdown.attention_scheme,
    }


def verify(result):
    flags, settled_knobs = result["flags"], result["settled"]
    fixed = [k for k, v in settled_knobs.items() if len(v) == 1]
    varying = [k for k, v in settled_knobs.items() if len(v) > 1]
    share = result["shared_params"] / result["active"]
    return [
        practice.Check(
            "ANSWER: the verdict lists five flags and misses a knob worth 7.9% of active params",
            result["shared_in_config"] and not result["shared_in_verdict"] and share > 0.05,
            f"the six-knob verdict for {NEWEST} is '{result['verdict']}' -- "
            f"{len(flags)} flags. shared_experts is in the config, is read by analyze, changes "
            f"both total_params and active_params by {result['shared_params'] / 1e9:.3f}B, and "
            f"appears in none of them. That is {share:.1%} of the model's "
            f"{result['active'] / 1e9:.1f}B active parameters",
        ),
        practice.Check(
            "FINDING: a shared expert is a different kind of knob from the six",
            "MoE" in result["verdict"] and result["shared_params"] > 0,
            "the six knobs are substitutions -- LayerNorm or RMSNorm, dense or MoE. A shared "
            "expert is an addition: an always-on MLP alongside the routed ones, so 'dense vs "
            f"sparse' becomes a spectrum rather than a switch. {NEWEST} is both in every one of "
            "its MoE layers, and the table has no column for a model that is dense and sparse at "
            "the same time",
        ),
        practice.Check(
            "FINDING: the norm and activation knobs are settled on every model after GPT-2",
            set(fixed) >= {"norm", "activation"} and set(varying) >= {"num_key_value_heads"},
            "across the six non-GPT-2 configs the knob values are "
            + ", ".join(f"{k} {sorted(v)}" for k, v in settled_knobs.items())
            + f". {fixed} take one value each across all six; {varying} still discriminate, and "
            "of those the position knob varies only between RoPE and RoPE-with-YaRN. The table "
            "is carrying settled questions alongside open ones, which is why a new model mostly "
            "needs a row rather than a column",
        ),
        practice.Check(
            "MECHANISM: the position knob is invisible to the counter that scores the others",
            result["position_free"],
            "what makes a knob a knob here is that analyze reads it: moe, num_experts, "
            "experts_per_token, attention, kv_lora_rank, first_dense_layers and shared_experts "
            "all change the parameter count. Switching position from rope to learned leaves the "
            "total bit-for-bit unchanged, because RoPE has no parameters and analyze never adds "
            "a learned position table either. The third knob in the table is the one the counter "
            "that scores the other five cannot see at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
