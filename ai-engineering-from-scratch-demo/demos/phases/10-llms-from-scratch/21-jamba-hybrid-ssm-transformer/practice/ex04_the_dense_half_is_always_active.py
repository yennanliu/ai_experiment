"""Exercise 4 — 23.6% against 5.5%, and the 4.3x is the half of Jamba that has no experts to skip.

    Compute the parameter overhead of MoE-every-other-layer in Jamba 1.5 Large
    (398B total, 94B active). Compare the active ratio to DeepSeek-V3 (37B/671B)
    and explain why Jamba's architecture pushes the active ratio higher.

Reading of the exercise: the two published pairs are taken as given, because the
lesson's calculator has no parameter count in it at all -- its `HybridConfig`
holds layer counts and head dimensions and nothing that could be summed into a
model size. The explanation is then made arithmetic by reconstructing the
active-parameter floor each architecture imposes, which is what the ratio is
measuring.

**ANSWER: 23.6% against 5.5% -- a factor of 4.3.**

    model               total    active   active ratio
    Jamba 1.5 Large      398B      94B       23.6%
    DeepSeek-V3          671B      37B        5.5%

**MECHANISM: Jamba routes on half its layers and DeepSeek routes on 95% of
them.** DeepSeek's first 3 of 61 layers are dense and the other 58 are MoE, so
**95.1%** of its layers can skip experts. Jamba alternates, so **50%** can --
and the Mamba layers that fill the other half have no experts to skip by
construction, because the Mamba block is not an MLP with a router in front of it.
Every parameter in those layers is active on every token.

**FINDING: the ratio is a floor plus a fraction, and Jamba's floor is the
architecture.** Write the active count as `dense_layers + top_k/experts *
moe_layers`. At DeepSeek's 58 of 61 MoE layers and 8 of 257 experts the routed
share of the model is tiny; at Jamba's 1 in 2 it cannot fall below half the
model whatever the expert count is. **50%** of Jamba's layers are a hard floor on
its active ratio in layer terms.

**FINDING: the calculator cannot price any of this.** `HybridConfig` has no
`num_experts`, no `experts_per_token`, no `intermediate_size` and no
`vocab_size`; `kv_cache_bytes` and `ssm_state_bytes` are the only two functions
in the file. The exercise's whole subject -- parameters -- is outside the module
it is attached to.

**FINDING: the two models are not comparable on this axis anyway.** DeepSeek's
5.5% is a statement about *expert* sparsity; Jamba's 23.6% is that plus the
memory story Exercises 1 and 2 are about, which DeepSeek does not have at all.
A hybrid buys long-context memory and gives up sparsity; a dense-attention MoE
does the reverse. The ratio compares them on the axis where one of them is not
trying.

Structure: `published` holds the two reported pairs; `layer_split` reconstructs
what fraction of each model's layers can route.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "21-jamba-hybrid-ssm-transformer"
JAMBA = {"total": 398e9, "active": 94e9, "layers": 32, "moe_layers": 16}
DEEPSEEK = {"total": 671e9, "active": 37e9, "layers": 61, "moe_layers": 58}


def published():
    return {"Jamba 1.5 Large": JAMBA, "DeepSeek-V3": DEEPSEEK}


def layer_split(model):
    """What fraction of a model's layers have experts to skip."""
    return {"ratio": model["active"] / model["total"],
            "routable": model["moe_layers"] / model["layers"],
            "always_on": 1 - model["moe_layers"] / model["layers"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: dict(model, **layer_split(model)) for name, model in published().items()}
    fields = [field.name for field in dataclasses.fields(ref.HybridConfig)]
    functions = [name for name in dir(ref)
                 if callable(getattr(ref, name)) and not name.startswith("_")
                 and name not in ("HybridConfig", "dataclass", "annotations")]
    return {
        "rows": rows,
        "gap": (rows["Jamba 1.5 Large"]["ratio"] / rows["DeepSeek-V3"]["ratio"]),
        "fields": fields,
        "param_fields": [f for f in fields if any(word in f for word in
                                                  ("expert", "vocab", "intermediate", "param"))],
        "functions": sorted(functions),
    }


def column(rows, field, fmt):
    return ", ".join(f"{name} {format(row[field], fmt)}" for name, row in rows.items())


def verify(result):
    rows = result["rows"]
    jamba, deepseek = rows["Jamba 1.5 Large"], rows["DeepSeek-V3"]
    return [
        practice.Check(
            "ANSWER: 23.6% against 5.5% -- a factor of 4.3",
            abs(jamba["ratio"] - 0.236) < 0.002 and abs(result["gap"] - 4.3) < 0.1,
            "the two published pairs give active ratios of " + column(rows, "ratio", ".1%")
            + f", so Jamba's is {result['gap']:.1f}x DeepSeek's on totals of "
            + column(rows, "total", ",.0f")
            + " and active counts of " + column(rows, "active", ",.0f"),
        ),
        practice.Check(
            "MECHANISM: Jamba routes on half its layers and DeepSeek routes on 95% of them",
            abs(jamba["routable"] - 0.5) < 1e-9 and deepseek["routable"] > 0.9,
            f"DeepSeek's first {DEEPSEEK['layers'] - DEEPSEEK['moe_layers']} of "
            f"{DEEPSEEK['layers']} layers are dense and the rest are MoE, so "
            f"{deepseek['routable']:.1%} of its layers can skip experts. Jamba alternates, so "
            f"{jamba['routable']:.0%} can -- and the Mamba layers filling the other half have no "
            "experts to skip by construction, because a Mamba block is not an MLP with a router "
            "in front of it. Every parameter in those layers is active on every token",
        ),
        practice.Check(
            "FINDING: the ratio is a floor plus a fraction, and Jamba's floor is the architecture",
            jamba["always_on"] > 10 * deepseek["always_on"],
            f"write the active count as dense_layers + (top_k / experts) x moe_layers. The share "
            f"of layers with nothing to skip is " + column(rows, "always_on", ".1%")
            + f" -- Jamba's is {jamba['always_on'] / deepseek['always_on']:.0f}x DeepSeek's -- so "
            "Jamba's active ratio cannot fall below half the model in layer terms whatever its "
            "expert count is, while DeepSeek's floor is three layers of 61",
        ),
        practice.Check(
            "FINDING: the calculator cannot price any of this",
            not result["param_fields"] and len(result["functions"]) <= 4,
            f"HybridConfig is {result['fields']} -- no num_experts, no experts_per_token, no "
            f"intermediate_size, no vocab_size -- and the module's callables are "
            f"{result['functions']}. Both of the two that compute anything return bytes of cache. "
            "The exercise's whole subject, parameters, is outside the module it is attached to",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
