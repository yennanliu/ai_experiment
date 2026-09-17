"""Exercise 3 — the choice between Mamba-1 and Mamba-2 reaches this calculator as one field, and it cannot matter.

    Read Section 3 of the Jamba paper (arXiv:2403.19887). Explain why AI21 uses
    Mamba-1 rather than Mamba-2 despite Mamba-2 being faster. Hint: the hybrid
    ablation section documents this.

Reading of the exercise: the explanation is written as an audit of what the
lesson can express about the two, because the reason Jamba gives -- Mamba-2's
larger state and its interaction with the attention layers in the hybrid -- is a
claim about memory, and this module is a memory calculator. Every place the
distinction could enter is enumerated and then swept.

**ANSWER: the only field that distinguishes them is `ssm_state_size`, and
raising it 16x changes nothing that matters.**

    ssm_state_size   SSM state at 256k   as a share of the KV cache
          16             3.50 MB               0.0214%
          64            14.00 MB               0.0854%
         128            28.00 MB               0.1709%
         256            56.00 MB               0.3418%

Mamba-2's state is roughly an order of magnitude wider than Mamba-1's. On this
calculator that takes the SSM term from 0.02% of the hybrid's cache to 0.34% --
still three decimal places below the number the lesson exists to report.

**MECHANISM: the state is constant in context and the cache is linear in it.**
`ssm_state_bytes` takes no context argument at all. So at any context long
enough for the hybrid to be interesting, the Mamba-1-versus-Mamba-2 decision is
invisible to every figure this module prints -- and the reason AI21 gives is not
a memory reason the file can hold.

**FINDING: `HybridConfig` has eight fields and none of them is a Mamba
version.** `name, total_layers, attn_layers, hidden, n_q_heads, n_kv_heads,
head_dim, ssm_state_size` -- the discretisation, the state's realness, the
head structure of the SSM and the presence of an outer convolution, which are
what actually separate the two, have no representation.

**FINDING: the state would have to be 64,000 wide to reach 1 GB.** Solving
`28 * hidden * state * 2 = 1 GB` gives **state = 4,681**, and at that width the
SSM term is still 6% of the 1:7 hybrid's KV cache. There is no setting of the
one available knob at which the exercise's question becomes a memory question.

Structure: `state_sweep` runs the lesson's own `ssm_state_bytes` across widths;
`width_for` inverts it to ask what width a given memory target implies.
"""

from __future__ import annotations

import dataclasses
import inspect

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "21-jamba-hybrid-ssm-transformer"
CONTEXT, BYTES, GIB, MIB = 262_144, 2, 1024 ** 3, 1024 ** 2
LAYERS, ATTN, HIDDEN, HEAD_DIM = 32, 4, 4096, 128
WIDTHS = (16, 64, 128, 256)


def shape(ref, state):
    return ref.HybridConfig(name=f"state {state}", total_layers=LAYERS, attn_layers=ATTN,
                            hidden=HIDDEN, n_q_heads=32, n_kv_heads=32,
                            head_dim=HEAD_DIM, ssm_state_size=state)


def state_sweep(ref):
    rows = {}
    for width in WIDTHS:
        cfg = shape(ref, width)
        kv = ref.kv_cache_bytes(cfg, CONTEXT, BYTES)
        state = ref.ssm_state_bytes(cfg, BYTES)
        rows[width] = {"state_mb": state / MIB, "share": state / kv, "kv_gb": kv / GIB}
    return rows


def width_for(target_bytes):
    """The state width at which the SSM term reaches `target_bytes`."""
    return target_bytes / ((LAYERS - ATTN) * HIDDEN * BYTES)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = state_sweep(ref)
    cfg = shape(ref, 16)
    gib_width = width_for(GIB)
    return {
        "rows": rows,
        "fields": [field.name for field in dataclasses.fields(ref.HybridConfig)],
        "context_free": "ctx" not in inspect.signature(ref.ssm_state_bytes).parameters,
        "state_params": list(inspect.signature(ref.ssm_state_bytes).parameters),
        "kv_params": list(inspect.signature(ref.kv_cache_bytes).parameters),
        "gib_width": gib_width,
        "gib_share": ((LAYERS - ATTN) * HIDDEN * gib_width * BYTES
                      / ref.kv_cache_bytes(cfg, CONTEXT, BYTES)),
        "growth": rows[256]["share"] / rows[16]["share"],
    }


def column(rows, field, fmt):
    return ", ".join(f"{width} {format(row[field], fmt)}" for width, row in rows.items())


def verify(result):
    rows = result["rows"]
    return [
        practice.Check(
            "ANSWER: the only distinguishing field is ssm_state_size, and 16x changes nothing",
            rows[256]["share"] < 0.005 and abs(result["growth"] - 16) < 1e-9,
            "sweeping the state width gives SSM totals of " + column(rows, "state_mb", ".2f")
            + " MB, which as a share of the hybrid's KV cache is " + column(rows, "share", ".4%")
            + f". Mamba-2's state is roughly an order of magnitude wider than Mamba-1's, and on "
            f"this calculator that takes the SSM term from {rows[16]['share']:.4%} to "
            f"{rows[256]['share']:.4%} -- still three decimal places below the number the module "
            "exists to report",
        ),
        practice.Check(
            "MECHANISM: the state is constant in context and the cache is linear in it",
            result["context_free"],
            f"ssm_state_bytes takes no context argument at all -- its parameters are "
            f"{result['state_params']} against kv_cache_bytes' {result['kv_params']}. So at any "
            "context "
            f"long enough for the hybrid to be interesting, the Mamba-1-versus-Mamba-2 decision is "
            f"invisible to every figure this module prints, and the reason AI21 gives is not a "
            "memory reason the file can hold",
        ),
        practice.Check(
            "FINDING: HybridConfig has eight fields and none of them is a Mamba version",
            len(result["fields"]) == 8 and "ssm_state_size" in result["fields"],
            f"the config is {result['fields']}. The discretisation, whether the state is complex "
            "or real, the SSM's head structure and the presence of an outer convolution -- which "
            "are what actually separate Mamba-1 from Mamba-2 -- have no representation here, so "
            "the choice reaches the calculator as one integer or not at all",
        ),
        practice.Check(
            "FINDING: the state would have to be 4,681 wide to reach 1 GB",
            result["gib_width"] > 1000 and result["gib_share"] < 0.1,
            f"solving {LAYERS - ATTN} x {HIDDEN} x state x {BYTES} = 1 GB gives state = "
            f"{result['gib_width']:,.0f}, roughly {result['gib_width'] / 16:.0f}x Mamba-1's width, "
            f"and even there the SSM term is {result['gib_share']:.1%} of the 1:7 hybrid's KV "
            "cache. There is no setting of the one available knob at which the exercise's "
            "question becomes a memory question",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
