"""Exercise 3 — the answer is 5.0000% at every sequence length, and seq=8192 is where that is true.

    Implement selective checkpointing: store the attention-module input but not
    its intermediates. Measure the FLOP overhead vs full-layer checkpointing for
    a 32-layer model at seq=8192.

Reading of the exercise: the measurement is run through the lesson's own
`checkpoint_cost(selective=True)` at the configuration named, and then at four
other sequence lengths, because the one number the exercise varies is the one
the function does not take. The lesson's own Key Terms definition of the
attention-softmax volume is then used to compute what the parameter should have
been.

**ANSWER: 5.0000% selective against 32.2917% full, a 6.46x reduction** -- and
5.0000% at seq=1024, 2048, 4096, 8192 and 16384 alike. `checkpoint_cost` has no
`seq` parameter, and its selective branch ignores `segment_size` too, so the
exercise's "for a 32-layer model at seq=8192" narrows nothing: the same call
returns the same number for every model the module can describe.

**FINDING: seq=8192 is the one place in the range where the constant is
correct.** The attention share of a layer's forward FLOPs is
`4*s^2*h / (24*s*h^2 + 4*s^2*h)` = `s / (6h + s)`. At h=8192 that is:

    seq    1024     2048     4096     8192     16384
    true   0.0204   0.0400   0.0769   0.1429   0.2500
    over   0.68%    1.33%    2.56%    4.76%    8.33%

The default `attention_fraction=0.15` is within 5% of the truth at exactly
seq=8192 and wrong by **7.4x** at seq=1024. The parameterisation is invisible
precisely because of where the exercise evaluates it.

**FINDING: the half of selective checkpointing that needs the sequence length
has no function at all.** Selective recomputation is a memory trade -- recompute
the cheap-in-FLOPs, huge-in-bytes softmax and keep the rest -- but
`memory_after_checkpoint` takes only `segment_size`, with no selective mode, so
the saving cannot be computed. The exercise asks for the FLOP overhead, which is
the half that does not need seq.

**MECHANISM: the memory model is linear in the sequence length.**
`activation_memory_mb` is `12*b*s*h*bytes`, 16.0x from seq 1024 to 16384. The
`s^2` softmax volume the lesson's own Key Terms calls "the O(L^2) problem" that
"dominates activation memory at long contexts" is **5.33x the entire modelled
per-layer figure** at seq=8192 and 10.67x at 16384, and it is not in the model.

Structure: `attention_share` is the closed form the default approximates;
`softmax_mb` is the lesson's own Key Terms definition of the missing term.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "34-gradient-checkpointing"
DEPTH, HIDDEN, HEADS, BATCH, BYTES = 32, 8192, 64, 1, 2
LENGTHS = (1024, 2048, 4096, 8192, 16384)
ASKED = 8192
DEFAULT = 0.15
SEGMENTS = (1, 4, 8, 32)


def attention_share(seq, hidden=HIDDEN):
    """Attention's share of a layer's forward FLOPs: 4 s^2 h over 24 s h^2 + 4 s^2 h."""
    return seq / (6 * hidden + seq)


def softmax_mb(seq, heads=HEADS, batch=BATCH, bytes_per_value=BYTES):
    """The Key Terms definition: s^2 * heads * batch floats, per layer."""
    return seq * seq * heads * batch * bytes_per_value / 1e6


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    selective = ref.checkpoint_cost(DEPTH, selective=True)["overhead_vs_no_ckpt"]
    full = ref.checkpoint_cost(DEPTH, segment_size=DEPTH)["overhead_vs_no_ckpt"]
    modelled = {seq: ref.activation_memory_mb(DEPTH, hidden=HIDDEN, seq=seq, batch=BATCH,
                                              bytes_per_value=BYTES) / DEPTH
                for seq in LENGTHS}
    return {
        "selective": selective,
        "full": full,
        "by_segment": {k: ref.checkpoint_cost(DEPTH, segment_size=k,
                                              selective=True)["overhead_vs_no_ckpt"]
                       for k in SEGMENTS},
        "cost_params": list(inspect.signature(ref.checkpoint_cost).parameters),
        "memory_params": list(inspect.signature(ref.memory_after_checkpoint).parameters),
        "true_share": {seq: attention_share(seq) for seq in LENGTHS},
        "true_overhead": {seq: attention_share(seq) / 3 for seq in LENGTHS},
        "modelled_mb": modelled,
        "softmax_mb": {seq: softmax_mb(seq) for seq in LENGTHS},
        "linear": modelled[16384] / modelled[1024],
        "selective_names": [name for name in dir(ref) if "selective" in name.lower()],
    }


def listing(values, fmt):
    return ", ".join(f"seq={seq} {format(value, fmt)}" for seq, value in values.items())


def verify(result):
    share, true_overhead = result["true_share"], result["true_overhead"]
    modelled, softmax = result["modelled_mb"], result["softmax_mb"]
    return [
        practice.Check(
            "ANSWER: 5.0000% selective against 32.2917% full, and 5.0000% at every seq",
            ("seq" not in result["cost_params"]
             and len(set(result["by_segment"].values())) == 1
             and result["full"] / result["selective"] > 6),
            f"checkpoint_cost(32, selective=True) is {result['selective']:.4%} against "
            f"{result['full']:.4%} for full checkpointing at k=32, a "
            f"{result['full'] / result['selective']:.2f}x reduction. It takes "
            f"{result['cost_params']} -- no seq -- and its selective branch ignores "
            "segment_size too: " + listing(result["by_segment"], ".4%").replace("seq=", "k=")
            + ". 'For a 32-layer model at seq=8192' narrows nothing",
        ),
        practice.Check(
            f"FINDING: seq={ASKED} is the one length in the range where 0.15 is right",
            (abs(share[ASKED] / DEFAULT - 1) < 0.1
             and DEFAULT / share[min(LENGTHS)] > 5),
            "the attention share of a layer's forward FLOPs is s / (6h + s), which at h=8192 "
            "gives " + listing(share, ".4f") + f" -- so the {DEFAULT} default is within "
            f"{abs(100 * (DEFAULT / share[ASKED] - 1)):.0f}% of the truth at exactly seq={ASKED} "
            f"and off by {DEFAULT / share[min(LENGTHS)]:.1f}x at seq={min(LENGTHS)}. The honest "
            "overheads are " + listing(true_overhead, ".2%")
            + ", a 12x spread the constant cannot show",
        ),
        practice.Check(
            "FINDING: the half that needs the sequence length has no function",
            (not result["selective_names"]
             and "seq" in result["memory_params"]
             and "selective" not in result["memory_params"]),
            f"selective recomputation is a memory trade -- redo the cheap-in-FLOPs, "
            f"huge-in-bytes softmax and keep the rest -- but memory_after_checkpoint takes "
            f"{result['memory_params']}, with no selective mode, and no name in the module "
            f"mentions selective at all ({result['selective_names']}). So the saving cannot be "
            "computed, and the exercise asks for the FLOP overhead, the half that does not need "
            "the sequence length",
        ),
        practice.Check(
            "MECHANISM: the memory model is linear in seq and the s^2 term is absent",
            (abs(result["linear"] - 16.0) < 1e-6
             and softmax[ASKED] / modelled[ASKED] > 5),
            f"activation_memory_mb is 12*b*s*h*bytes, so it grows {result['linear']:.1f}x from "
            f"seq 1024 to 16384 -- exactly 16x, linear. The s^2 softmax volume the lesson's own "
            f"Key Terms calls the O(L^2) problem that dominates at long contexts is "
            + listing({seq: softmax[seq] / modelled[seq] for seq in LENGTHS}, ".2f")
            + " times the entire modelled per-layer figure, and it is not in the model. At "
            f"seq={ASKED} that is {softmax[ASKED]:,.0f} MB per layer against a modelled "
            f"{modelled[ASKED]:,.0f} MB",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
