"""Exercise 5 — the model cannot express the answer it is asked for.

    DvD splits vision and LLM onto separate GPUs. Under what traffic pattern
    does DvD hurt throughput instead of helping?

Reading of the exercise: the question is put to the lesson's own
`dvd_throughput` first, because that is the model on offer, and the algebra of
that model is checked before any traffic pattern is proposed -- it turns out the
answer cannot be expressed in it, which is the useful half of the exercise. The
traffic patterns are then named against the terms the model leaves out.

**ANSWER: it cannot hurt, in this model.** `speedup` is
`(encoder + llm) / max(encoder, llm)`, which is `1 + min/max` -- bounded in
**[1, 2]** for every possible input. Swept over output lengths from 16 to 1,024
tokens the value never leaves that band: 1.43, 1.85, 1.59, **1.29**, 1.15, 1.04.

**FINDING: the model is maximised exactly at balance.** Speed-up is 2.0 when the
two stages cost the same, which for the shipped 300 GFLOP encoder and 8 GFLOP
per output token is **37.5 tokens**. The shipped configuration generates 128, so
it sits at **1.29x** on the far side of the peak -- the longer the reply, the
less DvD buys, and at 1,024 tokens it buys **4%**.

**FINDING: the term that would make it hurt is the one the model does not
have.** Decoupling moves the visual tokens across a wire. At the router's
high-res tier -- 2,048 tokens at a 4,096-wide hidden state in bf16 -- that is
**16.0 MiB** per request, charged at zero GFLOPs here.

**ANSWER: so the three patterns are the three missing terms.** Text-only
traffic, where the vision pool idles and the LLM pool is smaller than it would
have been colocated. Short replies, where the encoder is the bottleneck and the
LLM pool waits -- below 37 tokens the imbalance is the encoder's way. And bursty
traffic, where two independently queued pools each need headroom for the peak,
so the combined provisioning is worse than one pool sized for the same peak.

Structure: `speedup` runs the lesson's own `dvd_throughput`, `SWEEP` is the
output-length curve, `balance_point` solves for where the two stages cost the
same, and `transfer_bytes` prices the term the model omits.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "10-internvl3-native-multimodal"
ENCODER_GFLOPS, LLM_GFLOPS_PER_TOKEN, SHIPPED_TOKENS = 300, 8, 128
SWEEP = (16, 32, 64, 128, 256, 1024)
VISUAL_TOKENS, HIDDEN, DTYPE_BYTES = 2048, 4096, 2


def speedup(ref, tokens):
    return ref.dvd_throughput(ENCODER_GFLOPS, LLM_GFLOPS_PER_TOKEN, tokens)["speedup"]


def balance_point():
    return ENCODER_GFLOPS / LLM_GFLOPS_PER_TOKEN


def transfer_bytes():
    return VISUAL_TOKENS * HIDDEN * DTYPE_BYTES


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    curve = {tokens: round(speedup(ref, tokens), 2) for tokens in SWEEP}
    balanced = ref.dvd_throughput(ENCODER_GFLOPS, ENCODER_GFLOPS, 1)
    shipped = ref.dvd_throughput(ENCODER_GFLOPS, LLM_GFLOPS_PER_TOKEN, SHIPPED_TOKENS)
    return {
        "curve": curve, "shipped": round(shipped["speedup"], 2),
        "colocated": shipped["colocated"], "decoupled": shipped["decoupled"],
        "min": min(curve.values()), "max": max(curve.values()),
        "bounded": all(1.0 <= value <= 2.0 for value in curve.values()),
        "balanced_speedup": round(balanced["speedup"], 2),
        "balance_tokens": balance_point(),
        "peak_token": min(SWEEP, key=lambda t: abs(t - balance_point())),
        "long_reply": curve[max(SWEEP)],
        "transfer": transfer_bytes(),
        "transfer_mib": round(transfer_bytes() / 2 ** 20, 1),
        "transfer_gflops": 0,
        "patterns": 3,
    }


def verify(result):
    curve = result["curve"]
    return [
        practice.Check(
            "ANSWER: it cannot hurt, in this model -- speedup is 1 + min/max, bounded in [1, 2]",
            all([result["bounded"], result["min"] >= 1.0, result["max"] <= 2.0,
                 curve == {16: 1.43, 32: 1.85, 64: 1.59, 128: 1.29, 256: 1.15, 1024: 1.04}]),
            f"dvd_throughput returns (encoder + llm) / max(encoder, llm), which is "
            f"1 + min/max and can never fall below 1. Swept over output lengths "
            f"{list(SWEEP)} it gives {list(curve.values())} -- between "
            f"{result['min']} and {result['max']}, never outside",
        ),
        practice.Check(
            "FINDING: the model is maximised exactly at balance",
            all([result["balanced_speedup"] == 2.0, result["balance_tokens"] == 37.5,
                 result["peak_token"] == 32, curve[32] == result["max"]]),
            f"speed-up is {result['balanced_speedup']} when the two stages cost the same, "
            f"which for a {ENCODER_GFLOPS} GFLOP encoder at {LLM_GFLOPS_PER_TOKEN} GFLOPs a "
            f"token is {result['balance_tokens']} output tokens. The nearest swept point, "
            f"{result['peak_token']}, is the curve's maximum",
        ),
        practice.Check(
            "FINDING: the shipped configuration sits past the peak",
            all([result["shipped"] == 1.29, result["colocated"] == 1324,
                 result["decoupled"] == 1024, result["long_reply"] == 1.04]),
            f"at {SHIPPED_TOKENS} output tokens the colocated cost is "
            f"{result['colocated']:,} GFLOPs against a decoupled bottleneck of "
            f"{result['decoupled']:,} -- {result['shipped']}x. The longer the reply the less "
            f"DvD buys, and at {max(SWEEP):,} tokens it buys {result['long_reply']}x",
        ),
        practice.Check(
            "FINDING: the term that would make it hurt is the one the model does not have",
            all([result["transfer"] == 16_777_216, result["transfer_mib"] == 16.0,
                 result["transfer_gflops"] == 0, result["patterns"] == 3]),
            f"decoupling moves the visual tokens across a wire: {VISUAL_TOKENS:,} tokens at a "
            f"{HIDDEN:,}-wide hidden state in bf16 is {result['transfer_mib']} MiB a request, "
            f"charged at {result['transfer_gflops']} GFLOPs here. Text-only traffic, replies "
            f"shorter than {result['balance_tokens']} tokens, and bursty arrival are the "
            f"{result['patterns']} patterns that hurt, and none of them has a term in this "
            "function",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
