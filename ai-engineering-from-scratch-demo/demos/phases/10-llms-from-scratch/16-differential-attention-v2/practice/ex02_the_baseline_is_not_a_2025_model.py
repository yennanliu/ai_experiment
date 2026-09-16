"""Exercise 2 — V1 costs 8,192 parameters, and V2's 12.5% is 80% once the baseline has GQA.

    Compute the parameter-count delta from baseline to DIFF V1 and from baseline
    to DIFF V2 for a 7B-class model (hidden=4096, heads=32, d_head=128, 32
    layers). Show which components gained parameters and which stayed the same.

Reading of the exercise: the counts are the lesson's own
`compute_param_diff` at the shape the exercise names, broken back down into Q,
K, V, O and lambda by calling the three `attention_params_*` functions
component by component. The baseline those deltas are measured against is then
checked, because `attention_params_baseline` is `4 * hidden^2` -- multi-head
attention -- and every other model in this phase is grouped-query.

**ANSWER: per layer, V1 adds 8,192 parameters and V2 adds 8,404,992.**

    component   baseline      DIFF V1       DIFF V2
    Q          16,777,216   16,777,216   33,554,432   x2
    K          16,777,216   16,777,216    4,194,304   /4
    V          16,777,216   16,777,216    4,194,304   /4
    O          16,777,216   16,777,216   33,554,432   x2
    lambda              0        8,192       16,384
    total      67,108,864   67,117,056   75,513,856
    delta                      +8,192   +8,404,992  (+0.01%, +12.5%)

V1 is free: halving `d_head` and running two branches leaves Q and K exactly
where they were, so the only new parameters are the four lambda vectors.

**FINDING: the baseline is multi-head, and nothing in 2025 is.**
`attention_params_baseline` is `4 * hidden^2`, which is MHA. V2's own K and V are
sized at `kv_heads * d_head` -- it is a GQA model -- so the delta compares a GQA
model against an MHA baseline. Against a like-for-like GQA baseline at the same
8 KV heads, V2 costs **41.9M -> 75.5M, +80%**, not +12.5%. The comparison
understates V2 by **6.4x**.

**FINDING: the cost is entirely Q and O, and it is exactly double.** V2's
`q_params` is `hidden * (2 * n_heads * d_head)` and its `o_params` is the
transpose -- twice the baseline's, by construction, because V2's answer to
"where do the two branches come from" is *more heads* rather than *narrower*
ones. K and V are untouched, which is Exercise 5's finding arriving early.

**MECHANISM: V1 spends nothing and V2 spends 33.6M, for the same two branches.**
V1 halves `d_head` to 64 and runs two branches of 32 heads; V2 keeps `d_head` at
128 and runs 64 heads. Same number of `(head, branch)` pairs, same total Q width
in V1's case and double it in V2's -- the parameter delta between the two
versions is the decision to stop halving the head.

Structure: `components` calls the lesson's own three counters and splits each
into Q, K, V, O and lambda; `gqa_baseline` is the like-for-like baseline the
lesson does not ship.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "16-differential-attention-v2"
HIDDEN, HEADS, KV_HEADS = 4096, 32, 8
HEAD_DIM = HIDDEN // HEADS


def components():
    """Each version's Q, K, V, O and lambda, written the way the lesson writes them."""
    half = HEADS * HEAD_DIM // 2
    return {
        "baseline": {"q": HIDDEN * HIDDEN, "k": HIDDEN * HIDDEN,
                     "v": HIDDEN * HIDDEN, "o": HIDDEN * HIDDEN, "lam": 0},
        "diff_v1": {"q": 2 * HIDDEN * half, "k": 2 * HIDDEN * half,
                    "v": HIDDEN * HIDDEN, "o": HIDDEN * HIDDEN,
                    "lam": 4 * HEADS * (HEAD_DIM // 2)},
        "diff_v2": {"q": HIDDEN * (2 * HEADS * HEAD_DIM), "k": HIDDEN * (KV_HEADS * HEAD_DIM),
                    "v": HIDDEN * (KV_HEADS * HEAD_DIM),
                    "o": (2 * HEADS * HEAD_DIM) * HIDDEN, "lam": 4 * HEADS * HEAD_DIM},
    }


def gqa_baseline():
    """The baseline the lesson does not ship: MHA Q and O, GQA K and V at the same 8 heads."""
    return 2 * HIDDEN * HIDDEN + 2 * HIDDEN * (KV_HEADS * HEAD_DIM)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counted = ref.compute_param_diff(hidden=HIDDEN, n_heads=HEADS, kv_heads=KV_HEADS)
    parts = components()
    totals = {name: sum(part.values()) for name, part in parts.items()}
    gqa = gqa_baseline()
    return {
        "parts": parts,
        "totals": totals,
        "reference": (counted.baseline, counted.diff_v1, counted.diff_v2),
        "deltas": (counted.extra_v1, counted.extra_v2),
        "gqa_baseline": gqa,
        "v2_over_gqa": (counted.diff_v2 - gqa) / gqa,
        "v2_over_mha": counted.extra_v2 / counted.baseline,
        "unchanged_v1": [key for key in ("q", "k", "v", "o")
                         if parts["diff_v1"][key] == parts["baseline"][key]],
        "unchanged_v2": [key for key in ("q", "k", "v", "o")
                         if parts["diff_v2"][key] == parts["baseline"][key]],
    }


def verify(result):
    parts, totals = result["parts"], result["totals"]
    base, v1, v2 = result["reference"]
    extra_v1, extra_v2 = result["deltas"]
    return [
        practice.Check(
            "ANSWER: V1 adds 8,192 parameters per layer and V2 adds 8,404,992",
            totals["baseline"] == base and totals["diff_v1"] == v1 and totals["diff_v2"] == v2,
            f"the component split reproduces the lesson's own totals exactly -- baseline "
            f"{base:,}, V1 {v1:,}, V2 {v2:,} -- so the deltas are {extra_v1:+,} and "
            f"{extra_v2:+,}, or {extra_v1 / base:+.2%} and {extra_v2 / base:+.1%}. V1 is free: "
            f"halving d_head to {HEAD_DIM // 2} and running two branches leaves Q and K exactly "
            f"where they were, and the only new parameters are the four lambda vectors at "
            f"{parts['diff_v1']['lam']:,}",
        ),
        practice.Check(
            "FINDING: the baseline is multi-head, and V2 is not",
            parts["diff_v2"]["k"] < parts["baseline"]["k"]
            and result["v2_over_gqa"] > 5 * result["v2_over_mha"],
            f"attention_params_baseline is 4 x hidden^2, which is MHA, while V2's own K and V are "
            f"sized at kv_heads x d_head -- {parts['diff_v2']['k']:,} against the baseline's "
            f"{parts['baseline']['k']:,} -- so the delta compares a grouped-query model against a "
            f"multi-head baseline. Against a like-for-like GQA baseline at the same "
            f"{KV_HEADS} KV heads, {result['gqa_baseline']:,}, V2 costs "
            f"{result['v2_over_gqa']:+.0%} rather than {result['v2_over_mha']:+.1%}: the "
            f"comparison understates it by {result['v2_over_gqa'] / result['v2_over_mha']:.1f}x",
        ),
        practice.Check(
            "FINDING: the cost is entirely Q and O, and it is exactly double",
            parts["diff_v2"]["q"] == 2 * parts["baseline"]["q"]
            and parts["diff_v2"]["o"] == 2 * parts["baseline"]["o"],
            f"V2's q_params is hidden x (2 x n_heads x d_head) = {parts['diff_v2']['q']:,} and "
            f"its o_params is the transpose -- exactly twice the baseline's "
            f"{parts['baseline']['q']:,} each, by construction, because V2's answer to where the "
            f"two branches come from is more heads rather than narrower ones. K and V are left "
            "alone, which is Exercise 5's finding arriving early",
        ),
        practice.Check(
            "MECHANISM: V1 spends nothing and V2 spends 33.6M for the same two branches",
            set(result["unchanged_v1"]) == {"q", "k", "v", "o"} and not result["unchanged_v2"],
            f"V1 halves d_head to {HEAD_DIM // 2} and runs two branches of {HEADS} heads, so "
            f"{result['unchanged_v1']} are bit-for-bit the baseline's. V2 keeps d_head at "
            f"{HEAD_DIM} and runs {2 * HEADS} heads, so {result['unchanged_v2']} matches nothing "
            f"-- the {parts['diff_v2']['q'] - parts['baseline']['q']:,} extra in Q and the same "
            "again in O are the whole delta between the two versions, and they are the decision "
            "to stop halving the head",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
