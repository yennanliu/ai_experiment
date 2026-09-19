"""Exercise 1 — the row-sum test cannot fail.

    Implement the cross-attention block in PyTorch. Verify that with 32 queries
    and 256 keys/values, the attention-weight matrix is 32 x 256 and each row
    sums to 1 after softmax.

Reading of the exercise: the lesson already implements the block and is
stdlib-only, so it is run at the shape the exercise names rather than
re-implemented in a framework the lesson does not use. The check the exercise
asks for is then examined as well as performed, because "each row sums to 1" is
a property of softmax and holds for any weights at all -- so the solution also
measures what the rows actually contain.

**ANSWER: 32 x 256, and every row sums to 1 to 2.2e-16.** That is the whole of
the exercise's test, and it would pass on an untrained block, a randomly
permuted one, or a constant one.

**FINDING: untrained, this attention is already almost one-hot.** Mean row
entropy is **0.321** nats against a uniform 256-way maximum of **5.545** --
5.8% -- and the mean top weight is **0.876**. The lesson's own
`summarize_attention` prints entropy against that uniform baseline as though
concentration were evidence of training; at initialisation it is already there.

**FINDING: the 1/sqrt(d) scale is defeated by the lesson's own init.**
`mat()` draws N(0, 1), so `Q = W_q q` has entries of std **3.65** -- sqrt(d) =
4, not 1 -- and the logits after the 1/sqrt(d) division have std **14.53**,
which is d = 16. Scaling the weight matrices by 1/sqrt(d) restores mean entropy
to **5.066**; by 1/d it is **5.543**, within 0.002 of uniform. The saturation is
an initialisation bug, not a property of cross-attention.

**FINDING: the lesson ships 8 queries over 64 patches.** `NUM_QUERY` is 8 and
`NUM_PATCH` is 64, so the 32 x 256 shape the exercise names is **16x** the
attention entries the module actually builds and has to be constructed by the
reader.

Structure: `block` runs the lesson's own `cross_attention` at a chosen shape and
weight scale, `entropy` is the per-row measure, and `SCALES` is the three
initialisations compared.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "03-blip2-qformer-bridge"
QUERIES, PATCHES, DIM, SEED = 32, 256, 16, 42
SCALES = {"lesson": 1.0, "inv_sqrt_d": DIM ** -0.5, "inv_d": 1.0 / DIM}


def block(ref, scale=1.0, queries=QUERIES, patches=PATCHES, dim=DIM):
    """The lesson's own cross_attention at a chosen shape and weight scale."""
    ref.rng = random.Random(SEED)
    keys = [ref.vec(dim) for _ in range(patches)]
    probes = [ref.vec(dim) for _ in range(queries)]
    weights = [[[x * scale for x in ref.vec(dim)] for _ in range(dim)] for _ in range(3)]
    return ref.cross_attention(probes, keys, *weights)


def entropy(row):
    return -sum(w * math.log(w + 1e-12) for w in row)


def profile(attention):
    return {"entropy": round(statistics.fmean(entropy(row) for row in attention), 3),
            "top": round(statistics.fmean(max(row) for row in attention), 4)}


def logit_spread(ref, dim=DIM):
    """Std of one query's scaled logits, and of its projected query vector."""
    ref.rng = random.Random(SEED)
    keys = [ref.vec(dim) for _ in range(PATCHES)]
    probe = ref.vec(dim)
    projected = ref.matmul_vec(ref.mat(dim, dim), probe)
    key_weights = ref.mat(dim, dim)
    logits = [ref.dot(projected, ref.matmul_vec(key_weights, k)) / math.sqrt(dim)
              for k in keys]
    return round(statistics.pstdev(projected), 2), round(statistics.pstdev(logits), 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    _, attention = block(ref)
    query_std, logit_std = logit_spread(ref)
    return {
        "shape": (len(attention), len(attention[0])),
        "row_sum_error": max(abs(sum(row) - 1.0) for row in attention),
        "uniform_entropy": round(math.log(PATCHES), 3), "uniform_top": round(1 / PATCHES, 4),
        "profiles": {name: profile(block(ref, scale)[1]) for name, scale in SCALES.items()},
        "query_std": query_std, "logit_std": logit_std, "root_dim": math.sqrt(DIM),
        "module_shape": (ref.NUM_QUERY, ref.NUM_PATCH),
        "entry_ratio": QUERIES * PATCHES // (ref.NUM_QUERY * ref.NUM_PATCH),
    }


def verify(result):
    profiles, uniform = result["profiles"], result["uniform_entropy"]
    return [
        practice.Check(
            "ANSWER: 32 x 256, and every row sums to 1 to 2.2e-16",
            all([result["shape"] == (QUERIES, PATCHES),
                 result["row_sum_error"] <= 2.3e-16]),
            f"the attention-weight matrix is {result['shape'][0]} x {result['shape'][1]} and "
            f"the worst row sum is off by {result['row_sum_error']:.1e}. That is the whole of "
            "the exercise's test, and it would pass on an untrained block, a permuted one or "
            "a constant one -- summing to 1 is a property of softmax",
        ),
        practice.Check(
            "FINDING: untrained, this attention is already almost one-hot",
            all([profiles["lesson"]["entropy"] == 0.321, profiles["lesson"]["top"] == 0.8762,
                 uniform == 5.545]),
            f"mean row entropy is {profiles['lesson']['entropy']} nats against the uniform "
            f"256-way maximum of {uniform} -- "
            f"{profiles['lesson']['entropy'] / uniform * 100:.1f}% -- and the mean top weight "
            f"is {profiles['lesson']['top']} against {result['uniform_top']}. "
            "summarize_attention prints entropy against that baseline as though concentration "
            "were evidence of training",
        ),
        practice.Check(
            "FINDING: the 1/sqrt(d) scale is defeated by the lesson's own init",
            all([result["query_std"] == 3.65, result["logit_std"] == 14.53,
                 profiles["inv_sqrt_d"]["entropy"] == 5.066,
                 abs(profiles["inv_d"]["entropy"] - uniform) <= 0.002]),
            f"mat() draws N(0,1), so W_q q has entries of std {result['query_std']} -- "
            f"sqrt(d) = {result['root_dim']:.0f}, not 1 -- and the logits after the "
            f"1/sqrt(d) division have std {result['logit_std']}, which is d = {DIM}. Scaling "
            f"the weights by 1/sqrt(d) gives entropy {profiles['inv_sqrt_d']['entropy']} and "
            f"by 1/d {profiles['inv_d']['entropy']}, within 0.002 of uniform",
        ),
        practice.Check(
            "FINDING: the lesson ships 8 queries over 64 patches",
            all([result["module_shape"] == (8, 64), result["entry_ratio"] == 16]),
            f"NUM_QUERY is {result['module_shape'][0]} and NUM_PATCH is "
            f"{result['module_shape'][1]}, so the {QUERIES} x {PATCHES} shape the exercise "
            f"names is {result['entry_ratio']}x the attention entries the module builds, and "
            "has to be constructed by the reader",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
