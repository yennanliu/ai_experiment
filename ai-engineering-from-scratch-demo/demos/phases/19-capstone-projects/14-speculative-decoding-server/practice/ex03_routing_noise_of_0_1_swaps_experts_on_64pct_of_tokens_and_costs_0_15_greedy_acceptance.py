"""Exercise 3 -- routing noise of 0.1 swaps experts on 64% of tokens and costs 0.15 greedy acceptance.

    Run a controlled MoE experiment: same Qwen3-Coder-30B with routing noise injected vs without. Measure draft acceptance sensitivity.

Reading of the exercise: the target's `distribution` is replaced by one
routed MoE layer with Qwen3-Coder-30B-A3B's routing shape: 128 experts, top
8, and gates renormalised over the chosen 8 (`num_experts`,
`num_experts_per_tok` and `norm_topk_prob` in its config.json, read
2026-09-29 at
https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/raw/main/config.json).
Each expert is a sharpened lesson distribution. Router logits are seeded per
position. Routing noise is N(0, sigma) added to every logit on every target
call, freshly drawn, as batch-dependent routing does. The draft is the
lesson's `DraftModel` (alignment 0.9, k = 4), aligned to the noise-free
router. Everything else is held fixed: the same seeds, and the lesson's
`speculative_decode` over 4,000 tokens. Sensitivity is reported as the
lesson's `acceptance_rate` and as the expected greedy acceptance, meaning
the chance that a drafted token is the noisy target's argmax, each over
sigma in {0 .. 1}.

**ANSWER: greedy acceptance falls 0.152 by sigma = 0.1, a slope of -1.52 per
unit of router-logit noise.**

    sigma  expert set changed  argmax kept  greedy acc  lesson acc  tok/call
    0      0.000               1.000        0.920       0.924       4.69
    0.05   0.378               0.914        0.843       0.907       4.62
    0.1    0.636               0.832        0.768       0.899       4.60
    0.2    0.879               0.729        0.675       0.899       4.60
    0.5    0.993               0.509        0.476       0.776       4.10
    1.0    1.000               0.306        0.291       0.495       2.98

Most of the sensitivity comes from the top-8 boundary. The mean gap between
the 8th and 9th router logit is 0.062, with a median of 0.044, so noise of
0.05 already swaps an expert on 37.8% of tokens.

**FINDING: the lesson's acceptance metric reads the same noise as 6x
weaker.** Over the same 0 -> 0.1 step it moves 0.025 (slope -0.25). It
reads 0.899 at both sigma = 0.1 and sigma = 0.2, while greedy acceptance
drops a further 0.093. Its 0.5 x max rule accepts the near-ties that noise
creates. A dashboard on the lesson's metric would report MoE routing noise
as nearly harmless up to 0.2.

Structure: `experts()` builds 128 expert distributions; `moe()` routes one
position; `run()` swaps it into the lesson's `TargetModel` and measures;
`boundary_gap()` measures the top-8 margin.
"""

from __future__ import annotations

import math
import random
import statistics
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "14-speculative-decoding-server"
E, TOP, K, ALIGN, N, PROBE = 128, 8, 4, 0.9, 4000, 2000
SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def experts(ref):
    out = []
    for e in range(E):
        w = [p**4 for p in ref.softmax_from(1000 + e)]
        out.append([x / sum(w) for x in w])
    return out


def moe(ref, table, s, sigma, noise):
    """One routed layer: seeded router logits, plus N(0, sigma) routing noise per call."""
    r = random.Random(s * 7 + 13)
    logits = [r.gauss(0, 1) + sigma * noise.gauss(0, 1) for _ in range(E)]
    top = sorted(range(E), key=lambda i: -logits[i])[:TOP]
    gate = [math.exp(logits[i]) for i in top]
    dist = [sum(g / sum(gate) * table[i][v] for g, i in zip(gate, top)) for v in range(len(ref.VOCAB))]
    return dist, frozenset(top)


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def run(ref, table, sigma):
    noise = random.Random(99)
    target, clean = ref.TargetModel(), ref.TargetModel()
    target.distribution = lambda s: moe(ref, table, s, sigma, noise)[0]
    clean.distribution = lambda s: moe(ref, table, s, 0.0, noise)[0]
    draft = ref.DraftModel(alignment=ALIGN)
    aligned = types.SimpleNamespace(propose=lambda c, k, rng, _t: draft.propose(c, k, rng, clean))
    m = ref.speculative_decode(N, K, random.Random(7), target, aligned)
    probe = [(moe(ref, table, s, sigma, noise), moe(ref, table, s, 0.0, noise)) for s in range(1, PROBE + 1)]
    return {"lesson": round(m.acceptance_rate(K), 3), "tok_call": round(m.tokens_per_target_call(), 2),
            "set_flip": round(statistics.mean(a[1] != b[1] for a, b in probe), 3),
            "argmax_kept": round(statistics.mean(argmax(a[0]) == argmax(b[0]) for a, b in probe), 3),
            "greedy": round(statistics.mean(ALIGN * (argmax(a[0]) == argmax(b[0])) + (1 - ALIGN) * b[0][argmax(a[0])]
                                            for a, b in probe), 3)}


def boundary_gap():
    """Mean router-logit gap between the 8th and 9th expert, the margin routing noise has to cross."""
    gaps = []
    for s in range(1, PROBE + 1):
        r = random.Random(s * 7 + 13)
        top = sorted((r.gauss(0, 1) for _ in range(E)), reverse=True)
        gaps.append(top[TOP - 1] - top[TOP])
    return round(statistics.mean(gaps), 3), round(statistics.median(gaps), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = experts(ref)
    return {"rows": {s: run(ref, table, s) for s in SIGMAS}, "gap": boundary_gap()}


def verify(result):
    rows, gap = result["rows"], result["gap"]
    table = {s: (v["set_flip"], v["argmax_kept"], v["greedy"], v["lesson"], v["tok_call"]) for s, v in rows.items()}
    slope = {m: round((rows[0.1][m] - rows[0.0][m]) / 0.1, 2) for m in ("greedy", "lesson")}
    return [
        practice.Check(
            "ANSWER: greedy acceptance falls 0.152 by sigma 0.1, a slope of -1.52 per unit of routing noise",
            table == {0.0: (0, 1, 0.92, 0.924, 4.69), 0.05: (0.378, 0.914, 0.843, 0.907, 4.62),
                      0.1: (0.636, 0.832, 0.768, 0.899, 4.6), 0.2: (0.879, 0.729, 0.675, 0.899, 4.6),
                      0.5: (0.993, 0.509, 0.476, 0.776, 4.1), 1.0: (1, 0.306, 0.291, 0.495, 2.98)}
            and slope["greedy"] == -1.52 and gap == (0.062, 0.044),
            f"sigma -> (set changed, argmax kept, greedy, lesson, tok/call) {table}; top-8 gap mean/median {gap}",
        ),
        practice.Check(
            "FINDING: the lesson's acceptance metric reads the same noise as 6x weaker",
            slope["lesson"] == -0.25 and rows[0.1]["lesson"] == rows[0.2]["lesson"] == 0.899
            and round(slope["greedy"] / slope["lesson"], 1) == 6.1,
            f"slopes over 0 -> 0.1: {slope}; lesson metric at 0.1 and 0.2: "
            f"{rows[0.1]['lesson']}, {rows[0.2]['lesson']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
