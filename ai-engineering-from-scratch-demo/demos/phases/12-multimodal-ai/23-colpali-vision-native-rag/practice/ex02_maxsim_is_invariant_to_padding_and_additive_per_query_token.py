"""Exercise 2 — MaxSim is invariant to padding and additive per query token.

    MaxSim is Σ_i max_j cos(q_i, p_j). What does this sum capture that a simple
    mean similarity does not?

Reading of the exercise: the property is named and then isolated by the
experiment that separates the two scores -- padding a page with irrelevant
patches, which a mean must notice and a max must not. Two further properties of
the same formula follow from running it, and both are reasons a raw MaxSim score
is not comparable across queries or pages.

**ANSWER: it captures the best evidence per query token and ignores the rest of
the page.** On the lesson's own fixture MaxSim is **1.478** and the mean is
**0.501**; pad the page from 4 patches to 729 with irrelevant ones and MaxSim is
**still 1.478** while the mean collapses to **0.052**. A mean asks "is this page
about the query"; a max asks "is the answer on this page".

**FINDING: which is exactly why it cannot be compared across pages of different
sizes.** The invariance is the feature, and its cost is that a 729-patch page and
a 16-patch page are scored on the same scale with no normalisation for how many
chances each had to match.

**FINDING: the score is additive per query token, not linear in query length.**
Each token contributes its own max independently -- token 0 scores **0.774**,
token 1 **0.7036** -- and every total is exactly their running sum: 0.774 at one
token, **1.478** at two, **2.252** at three, **2.955** at four, **14.776** at
twenty. So the per-token average tracks *which* tokens the query has, not how
many: **0.774**, **0.7388**, **0.7505**, **0.7388** -- three distinct values. The
apparent linearity of 2, 4 and 20 is an artifact of repeating the same two-token
block. A fixed relevance threshold is wrong for every query in both length and
composition.

**FINDING: one patch can serve every query token.** A single patch matching both
tokens scores **1.481**, against **2.000** for two specialised patches -- so the
sum credits the same evidence twice without noticing. That is the same
double-counting Lesson 12.17 finds in its grounding metric and Lesson 12.19 in
its diarization scorer: a max with no assignment step.

Structure: `mean_sim` is the baseline, `padded` grows the page with irrelevant
patches, and `QUERY` and `PATCHES` are the lesson's own `compare_maxsim_vs_mean`
fixture.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "23-colpali-vision-native-rag"
QUERY = ([1.0, 0.1, 0.0], [0.0, 1.0, 0.1])
STRONG = [0.9, 0.9, 0.0]
OTHERS = ([0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.0, 0.0, 0.0])
FILLER = [0.0, 0.0, 0.001]
PADS = (0, 4, 16, 64, 725)
QUERY_LENGTHS = (1, 2, 3, 4, 20)
SPECIALISED = ([1.0, 0.1, 0.0], [0.0, 1.0, 0.1])
SHARED = [0.7, 0.7, 0.05]


def page(pad=0):
    return [STRONG] + list(OTHERS) + [list(FILLER)] * pad


def mean_sim(ref, query, patches):
    return sum(ref.cosine(q, p) for q in query for p in patches) / (len(query) * len(patches))


def stretched(length):
    return (list(QUERY) * 10)[:length]


def sweep(ref, query):
    """MaxSim and mean over the padding levels."""
    return {pad: (round(ref.maxsim(query, page(pad)), 4),
                  round(mean_sim(ref, query, page(pad)), 4)) for pad in PADS}


def by_length(ref):
    return {n: round(ref.maxsim(stretched(n), page()), 4) for n in QUERY_LENGTHS}


def token_scores(ref):
    return [round(ref.maxsim([list(token)], page()), 4) for token in QUERY]


def additivity(lengths, tokens):
    """Per-token averages, and whether each total is the running sum of tokens."""
    per_token = {n: round(v / n, 4) for n, v in lengths.items()}
    running = {n: sum(tokens[i % len(QUERY)] for i in range(n)) for n in QUERY_LENGTHS}
    return {
        "per_token": per_token,
        "distinct_per_token": len(set(per_token.values())),
        "additive": all(abs(lengths[n] - running[n]) < 1e-3 for n in QUERY_LENGTHS),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    query = [list(token) for token in QUERY]
    padded, lengths, tokens = sweep(ref, query), by_length(ref), token_scores(ref)
    shared = round(ref.maxsim(query, [list(SHARED)] + list(OTHERS)), 4)
    separate = round(ref.maxsim(query, [list(v) for v in SPECIALISED] + list(OTHERS)), 4)
    return {
        "padded": padded,
        "maxsim_flat": len({row[0] for row in padded.values()}) == 1,
        "mean_collapse": round(padded[0][1] / padded[725][1], 1),
        "patches": {pad: len(page(pad)) for pad in PADS},
        "lengths": lengths, "token_scores": tokens,
        **additivity(lengths, tokens),
        "shared": shared, "separate": separate,
        "double_counted": round(separate - shared, 3),
    }


def verify(result):
    padded, lengths = result["padded"], result["lengths"]
    return [
        practice.Check(
            "ANSWER: it captures the best evidence per query token and ignores the rest",
            all([padded[0] == (1.4776, 0.5007), padded[725] == (1.4776, 0.0522),
                 result["maxsim_flat"], result["patches"][725] == 729]),
            f"on the lesson's own fixture MaxSim is {padded[0][0]} and the mean "
            f"{padded[0][1]}; padding the page from {result['patches'][0]} patches to "
            f"{result['patches'][725]} leaves MaxSim at {padded[725][0]} and takes the mean "
            f"to {padded[725][1]}. A mean asks whether the page is about the query; a max "
            "asks whether the answer is on it",
        ),
        practice.Check(
            "FINDING: which is why it cannot be compared across pages of different sizes",
            all([result["maxsim_flat"], result["mean_collapse"] == 9.6,
                 len(result["patches"]) == len(PADS)]),
            f"the invariance is the feature, and the mean falls {result['mean_collapse']}x "
            f"over the same sweep. Its cost is that a {result['patches'][725]}-patch page and "
            f"a {result['patches'][0]}-patch page are scored on one scale with no "
            "normalisation for how many chances each had to match",
        ),
        practice.Check(
            "FINDING: the score is additive per query token, not linear in query length",
            all([lengths == {1: 0.774, 2: 1.4776, 3: 2.2515, 4: 2.9551, 20: 14.7755},
                 result["token_scores"] == [0.774, 0.7036], result["additive"],
                 result["distinct_per_token"] == 3]),
            f"each query token contributes its own max independently -- "
            f"{result['token_scores']} here -- and every total is exactly their running sum: "
            f"{list(lengths.values())} for lengths {list(QUERY_LENGTHS)}. So the per-token "
            f"average tracks *which* tokens the query has, not how many: "
            f"{result['per_token']}, {result['distinct_per_token']} distinct values. The "
            "apparent linearity of 2, 4 and 20 is an artifact of repeating the same "
            "two-token block; a fixed relevance threshold is wrong for every query in both "
            "length and composition",
        ),
        practice.Check(
            "FINDING: one patch can serve every query token",
            all([result["shared"] == 1.4807, result["separate"] == 2.0,
                 result["double_counted"] == 0.519]),
            f"a single patch matching both query tokens scores {result['shared']} against "
            f"{result['separate']} for two specialised patches -- the sum credits the same "
            "evidence twice without noticing. The same double-counting Lesson 12.17 finds in "
            "its grounding metric and Lesson 12.19 in its diarization scorer: a max with no "
            "assignment step",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
