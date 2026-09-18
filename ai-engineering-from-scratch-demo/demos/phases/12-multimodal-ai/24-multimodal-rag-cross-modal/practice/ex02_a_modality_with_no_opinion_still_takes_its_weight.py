"""Exercise 2 — a modality with no opinion still takes its weight.

    Score fusion is a simple weighted sum. What failure mode does it have that
    MoE fusion avoids?

Reading of the exercise: the failure is found by running the lesson's own three
retrievers on a query that gives one of them nothing to say, because that is the
case a weighted sum cannot represent -- it has a weight for every modality and no
way to express "this one has no opinion".

**ANSWER: a constant score is indistinguishable from a low one.** Ask the
lesson's corpus for "find me a vegan brunch" and `audio_retrieve` returns
**0.5 for all five** restaurants. A constant added with weight 0.3 shifts every
fused score by 0.15 and changes no ranking -- so the audio retriever consumes
**30%** of the weight budget and contributes nothing. A gate that reads the query
routes that weight elsewhere; a fixed weight cannot.

**FINDING: the same query makes the image retriever return zero for everything.**
With no tag hints, `image_retrieve` scores **0.0 x 5** -- a different constant,
from a different branch, and equally invisible to the sum. Two of the three
modalities are mute and the fusion is still a three-term weighted average.

**FINDING: and the three scores are on three different scales.** On the demo
query, text spans 0.0-0.43, image is 0 or 1, and audio spans 0.10-0.53. The
weights are applied as though the ranges were comparable, so the 0.4 on image is
worth far more than the 0.3 on text -- the effective weighting is the nominal one
times each retriever's spread, and nobody computes the product.

**FINDING: which is why the agentic trigger is meaningless.** `agentic_loop`
compares the top fused score against a floor. Doubling every weight -- which
cannot change any ranking -- takes that score from **0.686** to **1.372** and
turns a reformulation into an accept. The threshold is on an unnormalised sum,
so it measures the weights rather than the retrieval.

Structure: `signals` runs the lesson's three retrievers on one query, `constant`
detects a retriever with no opinion, and `SCALED` is the weight vector that
doubles every term.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "24-multimodal-rag-cross-modal"
RICH = "find me a quiet vegan brunch with natural light"
SPARSE = "find me a vegan brunch"
WEIGHTS = (0.3, 0.4, 0.3)
SCALED = (0.6, 0.8, 0.6)


def signals(ref, query):
    return {"text": ref.text_retrieve(query),
            "image": ref.image_retrieve(query),
            "audio": ref.audio_retrieve(query)}


def constant(scores):
    return len(set(scores.values())) == 1


def spread(scores):
    return round(max(scores.values()) - min(scores.values()), 4)


def ranking(ref, query, weights):
    rows = signals(ref, query)
    fused = ref.fuse(list(rows.values()), list(weights))
    return [doc for doc, _ in ref.top_k(fused, len(fused))], fused


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sparse = signals(ref, SPARSE)
    rich = signals(ref, RICH)
    order, fused = ranking(ref, RICH, WEIGHTS)
    scaled_order, scaled = ranking(ref, RICH, SCALED)
    return {
        "sparse_constant": {name: constant(scores) for name, scores in sparse.items()},
        "mute": [name for name, scores in sparse.items() if constant(scores)],
        "audio_value": sorted(set(sparse["audio"].values())),
        "image_value": sorted(set(sparse["image"].values())),
        "mute_weight": round(sum(w for w, name in zip(WEIGHTS, sparse)
                                 if constant(sparse[name])), 1),
        "spreads": {name: spread(scores) for name, scores in rich.items()},
        "effective": {name: round(weight * spread(rich[name]), 4)
                      for weight, name in zip(WEIGHTS, rich)},
        "nominal_max": max(WEIGHTS),
        "order": order, "scaled_order": scaled_order,
        "same_order": order == scaled_order,
        "top_score": round(max(fused.values()), 4),
        "scaled_top": round(max(scaled.values()), 4),
        "score_ratio": round(max(scaled.values()) / max(fused.values()), 1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a constant score is indistinguishable from a low one",
            all([result["sparse_constant"]["audio"], result["audio_value"] == [0.5],
                 "audio" in result["mute"], result["mute_weight"] == 0.7]),
            f"on {SPARSE!r} the lesson's audio_retrieve returns {result['audio_value'][0]} "
            f"for every restaurant. A constant added with weight 0.3 shifts every fused score "
            f"and changes no ranking, so the retriever consumes its share of the budget and "
            "contributes nothing. A gate that reads the query routes that weight elsewhere; "
            "a fixed weight cannot",
        ),
        practice.Check(
            "FINDING: the same query makes the image retriever return zero for everything",
            all([result["sparse_constant"]["image"], result["image_value"] == [0.0],
                 sorted(result["mute"]) == ["audio", "image"],
                 not result["sparse_constant"]["text"]]),
            f"with no tag hints image_retrieve scores {result['image_value'][0]} for all five "
            f"-- a different constant from a different branch, equally invisible to the sum. "
            f"{sorted(result['mute'])} are both mute and the fusion is still a three-term "
            "weighted average",
        ),
        practice.Check(
            "FINDING: the three scores are on three different scales",
            all([result["spreads"] == {"text": 0.4286, "image": 1.0, "audio": 0.425},
                 result["effective"] == {"text": 0.1286, "image": 0.4, "audio": 0.1275},
                 result["nominal_max"] == 0.4]),
            f"on the rich query the spans are {result['spreads']} and the weights are applied "
            f"as though they were comparable. Effective weighting is the nominal weight times "
            f"the spread -- {result['effective']} -- so the 0.4 on image is worth three times "
            "the 0.3 on text, and nobody computes the product",
        ),
        practice.Check(
            "FINDING: which is why the agentic trigger is meaningless",
            all([result["same_order"], result["top_score"] == 0.6861,
                 result["scaled_top"] == 1.3721, result["score_ratio"] == 2.0]),
            f"doubling every weight cannot change a ranking -- and does not, "
            f"{result['order'][:3]} either way -- but takes the top fused score from "
            f"{result['top_score']} to {result['scaled_top']}, {result['score_ratio']}x. "
            "agentic_loop compares that number against a floor, so its trigger measures the "
            "weights rather than the retrieval",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
