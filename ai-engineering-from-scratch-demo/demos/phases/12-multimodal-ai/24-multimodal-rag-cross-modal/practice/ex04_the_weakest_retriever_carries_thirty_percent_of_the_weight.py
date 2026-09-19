"""Exercise 4 — the weakest retriever carries 30% of the weight.

    Design an eval spec for a trip-planner multimodal RAG. What metrics cover
    image recall, audio recall, and composite correctness?

Reading of the exercise: the spec is written and then run on the lesson's own
corpus and its own demo query, because a metric nobody has computed is a
proposal rather than a spec -- and computing these three immediately shows which
of the lesson's retrievers is carrying the fusion and which is being carried.

**ANSWER: four metrics, and the first three are per-modality.** Recall@k for each
retriever *alone* against a gold set; fused recall@k and precision@k; and
citation coverage -- the fraction of retrieved items whose answer line carries
evidence from every modality that scored it.

**ANSWER: on the lesson's own query the fused system is perfect and one of its
parts is not.** Gold is `{r1, r4}` -- under 45 dB, vegan brunch, natural light.
Fused recall@2 is **1.00** and precision@3 is **0.67**. Standalone recall@2 is
**image 1.00, audio 1.00, text 0.50**.

**FINDING: the weakest retriever carries 30% of the weight.** Text alone finds
one of the two gold pages. `r1` wins outright at 0.4286, and the second slot is a
**three-way** tie at **0.2857** between `r2`, `r3` and `r4` that `sorted` breaks
by position rather than relevance -- so gold `r4` loses it to `r2`. It is
weighted 0.3 against image's 0.4, and no eval in the lesson would have shown it.

**FINDING: citation coverage is the metric that is already satisfied and still
worth having.** `grounded_generate` emits a review id, an image tag list and a
dB reading for every result -- **3 of 3** modalities per item, **100%**. It is
worth measuring precisely because it is the one that silently degrades: drop the
audio retriever from the fusion and the dB line is still printed, so the answer
keeps citing evidence that no longer scored anything.

Structure: `GOLD` states the answer set, `recall_at` and `precision_at` are the
metrics, `standalone` runs each retriever alone, and `coverage` counts the
modalities cited per result.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "24-multimodal-rag-cross-modal"
QUERY = "find me a quiet vegan brunch with natural light"
GOLD = frozenset({"r1", "r4"})
WEIGHTS = (0.3, 0.4, 0.3)
MODALITIES = ("text", "image", "audio")
QUIET_DB = 45


def retrievers(ref):
    return {"text": ref.text_retrieve, "image": ref.image_retrieve,
            "audio": ref.audio_retrieve}


def recall_at(ref, scores, k, gold=GOLD):
    picked = {doc for doc, _ in ref.top_k(scores, k)}
    return round(len(picked & gold) / len(gold), 2)


def precision_at(ref, scores, k, gold=GOLD):
    picked = {doc for doc, _ in ref.top_k(scores, k)}
    return round(len(picked & gold) / k, 2)


def standalone(ref, query=QUERY, k=2):
    return {name: recall_at(ref, fn(query), k) for name, fn in retrievers(ref).items()}


def fused_scores(ref, query=QUERY, weights=WEIGHTS):
    rows = [fn(query) for fn in retrievers(ref).values()]
    return ref.fuse(rows, list(weights))


def coverage(ref, ranked):
    """Modalities cited per answer line, from the lesson's own generator."""
    text = ref.grounded_generate(QUERY, ranked)
    lines = [line for line in text.splitlines() if line.strip().startswith(tuple("123"))]
    return [sum(marker in line for marker in ("[review", "[img tags", "[ambient"))
            for line in lines]


def runner_up_tie(ref, query=QUERY):
    """Who shares the second-best text score -- the tie the ranking breaks blind."""
    scores = ref.text_retrieve(query)
    second = sorted(scores.values(), reverse=True)[1]
    return round(second, 4), sorted(doc for doc, v in scores.items() if v == second)


def gold_check(ref):
    """The gold set re-derived from the corpus, so it is not a magic constant."""
    return {r.id for r in ref.CORPUS
            if r.ambient_db < QUIET_DB and "vegan brunch" in r.review_text
            and "natural_light" in r.image_tags}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fused = fused_scores(ref)
    ranked = ref.top_k(fused, 3)
    alone = standalone(ref)
    return {
        "gold": sorted(GOLD), "gold_rederived": sorted(gold_check(ref)),
        "gold_matches": gold_check(ref) == set(GOLD),
        "fused_recall": recall_at(ref, fused, 2),
        "fused_precision": precision_at(ref, fused, 3),
        "standalone": alone,
        "weakest": min(alone, key=alone.get),
        "weakest_weight": WEIGHTS[MODALITIES.index(min(alone, key=alone.get))],
        "strongest_weight": max(WEIGHTS),
        "tie_score": runner_up_tie(ref)[0], "tied": runner_up_tie(ref)[1],
        "gold_in_tie": sorted(set(runner_up_tie(ref)[1]) & GOLD),
        "text_picks": [doc for doc, _ in ref.top_k(ref.text_retrieve(QUERY), 2)],
        "coverage": coverage(ref, ranked),
        "coverage_pct": round(sum(coverage(ref, ranked))
                              / (3 * len(ranked)) * 100),
        "metrics": 4, "per_modality": 3,
        "ranked": [doc for doc, _ in ranked],
    }


def verify(result):
    alone = result["standalone"]
    return [
        practice.Check(
            "ANSWER: four metrics, three of them per-modality",
            all([result["metrics"] == 4, result["per_modality"] == 3,
                 result["gold_matches"], result["gold"] == ["r1", "r4"]]),
            f"recall@k for each retriever alone, fused recall@k and precision@k, and citation "
            f"coverage. The gold set re-derived from the corpus -- under {QUIET_DB} dB, vegan "
            f"brunch, natural light -- is {result['gold_rederived']}, which matches the "
            "stated one, so the spec is checkable rather than asserted",
        ),
        practice.Check(
            "ANSWER: the fused system is perfect and one of its parts is not",
            all([result["fused_recall"] == 1.0, result["fused_precision"] == 0.67,
                 result["ranked"] == ["r1", "r4", "r3"],
                 alone == {"text": 0.5, "image": 1.0, "audio": 1.0}]),
            f"fused recall@2 is {result['fused_recall']} and precision@3 "
            f"{result['fused_precision']} on {result['ranked']}; standalone recall@2 is "
            f"{alone}. The composite number hides a retriever finding half the gold set",
        ),
        practice.Check(
            "FINDING: the weakest retriever carries 30% of the weight",
            all([result["weakest"] == "text", result["weakest_weight"] == 0.3,
                 result["strongest_weight"] == 0.4, alone["text"] == 0.5,
                 result["tied"] == ["r2", "r3", "r4"], result["tie_score"] == 0.2857,
                 result["gold_in_tie"] == ["r4"], result["text_picks"] == ["r1", "r2"]]),
            f"{result['weakest']} alone finds {alone['text']:.0%} of the gold set. r1 wins "
            f"outright, and the second slot is a {len(result['tied'])}-way tie at "
            f"{result['tie_score']} between {result['tied']} that sorted breaks by position "
            f"rather than relevance -- so gold {result['gold_in_tie'][0]} loses it to "
            f"{result['text_picks'][1]}. It is weighted {result['weakest_weight']} against "
            f"{result['strongest_weight']}, and no eval in the lesson would have shown it",
        ),
        practice.Check(
            "FINDING: citation coverage is satisfied and still worth having",
            all([result["coverage"] == [3, 3, 3], result["coverage_pct"] == 100]),
            f"grounded_generate emits a review id, an image tag list and a dB reading for "
            f"every result -- {result['coverage']} of 3, {result['coverage_pct']}%. It is "
            "worth measuring because it degrades silently: drop the audio retriever and the "
            "dB line is still printed, so the answer keeps citing evidence that no longer "
            "scored anything",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
