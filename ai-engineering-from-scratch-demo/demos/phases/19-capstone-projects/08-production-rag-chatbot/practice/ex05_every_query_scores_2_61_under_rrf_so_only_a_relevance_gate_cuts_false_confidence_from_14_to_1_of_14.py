"""Exercise 5 -- every query scores 2/61 under RRF, so only a relevance gate cuts false confidence from 14 to 1 of 14.

    Add an "unsure" mode: if top reranked scores are below a threshold, the
    agent says "I do not have confident citations" instead of answering.
    Measure false-confidence reduction.

Reading of the exercise: false confidence is a cited answer to a question
the user's corpus slice cannot answer. The probe is 28 questions over the
lesson's `CORPUS`: 14 answerable (the gold chunk is visible to the asking
role and jurisdiction, and `retrieve` ranks it first), and 14 unanswerable
(off-domain, or answered only by a chunk the filter hides). `main.py` has no
reranker, so two scores are tried as "the top reranked score": the RRF score
`retrieve` returns, which is what `chat_turn` sees, and the best
`dense_score` of the three hits, a stand-in for a cross-encoder built from
the lesson's own scorer. Unsure mode wraps `chat_turn`: below the threshold
it answers "I do not have confident citations" with no citations; otherwise
`chat_turn` runs unchanged. Thresholds are swept, not tuned on one point.

**ANSWER: gating on the relevance score at 0.10 cuts false confidence from
14/14 (100%) to 1/14 (7.1%) and still answers 14/14 answerable questions.**
The one survivor is counsel/HIPAA asking for the restricted-data deletion
deadline: the stopwords plus "deletion", "data" and "restricted" overlap the
public FAQ chunk (0.125). At 0.15 false confidence is 0/14, but 2 of the 14
answerable questions (14.3%) are refused.

**FINDING: the RRF score cannot carry an unsure mode.** The top RRF score is
exactly 2/61 = 0.03279 on all 28 questions, answerable or not. RRF scores
rank, not relevance, and here the same chunk heads both lists every time.
Every threshold therefore refuses all 28 questions or none of them.

**FINDING: the lesson's own "I do not have confident citations" branch is
unreachable.** `chat_turn` refuses only when `retrieve` returns nothing. The
public `general-privacy-faq` chunk is tagged `any`, so it passes the filter
for every role and jurisdiction, including made-up ones. Over 25 (role,
jurisdiction) pairs asking "zzz", 0 get the refusal.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "08-production-rag-chatbot"
UNSURE = "I do not have confident citations."
MSA, DPA, FAQ = "MSA-2024-03-11 s12.4", "DPA-v2.1 s5", "general-privacy-faq Q1"
BAA, SOC = "HIPAA-BAA-2024 s7", "SOC2-policy-v3 AC-2"
AG, CH, CS = ("analyst", "GDPR"), ("counsel", "HIPAA"), ("counsel", "SOC2")
ANSWERABLE = [
    (AG, "how many days to delete EU user profiles after termination", MSA),
    (AG, "when must EU user profiles be deleted", MSA), (AG, "which GDPR article governs profile deletion", MSA),
    (AG, "deletion deadline for the restricted data category", DPA),
    (AG, "how fast is restricted data deleted after a termination notice", DPA),
    (AG, "how can users request a data export", FAQ), (AG, "where do users export their data", FAQ),
    (CH, "what happens to PHI when the agreement terminates", BAA), (CH, "how many days to return or destroy PHI", BAA),
    (CH, "must PHI be destroyed after termination", BAA), (CS, "how often are privileged users access reviewed", SOC),
    (CS, "access review cadence for standard users", SOC), (CS, "is the access review quarterly or annual", SOC),
    (CH, "can users request data export through the portal", FAQ),
]
UNANSWERABLE = [
    (AG, "what is the weather in paris today"), (AG, "who won the world cup"),
    (AG, "what is the penalty for late invoice payment"), (AG, "how many vacation days do employees get"),
    (AG, "what is the obligation for PHI after termination"), (AG, "how often are privileged users reviewed"),
    (CH, "how many days to delete EU user profiles"), (CH, "what is the deletion deadline for restricted data"),
    (CH, "write me a poem about the ocean"), (CH, "what is the capital of france"),
    (CS, "what does GDPR article 17 say"), (CS, "how long is PHI retained"),
    (CS, "what is the refund policy"), (CS, "what is the uptime SLA"),
]
DENSE_T = [0.05, 0.10, 0.15, 0.20]


def scores(ref, who, q):
    hits = ref.retrieve(q, *who, ref.CORPUS, k=3)
    return {"rrf": round(hits[0][1], 5), "dense": max(ref.dense_score(q, c) for c, _ in hits)}


def unsure_turn(ref, who, q, kind, threshold, cache):
    if scores(ref, who, q)[kind] < threshold:
        return {"answer": UNSURE, "citations": []}
    return ref.chat_turn(q, *who, ref.CORPUS, cache)


def rates(ref, kind, threshold):
    cache = ref.PromptCache()
    ans = [unsure_turn(ref, w, q, kind, threshold, cache) for w, q, _ in ANSWERABLE]
    una = [unsure_turn(ref, w, q, kind, threshold, cache) for w, q in UNANSWERABLE]
    return {"false_conf": sum(bool(r["citations"]) for r in una),
            "answered": sum(bool(r["citations"]) and r["citations"][0] == g for r, (_, _, g) in zip(ans, ANSWERABLE))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rrf = {scores(ref, w, q)["rrf"] for w, q, *_ in ANSWERABLE + UNANSWERABLE}
    top = max(rrf)
    pairs = [(r, j) for r in ("analyst", "counsel", "public", "intern", "admin")
             for j in ("GDPR", "HIPAA", "SOC2", "any", "CCPA")]
    cache = ref.PromptCache()
    survivors = [q for w, q in UNANSWERABLE if scores(ref, w, q)["dense"] >= 0.10]
    return {
        "baseline": rates(ref, "dense", 0.0), "rrf_values": sorted(rrf),
        "rrf_sweep": {t: rates(ref, "rrf", t) for t in (0.0, top, top + 1e-5)},
        "dense_sweep": {t: rates(ref, "dense", t) for t in DENSE_T}, "survivors": survivors,
        "refusals": sum(ref.chat_turn("zzz", r, j, ref.CORPUS, cache)["answer"].startswith("I do not") for r, j in pairs),
        "pairs": len(pairs),
    }


def verify(result):
    r = result
    d = r["dense_sweep"]
    table = {t: (v["false_conf"], v["answered"]) for t, v in d.items()}
    return [
        practice.Check(
            "ANSWER: a 0.10 relevance gate cuts false confidence 14/14 -> 1/14 and keeps 14/14 answerable",
            r["baseline"] == {"false_conf": 14, "answered": 14}
            and table == {0.05: (12, 14), 0.10: (1, 14), 0.15: (0, 12), 0.20: (0, 11)}
            and r["survivors"] == ["what is the deletion deadline for restricted data"],
            f"baseline {r['baseline']}; (false-confident /14, answered correctly /14) by threshold {table}; "
            f"survivor at 0.10: {r['survivors']}",
        ),
        practice.Check(
            "FINDING: the top RRF score is 2/61 on all 28 questions, so an RRF threshold is all-or-nothing",
            r["rrf_values"] == [round(2 / 61, 5)]
            and [(v["false_conf"], v["answered"]) for v in r["rrf_sweep"].values()] == [(14, 14), (14, 14), (0, 0)],
            f"distinct top RRF scores {r['rrf_values']}; sweep {r['rrf_sweep']}",
        ),
        practice.Check(
            "FINDING: the lesson's 'I do not have confident citations' branch is unreachable",
            (r["refusals"], r["pairs"]) == (0, 25),
            f"{r['refusals']}/{r['pairs']} (role, jurisdiction) pairs asking 'zzz' get the refusal",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
