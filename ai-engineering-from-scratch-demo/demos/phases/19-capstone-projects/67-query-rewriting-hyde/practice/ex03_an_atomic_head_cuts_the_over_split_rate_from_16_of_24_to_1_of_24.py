"""Exercise 3 — an atomic head cuts the over-split rate from 16 of 24 to 1 of 24.

    Train the decomposer to recognize atomic queries by adding a "is the question atomic" head. Measure the over-split rate before and after.

Reading of the exercise: the decomposer is the lesson's `DecomposeRewriter`.
Off its lookup table it splits on every " and ". The head is a
logistic regression (numpy, 400 steps of gradient descent from zero) over
four features of the " and " split. Does the right half start with a
question word? Does it carry its own verb? Do the two halves retrieve
different top-1 documents from the lesson's `HybridRetriever`? How long is
the shorter half? Questions with no " and " are atomic by construction. The
labelled fixture is 36 questions about the lesson's corpus: 24 atomic (16 of
them contain "and", as in "lexical and semantic retrieval") and 12 multi-topic,
including the lesson's two `DECOMP_TABLE` queries. Each atomic question has
one gold document. The head is scored by 2-fold cross-validation on
alternating rows. The over-split rate is the share of atomic questions that
come back as more than one sub-question.

**ANSWER: the over-split rate is 66.7% (16 of 24) before and 4.2% (1 of
24) after.** Before, every atomic question containing "and" is split. With
the head, under 2-fold cross-validation, 1 of 24 atomic questions is still
split and 12 of 12 multi-topic ones still are. The one miss is "how much
memory do vectors and indexes need": its right half has a verb of its own,
and its two halves retrieve different top-1 documents.

**FINDING: the lesson's warning about over-splitting holds, and the miss is
one of the costly cases.** Unsplit, all 16 atomic "and" questions put their
gold document first (mean rank 1.00). Split, the mean rank is 1.44: 3
questions lose first place, and two of them, including the head's miss,
fall to rank 4. No question gains.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "67-query-rewriting-hyde"
WH = {"how", "what", "where", "when", "which", "why", "who"}
VERBS = {"is", "are", "does", "do", "happens", "get", "work", "need", "cover", "enter"}
ATOMIC = [("how is lexical and semantic retrieval combined", "d6"), ("what does rank fusion do with BM25 and dense rankings", "d6"),
          ("how are principal resource and action checked", "d4"), ("which function handles user and service accounts", "d4"),
          ("how much memory do vectors and indexes need", "d7"), ("how does the transfer manager split and track parts", "d2"),
          ("what stops the worker and releases the queue slot", "d8"), ("how are aborted transfers and reserved keys handled", "d2"),
          ("what happens to the bucket retry quota and cooldown", "d3"), ("how do the OPA runtime and evaluate relate", "d5"),
          ("how does search merge ranks and not scores", "d6"), ("what memory do 256 dimension vectors and their index use", "d7"),
          ("how does check_permission weigh principal and resource", "d4"), ("how do cancellation signals stop jobs and workers", "d8"),
          ("how is a multipart transfer aborted and the quota decremented", "d1"), ("what is cached by the policy engine and for how long", "d5"),
          ("how is an in-flight multipart upload aborted", "d1"), ("where is the central permission check", "d4"),
          ("how big is the vector index", "d7"), ("how do long-running jobs get cancelled", "d8"),
          ("what does the policy engine wrap", "d5"), ("how does reciprocal rank fusion work", "d6"),
          ("what happens when a bucket quota hits zero", "d3"), ("how are file parts tracked during a transfer", "d2")]
MULTI = ["how is a multipart upload aborted and how does the quota cooldown work",
         "how is authorization checked and how is the policy engine cached",
         "how big is the vector index and how does rank fusion work",
         "how are jobs cancelled and how are transfers split into parts",
         "what does check_permission evaluate and what happens when the retry quota reaches zero",
         "how does search fuse rankings and how much memory does the index need",
         "how are aborted transfers released and how are long running jobs stopped",
         "what runtime does the policy engine wrap and which accounts does check_permission cover",
         "what happens when an upload fails and the retry budget is exhausted",
         "how is authorization handled and how do policies get evaluated",
         "how do uploads fail and what is the vector size", "where are parts tracked and when does a bucket enter cooldown"]


def features(ref, retriever, query):
    left, right = query.split(" and ", 1)
    lt, rt = ref.tokenize(left), ref.tokenize(right)
    top = [retriever.search(half, k_each=5, k_out=1)[0][0].doc_id for half in (left, right)]
    return [1.0, float(rt[0] in WH), float(bool(VERBS & set(rt))), float(top[0] != top[1]), min(len(lt), len(rt)) / 10]


def train_head(x, y, steps=400, lr=0.5):
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-x @ w))
        w -= lr * x.T @ (p - y) / len(y)
    return w


def cross_validated_splits(ref, retriever, queries, labels):
    """1 = the head says multi-topic (split), per question, 2-fold on alternating rows."""
    has_and = [i for i, q in enumerate(queries) if " and " in q]
    x = np.array([features(ref, retriever, queries[i]) for i in has_and])
    y = np.array([labels[i] for i in has_and], dtype=float)
    split = [0] * len(queries)
    for fold in (0, 1):
        test = np.arange(len(has_and)) % 2 == fold
        w = train_head(x[~test], y[~test])
        for j in np.flatnonzero(test):
            split[has_and[j]] = int(x[j] @ w > 0)
    return split


def gold_rank(ref, retriever, rewriter, query, gold):
    ids = [d.doc_id for d, _ in ref.retrieve_with_rewriter(query, rewriter, retriever, 8, 8)["results"]]
    return ids.index(gold) + 1


def split_cost(ref, retriever, llm, missed):
    """Gold rank of each atomic "and" question, unsplit and split, plus the head's misses."""
    and_atomic = [(q, g) for q, g in ATOMIC if " and " in q]
    whole = [gold_rank(ref, retriever, ref._IdentityRewriter(), q, g) for q, g in and_atomic]
    split = [gold_rank(ref, retriever, ref.DecomposeRewriter(llm=llm), q, g) for q, g in and_atomic]
    return whole, split, [s for s, (q, _) in zip(split, and_atomic) if q in missed]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    retriever, llm = ref.build_retriever(), ref.MockLLM()
    queries = [q for q, _ in ATOMIC] + MULTI
    labels = [0] * len(ATOMIC) + [1] * len(MULTI)
    before = [int(len(llm.decompose(q)) > 1) for q in queries]
    after = cross_validated_splits(ref, retriever, queries, labels)
    n = len(ATOMIC)
    whole, split, miss_ranks = split_cost(ref, retriever, llm, set(q for q, x in zip(queries[:n], after[:n]) if x))
    return {"over_before": sum(before[:n]), "over_after": sum(after[:n]), "atomic": n,
            "multi_split_before": sum(before[n:]), "multi_split_after": sum(after[n:]), "multi": len(MULTI),
            "rank_whole": float(np.mean(whole)), "rank_split": float(np.mean(split)),
            "split_ranks": sorted(split), "miss_ranks": miss_ranks, "worse": sum(s > w for s, w in zip(split, whole)), "better": sum(s < w for s, w in zip(split, whole))}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the over-split rate falls from 66.7% to 4.2% and every multi-topic question is still split",
            (r["over_before"], r["over_after"], r["atomic"], r["multi_split_before"], r["multi_split_after"], r["multi"])
            == (16, 1, 24, 12, 12, 12),
            f"over-split before {r['over_before']}/{r['atomic']} ({r['over_before'] / r['atomic']:.1%}), after "
            f"{r['over_after']}/{r['atomic']}; multi-topic split before {r['multi_split_before']}/{r['multi']}, after {r['multi_split_after']}/{r['multi']}",
        ),
        practice.Check(
            "FINDING: over-splitting takes 3 of 16 atomic questions off gold@1, as the lesson warns",
            (round(r["rank_whole"], 2), round(r["rank_split"], 2), r["worse"], r["better"]) == (1.0, 1.44, 3, 0)
            and r["split_ranks"][-3:] == [2, 4, 4] and r["miss_ranks"] == [4],
            f"mean gold rank unsplit {r['rank_whole']:.2f}, split {r['rank_split']:.2f}; split worse on {r['worse']}, better on {r['better']} of 16; worst split ranks {r['split_ranks'][-3:]}, head's miss at {r['miss_ranks']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
