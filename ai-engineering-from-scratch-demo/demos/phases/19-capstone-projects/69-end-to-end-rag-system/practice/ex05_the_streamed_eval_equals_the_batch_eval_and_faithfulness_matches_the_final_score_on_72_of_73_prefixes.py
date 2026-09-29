"""Exercise 5 — the streamed eval matches the batch eval exactly, and faithfulness cannot tell a one-word prefix from the answer.

    Add a streaming generator interface and feed it into the eval. Confirm that faithfulness is computed on the final string and not on the streamed prefix.

Reading of the exercise: `stream_answer(query, ranked)` is a Python
generator over the lesson's mock generator. It yields the answer one word
at a time, with trailing whitespace, and the citation anchors as they fall.
`stream_eval` is the lesson's `run_eval` with the stream in place of the
string. For each eval query it runs the pipeline's retrieve and rerank,
drains the stream while keeping the prefix seen so far, and scores the
lesson's five metrics on the joined final string. To confirm the
difference, every intermediate prefix is also scored, and so is an eval that
stops at the first streamed piece, which is what scoring on the prefix would
give.

**ANSWER: the streamed eval scores the final string and matches the batch
eval exactly.** All five metrics are equal to `run_eval`'s: recall@5 1.000,
precision@1 0.500, mrr 0.708, faithfulness 1.000, answer_relevance 1.000.
On every query the joined stream equals the batch answer. The 4 answers
stream as 73 pieces.

**FINDING: on this metric, faithfulness is almost the same on a prefix as
on the final string, so it cannot catch a prefix bug.** Of the 73 streamed
prefixes, 72 already have the final faithfulness of 1.0. The one exception
is the first piece "The ", which has no content token. An eval that scores
only the first piece gets faithfulness 0.750, still passing the 0.75
threshold. What catches it is answer_relevance: 22 of the 73 prefixes differ
from the final value, and the first-piece eval scores 0.250 against its
0.75 threshold. Only "Long-running " already shares 30% of its question's
words.
"""

from __future__ import annotations

import re
from collections import defaultdict

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "69-end-to-end-rag-system"


def stream_answer(ref, query, ranked):
    """The mock generator as a stream of word pieces."""
    answer, _ = ref.generate_answer(query, ranked)
    yield from re.findall(r"\S+\s*", answer)


def score(ref, eq, top_k, answer):
    return {
        "recall@5": ref.doc_level_recall(top_k, eq.gold_doc_ids, 5),
        "precision@1": ref.doc_level_precision(top_k, eq.gold_doc_ids, 1),
        "mrr": ref.doc_level_mrr(top_k, eq.gold_doc_ids),
        "faithfulness": ref.faithfulness_score(answer, top_k),
        "answer_relevance": ref.answer_relevance_score(eq.query, answer),
    }


def stream_eval(ref, p, stop_after=None):
    sums, prefixes, finals = defaultdict(float), [], []
    for eq in ref.EVAL_QUERIES:
        top_k = p.query(eq.query).top_k
        seen = ""
        for i, piece in enumerate(stream_answer(ref, eq.query, top_k)):
            seen += piece
            prefixes.append((eq, top_k, seen))
            if stop_after is not None and i + 1 >= stop_after:
                break
        finals.append(seen)
        for k, v in score(ref, eq, top_k, seen).items():
            sums[k] += v
    n = len(ref.EVAL_QUERIES)
    return {k: round(v / n, 3) for k, v in sums.items()}, prefixes, finals


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    p = ref.build_pipeline()
    batch = {k: round(v, 3) for k, v in ref.run_eval(p).items()}
    streamed, prefixes, finals = stream_eval(ref, p)
    first, _, _ = stream_eval(ref, p, stop_after=1)
    final_of = dict(zip([e.qid for e in ref.EVAL_QUERIES], finals))
    differ = defaultdict(int)
    for eq, top_k, seen in prefixes:
        now, end = score(ref, eq, top_k, seen), score(ref, eq, top_k, final_of[eq.qid])
        for k in ("faithfulness", "answer_relevance"):
            differ[k] += now[k] != end[k]
    return {
        "batch": batch, "streamed": streamed, "first_piece": first, "pieces": len(prefixes),
        "joined_equal": sum(f == p.query(e.query).answer for f, e in zip(finals, ref.EVAL_QUERIES)),
        "faith_differ": differ["faithfulness"], "rel_differ": differ["answer_relevance"],
        "threshold": ref.THRESHOLDS["faithfulness"],
    }


def verify(result):
    r = result
    f = r["first_piece"]
    return [
        practice.Check(
            "ANSWER: the streamed eval scores the joined final string and equals run_eval on all 5 metrics",
            r["streamed"] == r["batch"] and r["joined_equal"] == 4 and r["pieces"] == 73
            and r["batch"] == {"recall@5": 1.0, "precision@1": 0.5, "mrr": 0.708, "faithfulness": 1.0, "answer_relevance": 1.0},
            f"streamed {r['streamed']} vs batch {r['batch']}; joined == batch answer on "
            f"{r['joined_equal']}/4; {r['pieces']} pieces",
        ),
        practice.Check(
            "FINDING: faithfulness matches the final value on 72 of 73 prefixes; a first-piece eval still passes it",
            (r["faith_differ"], r["rel_differ"], f["faithfulness"], f["answer_relevance"]) == (1, 22, 0.75, 0.25)
            and f["faithfulness"] >= r["threshold"],
            f"prefixes differing from the final: faithfulness {r['faith_differ']}/{r['pieces']}, answer_relevance "
            f"{r['rel_differ']}/{r['pieces']}; first-piece eval faithfulness {f['faithfulness']} "
            f"(threshold {r['threshold']}), answer_relevance {f['answer_relevance']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
