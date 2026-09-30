"""Exercise 5 -- faithfulness is constant on the fixture so r is undefined, and a verbosity sweep gives -0.872.

    Add an "answer length" metric and correlate it with faithfulness. Plot the curve.

Reading of the exercise: answer length is the answer's word count. It is
correlated (Pearson) with the lesson's `faithfulness` first on the fixture as
shipped: the 12 answers `evaluate_pipeline` grades, from three pipelines on
4 `QRELS`. Those answers are all fully faithful, so a curve needs answers of
varying length. It comes from a verbosity dial on the hybrid pipeline: the
answer is its first m sentences, read from the retrieved docs first and then
running on into the unretrieved ones, while the judged context stays the
retrieved top 5; m = 1 to 18 on each query gives 72 answers. The plot is a
text chart of mean faithfulness per m, printed when the file is run.

**ANSWER: on the fixture the correlation is undefined; on the sweep it is
r = -0.872.** The 12 fixture answers run 15 to 37 words and every one scores
faithfulness 1.0, so `statistics.correlation` raises "at least one of the
inputs is constant". On the sweep, faithfulness stays at 1.0 while the
answer is inside the retrieved context (up to 32 words), then falls:

    14 words |########################################| 1.00
    32 words |########################################| 1.00
    44 words |################################        | 0.81
    61 words |#######################                 | 0.58
    95 words |#################                       | 0.42
   141 words |############                            | 0.31
   197 words |########                                | 0.21

**FINDING: the curve is a hyperbola, not a line.** Once the retrieved
sentences run out (5 for q1, 3 for q2-q4), the number of supported claims
stays fixed, so faithfulness is that count over the claim count. Pearson's
-0.872 measures how straight that is, not how strong it is.

**FINDING: the mock judge passes a sentence q1 never retrieved.** d2's
"Cancelled uploads release the reserved keys after three failed parts." has 4
of its 9 content words (after, three, failed, parts) in q1's context, which
clears the 0.4 threshold, so q1 ends at 6 supported claims, not 5.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "68-rag-eval-precision-recall"


def answer_length(answer):
    """Words in the answer: whitespace-split, the unit a reader pays for."""
    return len(answer.split())


def pearson(xs, ys):
    try:
        return round(statistics.correlation(xs, ys), 3)
    except statistics.StatisticsError as exc:
        return f"undefined ({exc})"


def fixture_points(ref):
    """(length, faithfulness) for the 12 answers the lesson's evaluator grades."""
    by_id, judge, pts = {d.doc_id: d for d in ref.CORPUS}, ref.MockJudge(), []
    for fn in (ref.baseline_pipeline, ref.hybrid_pipeline, ref.hybrid_plus_rerank_pipeline):
        for q in ref.QRELS:
            ids, answer = fn(q.query, 5)
            ctx = [by_id[i].text() for i in ids]
            pts.append((answer_length(answer), ref.faithfulness(ref.extract_claims(answer), ctx, judge)))
    return pts


def sweep_points(ref):
    """A verbosity dial: the answer is the first m sentences of the hybrid ranking,
    then of the docs it did not retrieve; the context stays the retrieved top 5.
    Returns (qid, m, words, faithfulness) points and the retrieved sentence count per query."""
    judge, pts, inside = ref.MockJudge(), [], {}
    for q in ref.QRELS:
        ids, _ = ref.hybrid_pipeline(q.query, 5)
        retrieved = [d for i in ids for d in ref.CORPUS if d.doc_id == i]
        rest = [d for d in ref.CORPUS if d.doc_id not in ids]
        sents = sentences(ref, retrieved) + sentences(ref, rest)
        inside[q.qid] = len(sentences(ref, retrieved))
        for m in range(1, len(sents) + 1):
            answer, ctx = " ".join(sents[:m]), [d.text() for d in retrieved]
            pts.append((q.qid, m, answer_length(answer), ref.faithfulness(ref.extract_claims(answer), ctx, judge)))
    return pts, inside


def sentences(ref, docs):
    return [s for d in docs for s in ref.extract_claims(d.body)]


def supported_after(pts, inside):
    """Supported-claim counts (faithfulness x claims) once the retrieved sentences have run out."""
    return {q: sorted({round(f * m) for qid, m, _, f in pts if qid == q and m >= n}) for q, n in inside.items()}


def plot(pts):
    """Mean faithfulness per answer length (sentences), one text row each."""
    rows = []
    for m in sorted({p[1] for p in pts}):
        at = [p for p in pts if p[1] == m]
        words, faith = statistics.mean(p[2] for p in at), statistics.mean(p[3] for p in at)
        rows.append((m, round(words), round(faith, 3), f"{round(words):>4} words |{'#' * round(40 * faith):<40}| {faith:.2f}"))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fix, (sweep, inside) = fixture_points(ref), sweep_points(ref)
    xs, ys = zip(*fix)
    return {
        "fixture_lengths": sorted(set(xs)), "fixture_faith": sorted(set(ys)), "fixture_r": pearson(xs, ys),
        "sweep_n": len(sweep), "sweep_r": pearson([p[2] for p in sweep], [p[3] for p in sweep]),
        "curve": [row[:3] for row in plot(sweep)], "chart": [row[3] for row in plot(sweep)],
        "retrieved_sents": inside, "supported": supported_after(sweep, inside),
    }


def verify(result):
    r, curve = result, result["curve"]
    faith = [f for _, _, f in curve]
    return [
        practice.Check(
            "ANSWER: on the fixture the correlation is undefined; on a verbosity sweep r = -0.872",
            (r["fixture_faith"], r["fixture_r"][:9], r["fixture_lengths"], r["sweep_n"], r["sweep_r"])
            == ([1.0], "undefined", [15, 17, 18, 20, 32, 34, 37], 72, -0.872)
            and (faith[:3], faith[-2:], faith == sorted(faith, reverse=True)) == ([1.0] * 3, [0.221, 0.208], True),
            f"fixture: lengths {r['fixture_lengths']} words, faithfulness {r['fixture_faith']}, r {r['fixture_r']}; "
            f"sweep over {r['sweep_n']} answers r = {r['sweep_r']}; curve (sentences, words, faithfulness) {curve}",
        ),
        practice.Check(
            "FINDING: past the retrieved context the curve is supported/length, a hyperbola, not a line",
            (r["retrieved_sents"], r["supported"])
            == ({"q1": 5, "q2": 3, "q3": 3, "q4": 3}, {"q1": [5, 6], "q2": [3], "q3": [3], "q4": [3]}),
            f"retrieved sentences {r['retrieved_sents']}; faithfulness x sentences once they run out {r['supported']}",
        ),
        practice.Check(
            "FINDING: the mock judge passes one sentence from a doc q1 never retrieved",
            r["supported"]["q1"][-1] - r["retrieved_sents"]["q1"] == 1,
            f"q1 supported count rises from {r['retrieved_sents']['q1']} to {r['supported']['q1'][-1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    print("\n".join(solve()["chart"]))  # the curve: mean faithfulness per answer length on the sweep
    raise SystemExit(practice.selfcheck(globals()))
