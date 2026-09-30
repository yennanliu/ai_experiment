"""Exercise 2 -- binary faithfulness passes 7 of 8 one-fact edits, and every fixture claim is verbatim context.

    Implement a graded faithfulness: 0 (unsupported), 1 (partially supported), 2 (fully supported). Update the metric accordingly.

Reading of the exercise: the grade comes from the lesson's own `MockJudge`
signal, the fraction of a claim's content tokens found in the retrieved
context. 2 (fully supported) is every token present, 1 (partially) is at
least the judge's 0.4 threshold, 0 is below it. The metric becomes the mean
grade divided by 2, so it stays on the 0-1 axis of the binary metric it
replaces. It is run on the three pipelines over the lesson's 4 `QRELS`, and
on 8 claims copied from the hybrid answers with one fact changed each
(three -> five failed parts, `k = 60` -> `k = 10`, ...).

**ANSWER: `graded_faithfulness` below.** A claim with one changed fact
grades 1, not 2. Swapping one of q1's four claims for "... at five failed
parts" leaves binary faithfulness at 1.0 and takes the graded metric to
0.875. Seven of the 8 edits grade 1 and one grades 0.

**FINDING: binary faithfulness passes 7 of the 8 edited claims.** Only
"Authorization is scattered across every service" falls under 0.4. Changing
a number, a scope or a verb leaves most tokens in place, so the lesson's
judge calls the claim supported.

**FINDING: on the lesson's fixture the update changes nothing.** Every one of
the 33 claims the three pipelines produce (7, 13, 13) is a verbatim
substring of the retrieved context, because each "generator" pastes doc
bodies. Binary and graded are both 1.0 on all 12 answers.

**FINDING: grade 2 is a bag-of-words verdict.** "The upload threshold is
configured per stale records." uses only words of q1's context, says
something false, and grades 2.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "68-rag-eval-precision-recall"
# one fact changed in a verbatim hybrid-pipeline claim, keyed by the qrel whose context judges it
EDITED = [
    ("q1", "The abort threshold is configured per bucket at five failed parts."),
    ("q1", "The abort threshold is configured globally at three failed parts."),
    ("q1", "Past that threshold the upload is retried forever."),
    ("q2", "Authorization is centralized in the policy engine."),
    ("q2", "Authorization is scattered across every service."),
    ("q3", "Production search combines lexical and semantic retrieval through reciprocal rank fusion at k = 10."),
    ("q3", "Production search uses only semantic retrieval."),
    ("q4", "Long-running jobs ignore the cancellation signal."),
]
# every content token is in the q1 context, the sentence is false
RECOMBINED = ("q1", "The upload threshold is configured per stale records.")


def grade(ref, claim, context, threshold):
    """2 when every content token of the claim is in the context, 1 when at least
    the lesson's judge threshold is, 0 otherwise -- the mock judge's own overlap."""
    tokens = ref._content_tokens(claim)
    if not tokens:
        return 0
    overlap = len(tokens & ref._content_tokens(context)) / len(tokens)
    return 2 if overlap == 1.0 else 1 if overlap >= threshold else 0


def graded_faithfulness(ref, claims, context_texts, judge):
    """Mean grade over claims, scaled to 0-1 so it sits on the binary metric's axis."""
    if not claims:
        return 0.0
    blob = " ".join(context_texts)
    return sum(grade(ref, c, blob, judge.overlap_threshold) for c in claims) / (2 * len(claims))


def contexts(ref, fn=None):
    by_id = {d.doc_id: d for d in ref.CORPUS}
    fn = fn or ref.hybrid_pipeline
    out = {}
    for q in ref.QRELS:
        ids, answer = fn(q.query, 5)
        out[q.qid] = ([by_id[i].text() for i in ids], answer)
    return out


def fixture_row(ref, fn, judge):
    """Binary and graded faithfulness per qrel for one lesson pipeline, plus claim counts."""
    runs = contexts(ref, fn)
    claims = {qid: ref.extract_claims(a) for qid, (_, a) in runs.items()}
    return {
        "binary": [ref.faithfulness(claims[q], runs[q][0], judge) for q in runs],
        "graded": [graded_faithfulness(ref, claims[q], runs[q][0], judge) for q in runs],
        "claims": sum(map(len, claims.values())),
        "verbatim": sum(c in " ".join(runs[q][0]) for q in runs for c in claims[q]),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    judge, ctx = ref.MockJudge(), contexts(ref)
    blob = {q: " ".join(texts) for q, (texts, _) in ctx.items()}
    mixed = ref.extract_claims(ctx["q1"][1])[1:] + [EDITED[0][1]]  # q1's answer, first claim swapped for an edit
    return {
        "fixture": {n: fixture_row(ref, fn, judge) for n, fn in (
            ("baseline", ref.baseline_pipeline), ("hybrid", ref.hybrid_pipeline),
            ("hybrid+rerank", ref.hybrid_plus_rerank_pipeline))},
        "probe_binary": [judge.supported(c, blob[q]) for q, c in EDITED + [RECOMBINED]],
        "probe_grade": [grade(ref, c, blob[q], judge.overlap_threshold) for q, c in EDITED + [RECOMBINED]],
        "mixed": {"binary": ref.faithfulness(mixed, ctx["q1"][0], judge),
                  "graded": graded_faithfulness(ref, mixed, ctx["q1"][0], judge)},
    }


def verify(result):
    r, f = result, result["fixture"]
    return [
        practice.Check(
            "ANSWER: graded faithfulness is the mean 0/1/2 grade over 2; one edited claim takes q1 from 1.0 to 0.875",
            (r["mixed"], r["probe_grade"][:-1]) == ({"binary": 1.0, "graded": 0.875}, [1, 1, 1, 1, 0, 1, 1, 1]),
            f"q1 answer with one edited claim: {r['mixed']}; grades of the 8 edits {r['probe_grade'][:-1]}",
        ),
        practice.Check(
            "FINDING: binary faithfulness passes 7 of 8 one-fact edits as supported",
            r["probe_binary"][:-1] == [True, True, True, True, False, True, True, True],
            f"MockJudge.supported on the edits: {r['probe_binary'][:-1]}",
        ),
        practice.Check(
            "FINDING: on the lesson's fixture the update changes nothing -- all 33 claims are verbatim context",
            [(v["binary"], v["graded"], v["claims"], v["verbatim"]) for v in f.values()]
            == [([1.0] * 4, [1.0] * 4, n, n) for n in (7, 13, 13)],
            f"binary and graded per query: {[v['graded'] for v in f.values()]}; (claims, verbatim substrings of "
            f"the context) {[(v['claims'], v['verbatim']) for v in f.values()]}",
        ),
        practice.Check(
            "FINDING: a false sentence built from context words grades 2, fully supported",
            (r["probe_binary"][-1], r["probe_grade"][-1]) == (True, 2),
            f"'{RECOMBINED[1]}': binary {r['probe_binary'][-1]}, grade {r['probe_grade'][-1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
