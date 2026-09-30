"""Exercise 5 — keeping toxic traces makes the sample 6.9x as toxic, and a 28-word rant escapes the density eval.

    Build a tail-sampling policy that keeps 100% of traces with toxicity > 0.5 plus a 10% stratified sample of the rest. Measure sampling bias introduced.

Reading of the exercise: 10,000 traces come from the lesson's
`synth_trace` and are scored by its `enrich_with_evals`. Because the
lesson's traffic is never toxic, responses are swapped for hand-labelled
toxic ones at 2% for most users and 20% for `u_03`, the abusive user. The
toxic bank includes one long rant that carries a single slur. The policy
keeps every trace whose eval toxicity is > 0.5. The rest are stratified by
(model, user), 12 strata, and every 10th trace of each stratum is kept, a
counter the collector can run online. Bias is each statistic on the kept
set minus the same statistic on all 10,000 traces, read raw and
Horvitz-Thompson weighted (weight 1 for a forced keep, 10 otherwise). The
lesson's `TailSampler(0.10)` runs beside it on the same traces.

**ANSWER: the policy keeps 1,454 traces (14.5%), and the raw kept set is
6.9x as toxic as the traffic.** The eval-toxic share is 34.3% kept against
5.0% true (+29.3 points). `u_03` is 41.5% of kept traces against 25.4% true
(+16.1). The worst (model, user) share is off by 6.1 points. Inside the 10%
sample of the rest, stratification holds every stratum's share to within
0.05 points. The lesson's random `TailSampler(0.10)` is off by up to 2.8
there. Mean cost moves by +$0.0001. Weighted by 1 / keep-probability, the
toxic, `u_03` and stratum biases all round to 0.0 points, because the policy
knows each trace's keep probability exactly.

| statistic | true | kept, raw | kept, weighted |
|---|---:|---:|---:|
| eval-toxic share | 5.0% | 34.3% | 5.0% |
| `u_03` share | 25.4% | 41.5% | 25.4% |
| worst stratum-share error | 0 | 6.1 pts | 0.0 pts |

**FINDING: the lesson's toxicity eval is a word density, so a long toxic
answer escapes the keep rule.** `eval_toxicity` is 10 x (bad words /
words), capped at 1. The 28-word rant scores 0.36 and falls into the 10%
sample. So 166 of 665 hand-labelled toxic traces (25%) are kept at 10%
rather than 100%, and only 14 of those 166 survive. On the lesson's own
traffic, 0 of 10,000 traces score over 0.5, so its keep rule never fires.

**FINDING: a toxicity-keyed tail sampler needs the eval on 100% of
traffic.** `main()` runs `enrich_with_evals` on every trace before
`decide`. The skill's hard reject forbids exactly
this: "Evals that run at ingest rate without sampling". `main()` also
samples successes at 0.20, where the doc's step 1 says 10%.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "11-llm-observability-dashboard"
N, SEED, RATE = 10_000, 23, 0.10
TOXIC = ["you are stupid", "this is garbage and so are you", "i hate this stupid question",
         ("honestly that is a question i would expect from someone who has not read a single page of the manual "
          "so here is my answer you stupid person")]


def traffic(ref):
    rng, traces = random.Random(SEED), []
    for i in range(N):
        trace = ref.synth_trace(f"t{i:05d}", False, rng)
        toxic = rng.random() < (0.20 if trace[1].attributes["user_id"] == "u_03" else 0.02)
        if toxic:
            trace[1].attributes["response"] = rng.choice(TOXIC)
        traces.append((ref.enrich_with_evals(trace), toxic))
    return traces


def tox(trace):
    return next(s.attributes["toxicity"] for s in trace if s.name == "eval")


def stratum(trace):
    return trace[1].attributes["gen_ai.request.model"], trace[1].attributes["user_id"]


def policy(traces):
    """[(trace, weight)] kept: every toxic trace at weight 1, every 10th per (model, user) stratum at weight 10."""
    seen, kept = {}, []
    for trace, _ in traces:
        if tox(trace) > 0.5:
            kept.append((trace, 1.0))
            continue
        k = seen[stratum(trace)] = seen.get(stratum(trace), -1) + 1
        if k % round(1 / RATE) == 0:
            kept.append((trace, 1 / RATE))
    return kept


def stats(rows):
    """Weighted toxic share, u_03 share, mean cost and per-stratum shares over [(trace, weight)]."""
    total = sum(w for _, w in rows)
    share = lambda f: sum(w for t, w in rows if f(t)) / total
    return {"toxic": share(lambda t: tox(t) > 0.5), "u_03": share(lambda t: t[1].attributes["user_id"] == "u_03"),
            "cost": sum(w * t[1].attributes["cost_usd"] for t, w in rows) / total,
            "strata": {k: share(lambda t, k=k: stratum(t) == k) for k in {stratum(t) for t, _ in rows}}}


def bias(kept, truth):
    s, d = stats(kept), lambda k, x=100, n=1: round(x * (s[k] - truth[k]), n) + 0.0  # + 0.0 turns -0.0 into 0.0
    return {"toxic": d("toxic"), "u_03": d("u_03"), "cost": d("cost", 1, 4),
            "strata": round(100 * max(abs(s["strata"][k] - v) for k, v in truth["strata"].items()), 1) + 0.0}


def escapes(ref, traces, kept):
    labelled = [t for t, toxic in traces if toxic]
    escaped = [t for t in labelled if tox(t) <= 0.5]
    plain = [ref.enrich_with_evals(ref.synth_trace(f"p{i}", False, random.Random(i))) for i in range(N)]
    return {"rant_score": round(ref.eval_toxicity(TOXIC[-1]), 2), "rant_words": len(TOXIC[-1].split()),
            "labelled": len(labelled), "escaped": len(escaped),
            "escaped_kept": sum(any(t is k for k, _ in kept) for t in escaped),
            "plain_toxic": sum(tox(t) > 0.5 for t in plain)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    traces = traffic(ref)
    truth, kept = stats([(t, 1.0) for t, _ in traces]), policy(traces)
    sampler = ref.TailSampler(sample_rate=RATE, rng=random.Random(3))
    lesson_kept = [(t, 1.0) for t, _ in traces if sampler.decide(t)]
    rest = lambda rows: [(t, w) for t, w in rows if tox(t) <= 0.5]
    rest_truth = stats(rest([(t, 1.0) for t, _ in traces]))
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text()
    return {"kept": len(kept), "truth_toxic": round(100 * truth["toxic"], 1), "truth_u03": round(100 * truth["u_03"], 1),
        "raw": bias([(t, 1.0) for t, _ in kept], truth), "weighted": bias(kept, truth),
        "rest_strata": [bias(rest(rows), rest_truth)["strata"] for rows in (kept, lesson_kept)],
        **escapes(ref, traces, kept), "main_rate": "TailSampler(sample_rate=0.20" in source,
        "eval_first": source.index("enrich_with_evals(trace)") < source.index("sampler.decide(trace)"),
        "doc_rate": "10% of successes" in parity.doc_text(PHASE, LESSON)}


def verify(result):
    r, raw, wt = result, result["raw"], result["weighted"]
    return [
        practice.Check(
            "ANSWER: 1,454 kept; raw toxic share +29.3 points and u_03 +16.1; weighted, both biases round to 0",
            (r["kept"], r["truth_toxic"], raw["toxic"], r["truth_u03"], raw["u_03"], raw["strata"], raw["cost"]) ==
            (1454, 5.0, 29.3, 25.4, 16.1, 6.1, 0.0001) and r["rest_strata"] == [0.0, 2.8]
            and (wt["toxic"], wt["u_03"], wt["strata"]) == (0.0, 0.0, 0.0),
            f"kept {r['kept']}/{N}; true toxic {r['truth_toxic']}%, u_03 {r['truth_u03']}%; bias raw {raw}, "
            f"weighted {wt}; worst stratum error in the sampled rest, stratified vs lesson random {r['rest_strata']}",
        ),
        practice.Check(
            "FINDING: the density toxicity eval scores a 28-word rant 0.36, so 166 of 665 toxic traces escape the rule",
            (r["rant_words"], r["rant_score"], r["labelled"], r["escaped"], r["escaped_kept"], r["plain_toxic"]) ==
            (28, 0.36, 665, 166, 14, 0),
            f"rant: {r['rant_words']} words scores {r['rant_score']}; {r['escaped']}/{r['labelled']} labelled-toxic "
            f"traces score <= 0.5, {r['escaped_kept']} of them kept; lesson traffic over 0.5: {r['plain_toxic']}/{N}",
        ),
        practice.Check(
            "FINDING: main() evaluates every trace before sampling, and samples at 0.20 where the doc says 10%",
            (r["eval_first"], r["main_rate"], r["doc_rate"]) == (True, True, True),
            f"enrich_with_evals before decide in main(): {r['eval_first']}; main.py uses 0.20: {r['main_rate']}; "
            f"doc says 10%: {r['doc_rate']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
