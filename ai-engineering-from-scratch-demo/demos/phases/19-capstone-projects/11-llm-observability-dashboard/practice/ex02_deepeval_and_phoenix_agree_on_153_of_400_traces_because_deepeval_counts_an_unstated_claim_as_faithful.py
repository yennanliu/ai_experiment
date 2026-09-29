"""Exercise 2 — DeepEval and Phoenix agree on 153 of 400 traces because DeepEval counts an unstated claim as faithful.

    Swap DeepEval for Phoenix evaluators on the same traces. Measure score drift between the two eval engines.

Reading of the exercise: neither library is installed and both are
LLM-judged, so each engine is reduced to its published scoring rule, and
both rules read the same per-claim verdicts. The verdicts stand in for the
judge LLM's calls, so any drift that remains comes from the engines'
definitions and not from judge noise. DeepEval faithfulness is truthful
claims / all claims, where a claim is truthful if it does not contradict
the retrieval context; pass is score >= 0.5
(https://deepeval.com/docs/metrics-faithfulness, read 2026-09-29). Phoenix
hallucination is one binary label per response: "grounded" = 0.0 if every
claim restates or follows from the input, otherwise "hallucinated" = 1.0
(https://arize.com/docs/phoenix/evaluation/running-pre-tested-evals/hallucinations,
read 2026-09-29). The traces are 400 of the lesson's own `synth_trace`
spans, with the response rebuilt as 1-4 claims drawn from a labelled bank
(supported / unstated / contradicted against the span's context). The
lesson's `eval_faithfulness` is scored alongside.

**ANSWER: mean drift is 0.562 on a 0-1 faithfulness scale, and the two
engines agree on pass/fail for 153 of 400 traces (38.3%).** DeepEval
averages 0.907 and Phoenix, flipped to "1 = grounded", averages 0.345. All
of the disagreement runs one way: 247 traces pass DeepEval and are
hallucinated to Phoenix, and 0 go the other way. The cause is the unstated
claim. DeepEval counts a claim the context never mentions as truthful;
Phoenix counts it as a hallucination.

**FINDING: a drop-in swap also inverts polarity, and the bug looks like
better agreement.** Phoenix's score is 1.0 = hallucinated, while DeepEval's
is 1.0 = faithful. Read raw against DeepEval's 0.5 threshold, Phoenix
"passes" exactly the traces it calls hallucinated. That agrees with DeepEval
on 247/400 (61.8%), more than the correct reading's 153.

**FINDING: DeepEval's rule scores the lesson's PII regression 1.0.** The
leak response "your ssn is 123-45-6789" contradicts nothing in the context,
so faithfulness is 1.0; Phoenix labels it hallucinated. The lesson's own
`eval_faithfulness` token overlap gives the faithful answer "the weather in
Tokyo is mild" 0.5, with half its words ("the", "in", "is") missing from
the context. That is exactly the pass threshold. Over the 400 traces it
correlates with DeepEval at r = 0.13 and with Phoenix at r = 0.83, so it
behaves like the stricter engine.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "11-llm-observability-dashboard"
N, SEED = 400, 7
CONTEXT = "relevant weather context Tokyo mild"
BANK = {
    "supported": ["the weather in Tokyo is mild", "Tokyo is mild", "Tokyo weather is mild today"],
    "unstated": ["pack an umbrella for the evening", "cherry blossoms peak in early April",
                 "the Shinkansen reaches Osaka in about two hours"],
    "contradicted": ["Tokyo is freezing this week", "the weather in Tokyo is extreme heat"],
}
MIX = ["supported"] * 6 + ["unstated"] * 3 + ["contradicted"]


def make_fixture(ref):
    rng, traces = random.Random(SEED), []
    for i in range(N):
        span = ref.synth_trace(f"t{i:04d}", False, rng)[1]
        claims = [(v, rng.choice(BANK[v])) for v in (rng.choice(MIX) for _ in range(rng.randint(1, 4)))]
        span.attributes["response"] = ". ".join(c for _, c in claims)
        traces.append({"span": span, "verdicts": [v for v, _ in claims]})
    return traces


def deepeval_faithfulness(verdicts):
    return sum(v != "contradicted" for v in verdicts) / len(verdicts)


def phoenix_hallucination(verdicts):
    return 0.0 if all(v == "supported" for v in verdicts) else 1.0


def compare(de, ph):
    """Drift and pass/fail agreement, with Phoenix read as 1 = grounded and then raw."""
    grounded = [1 - p for p in ph]
    pairs = list(zip([d >= 0.5 for d in de], grounded))
    return {"de_mean": statistics.fmean(de), "ph_mean": statistics.fmean(grounded),
            "drift": statistics.fmean(abs(d - g) for d, g in zip(de, grounded)),
            "agree": sum(p == (g == 1.0) for p, g in pairs), "agree_raw": sum(p == (g == 0) for p, g in pairs),
            "de_only": pairs.count((True, 0.0)), "ph_only": pairs.count((False, 1.0))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    traces = make_fixture(ref)
    de = [deepeval_faithfulness(t["verdicts"]) for t in traces]
    ph = [phoenix_hallucination(t["verdicts"]) for t in traces]
    lesson = [ref.eval_faithfulness(t["span"].attributes["response"], CONTEXT) for t in traces]
    return {
        **compare(de, ph),
        "leak": (deepeval_faithfulness(["unstated"]), phoenix_hallucination(["unstated"])),
        "leak_text": ref.synth_trace("x", True, random.Random(0))[1].attributes["response"],
        "lesson_ok_answer": ref.eval_faithfulness("the weather in Tokyo is mild", CONTEXT),
        "r_lesson": [round(statistics.correlation(lesson, x), 2) for x in (de, [1 - p for p in ph])],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: mean drift 0.562 and pass/fail agreement 153/400, all of the gap one-way",
            (round(r["drift"], 3), round(r["de_mean"], 3), round(r["ph_mean"], 3), r["agree"], r["de_only"],
             r["ph_only"]) == (0.562, 0.907, 0.345, 153, 247, 0),
            f"DeepEval mean {r['de_mean']:.3f}, Phoenix grounded mean {r['ph_mean']:.3f}, mean |drift| "
            f"{r['drift']:.3f}; agree {r['agree']}/{N}; DeepEval-pass-Phoenix-hallucinated {r['de_only']}, "
            f"reverse {r['ph_only']}",
        ),
        practice.Check(
            "FINDING: a drop-in swap inverts polarity, and raw agreement rises to 247/400",
            r["agree_raw"] == 247, f"DeepEval pass vs raw Phoenix score >= 0.5: {r['agree_raw']}/{N}",
        ),
        practice.Check(
            "FINDING: DeepEval's rule scores the PII leak 1.0; the lesson's overlap scores a faithful answer 0.5",
            (r["leak"], r["leak_text"], r["lesson_ok_answer"], r["r_lesson"]) ==
            ((1.0, 1.0), "your ssn is 123-45-6789", 0.5, [0.13, 0.83]),
            f"leak '{r['leak_text']}': DeepEval {r['leak'][0]}, Phoenix hallucination {r['leak'][1]}; lesson "
            f"overlap on a faithful answer {r['lesson_ok_answer']}; r(lesson, DeepEval / Phoenix) {r['r_lesson']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
