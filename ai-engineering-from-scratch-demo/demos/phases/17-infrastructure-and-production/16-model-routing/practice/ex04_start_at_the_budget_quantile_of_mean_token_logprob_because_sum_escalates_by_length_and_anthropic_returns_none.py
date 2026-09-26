"""Exercise 4 — start at the budget quantile of mean token logprob, because sum escalates by length and Anthropic returns none.

    Implement a confidence check using logprobs from OpenAI / Anthropic APIs.
    What's the threshold you start with?

Reading of the exercise: the check is written against the real response
shape -- OpenAI Chat Completions with `logprobs=True` returns
`choices[0].logprobs.content[]`, one `{token, logprob, top_logprobs}` per
output token -- and run at T0 over synthetic bodies of that shape, one per
request in the reference's 1000-request workload. Each cheap answer is
correct with probability `quality("cheap", q)`; an output token is "unsure"
(p ~ U(0.05, 0.6), else U(0.95, 1)) with chance 3% in a correct answer and 6%
in a wrong one. Those two rates are this solution's assumption, not measured
model behaviour, so only comparisons made under them are claimed. The real
call is `client.chat.completions.create(model=..., messages=...,
logprobs=True, top_logprobs=5)`.

**ANSWER: start at the quantile that spends your escalation budget, on the
mean token logprob -- here -0.0946, a geometric-mean token probability of
0.910.** A fixed constant does not carry between models, since logprob
scales differ. So the starting point is the budget: the lesson's "~10% of
traffic", taken as the 10th percentile of a calibration sample. Under the
assumed traces that cascade escalates 100 requests and catches 37 of the
cheap model's 49 wrong answers: $1.44 at 98.8%, against the reference
cascade's $4.42 at 98.24% with 229 escalations.

**FINDING: summing logprobs escalates by length, not by doubt.** At the
same 10% budget the sequence logprob (sum) sends 76 hard, 24 medium and 0
simple queries up. It catches 29 wrong answers and costs $2.99, because a
long answer has a low total however sure each token is. The minimum token
logprob catches 14. Normalise by length.

**FINDING: Anthropic's Messages API returns no logprobs, and the lesson's
code reads none.** The Messages API reference lists no `logprobs` or
`top_logprobs` parameter and no per-token probabilities in the response
(checked 2026-09-26). On an Anthropic-shaped body the check returns None:
use a verifier call or a hedging/refusal check there. The reference
cascade's "confidence" is `q.difficulty` plus a coin, and `main.py` never
mentions a logprob.

Structure: `openai_response()` builds a body; `token_logprobs()` is the
provider-shape check; `calibrate()` sets the threshold; `cascade()` prices it
on the reference's `cost_of`.
"""

from __future__ import annotations

import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "16-model-routing"
BUDGET = 0.10  # the lesson's "~10% of traffic" escalated
SLIP = {True: 0.03, False: 0.06}  # assumed chance of an unsure token, by correctness
SCORES = {"mean": lambda lp: sum(lp) / len(lp), "min": min, "sum": sum}


def openai_response(rng, q, correct):
    """A Chat Completions body with logprobs=True."""
    content = []
    for i in range(q.output_tokens):
        p = (
            rng.uniform(0.05, 0.6)
            if rng.random() < SLIP[correct]
            else rng.uniform(0.95, 1.0)
        )
        content.append({"token": f"t{i}", "logprob": math.log(p), "top_logprobs": []})
    return {
        "choices": [{"message": {"content": "..."}, "logprobs": {"content": content}}]
    }


def token_logprobs(response):
    """Per-token logprobs, or None when the body carries none (an Anthropic Message)."""
    choices = response.get("choices")
    if not choices or not choices[0].get("logprobs"):
        return None
    return [t["logprob"] for t in choices[0]["logprobs"]["content"]]


def calibrate(scores, budget=BUDGET):
    """Escalate the least confident `budget` share of a calibration sample."""
    return sorted(scores)[int(budget * len(scores))]


def cascade(ref, traces, score):
    values = [SCORES[score](lp) for _, _, lp in traces]
    threshold = calibrate(values)
    up = [(q, ok) for (q, ok, _), v in zip(traces, values) if v < threshold]
    return summarize(ref, traces, up, threshold)


def summarize(ref, traces, up, threshold):
    caught = sum(not ok for _, ok in up)
    cost = sum(ref.cost_of("cheap", q) for q, _, _ in traces)
    cost += sum(ref.cost_of("frontier", q) for q, _ in up)
    wrong_kept = sum(not ok for _, ok, _ in traces) - caught
    kinds = [q.difficulty for q, _ in up]
    return {
        "threshold": round(threshold, 4),
        "cost": round(cost, 2),
        "quality": round(1 - wrong_kept / len(traces), 4),
        "caught": caught,
        "up": tuple(kinds.count(d) for d in ("simple", "medium", "hard")),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng, traces = random.Random(3), []
    for q in ref.make_workload():
        ok = rng.random() < ref.quality("cheap", q)
        traces.append((q, ok, token_logprobs(openai_response(rng, q, ok))))
    shipped = ref.simulate("CASCADE", ref.make_workload())
    return {
        **{k: cascade(ref, traces, k) for k in SCORES},
        "wrong": sum(not ok for _, ok, _ in traces),
        "shipped": (
            round(shipped["cost"], 2),
            round(shipped["mean_quality"], 4),
            shipped["escalated"],
        ),
        "anthropic": token_logprobs({"content": [{"type": "text", "text": "..."}]}),
        "ref_reads_logprobs": "logprob" in inspect.getsource(ref),
    }


def verify(result):
    mean, total, low = result["mean"], result["sum"], result["min"]
    return [
        practice.Check(
            "ANSWER: start at the budget quantile of the mean token logprob",
            mean["threshold"] == -0.0946
            and sum(mean["up"]) == 100
            and mean["caught"] == 37,
            f"threshold {mean['threshold']} (p={math.exp(mean['threshold']):.3f}) escalates "
            f"{sum(mean['up'])}, catching {mean['caught']} of {result['wrong']} wrong answers: "
            f"${mean['cost']} at {mean['quality']} vs the reference cascade {result['shipped']}",
        ),
        practice.Check(
            "FINDING: summing logprobs escalates by length, not by doubt",
            total["up"] == (0, 24, 76)
            and total["caught"] < mean["caught"]
            and low["caught"] == 14,
            f"sum escalates (simple, medium, hard) {total['up']}, catches {total['caught']} for "
            f"${total['cost']}; min catches {low['caught']}",
        ),
        practice.Check(
            "FINDING: Anthropic's Messages API returns no logprobs, and the lesson's code reads none",
            result["anthropic"] is None and not result["ref_reads_logprobs"],
            "an Anthropic-shaped body yields None; main.py's cascade confidence is the "
            "difficulty label plus a coin",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
