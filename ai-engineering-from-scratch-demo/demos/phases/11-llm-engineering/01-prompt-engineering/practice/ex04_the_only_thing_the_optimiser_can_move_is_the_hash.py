"""Exercise 4 — the optimiser is correct, and the only thing it can move is the hash.

    Implement a prompt optimizer. Given a prompt and a scoring criteria, run
    the prompt 5 times with temperature=0.7, score each output, identify the
    weakest criteria, and rewrite the prompt to address it. Repeat for 3
    iterations. Measure whether scores improve.

Reading of the exercise: the optimiser is built exactly as specified -- 5
samples at temperature 0.7, score, pick the lowest-scoring criterion, rewrite
the prompt to address that criterion, three rounds -- and run against the
lesson's own `score_response` and `simulate_llm_call`. "Measure whether scores
improve" is then a question with a provable answer rather than an empirical one.

**ANSWER: temperature 0.7 produces no variance and three rounds improve the
score by exactly 0.000.** The five samples of each round are one distinct
string. `simulate_llm_call` hashes the request, and temperature is part of the
request, not a source of randomness -- so sampling five times costs five sleeps
and yields one observation.

**FINDING: "identify the weakest criteria" is a constant.** It is
`keyword_coverage`, at 0.0, in every round of every test in the suite. The
canned reply contains no task word, so the step that is supposed to steer the
rewrite always points the same way.

**MECHANISM: the objective is `md5(request)[:8]`, and the prompt is only its
input.** The score reads the reply; the reply is a per-model constant plus that
hash. So the optimiser's entire search space maps onto 8 hex characters, and
every criterion except a hex-spellable keyword is constant on all of it.

**FINDING: give it a criterion inside the hash alphabet and it wins.** Asking
for the keyword "cafe", the same rewrite loop finds a prompt whose hash contains
it in a few thousand candidates, taking coverage 0.0 -> 1.0 and the composite
0.667 -> 1.0. The optimiser works. What it optimises is the hash.

**CONTROL: against a reply that reads the prompt, the same loop improves.**
Swapping in a stub that echoes the rendered prompt lifts the composite across
the three rounds, so what is flat here is the objective and not the search.

Structure: `optimise` is the loop the exercise describes, `rewrite` its
per-criterion edit, and `mine` the same loop aimed at a hex-spellable keyword.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "01-prompt-engineering"
MODEL, SAMPLES, ROUNDS = "gpt-4o", 5, 3
CRITERIA = {"max_words": 200, "required_keywords": ["rate limit", "API", "requests"],
            "forbidden_phrases": ["in conclusion"]}
EDITS = {
    "keyword_coverage": "Your answer must use these exact words: {required_keywords}.",
    "length_compliant": "Answer in at most {max_words} words.",
    "no_violations": "Never write any of: {forbidden_phrases}.",
    "format_valid": "Reply with one {expected_format} value and nothing else.",
}


def sample(ref, prompt, criteria, n=SAMPLES):
    replies = [ref.run_prompt_test(prompt, [MODEL])[MODEL]["response"] for _ in range(n)]
    return replies, [ref.score_response(r, criteria) for r in replies]


def weakest(scored):
    """The lowest-scoring criterion, over the keys the composite actually averages."""
    graded = {k: float(v) for k, v in scored[0].items()
              if k != "composite_score" and isinstance(v, (bool, float))}
    return min(graded, key=graded.get), min(graded.values())


def rewrite(prompt, key, criteria):
    edit = EDITS[key].format(**{k: v for k, v in criteria.items()})
    return {**prompt, "user": prompt["user"] + "\n\n" + edit}


def optimise(ref, prompt, criteria, rounds=ROUNDS):
    history = []
    for _ in range(rounds):
        replies, scored = sample(ref, prompt, criteria)
        composite = [s["composite_score"] for s in scored]
        key, low = weakest(scored)
        history.append({"distinct": len(set(replies)), "mean": statistics.mean(composite),
                        "spread": round(statistics.pstdev(composite), 6),
                        "weakest": key, "low": low})
        prompt = rewrite(prompt, key, criteria)
    return history, prompt


def mine(ref, prompt, target):
    """The same rewrite step, aimed at a keyword the hash alphabet can spell."""
    for tries in itertools.count(1):
        candidate = {**prompt, "user": prompt["user"] + "​" * tries}
        request = ref.format_openai_request(candidate)
        digest = hashlib.md5(json.dumps(request, sort_keys=True).encode()).hexdigest()[:8]
        if target in digest:
            return tries, candidate, digest


def echo_stub(model_name, request):
    words = " ".join(str(request).split()).split()[:120]
    return {"response": " ".join(words), "tokens_used": {"total": 0},
            "latency_ms": 0, "finish_reason": "stop"}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "prompt_engineering")
    base = ref.build_prompt("persona", ref.TEST_SUITE[0]["variables"])
    base["temperature"] = 0.7
    history, _ = optimise(ref, base, CRITERIA)
    every = [weakest(sample(ref, ref.build_prompt(t["pattern"], t["variables"]),
                            t["criteria"], 1)[1])[0] for t in ref.TEST_SUITE]
    tries, mined, digest = mine(ref, base, "cafe")
    hexed = dict(CRITERIA, required_keywords=["cafe"])
    before = ref.score_response(sample(ref, base, hexed, 1)[0][0], hexed)
    after = ref.score_response(sample(ref, mined, hexed, 1)[0][0], hexed)
    original = ref.simulate_llm_call
    try:
        ref.simulate_llm_call = echo_stub
        control, _ = optimise(ref, base, CRITERIA)
    finally:
        ref.simulate_llm_call = original
    return {
        "means": [round(h["mean"], 3) for h in history],
        "distinct": [h["distinct"] for h in history],
        "spreads": [h["spread"] for h in history],
        "weakest": sorted({h["weakest"] for h in history}),
        "lows": sorted({h["low"] for h in history}),
        "weakest_all": sorted(set(every)), "tries": tries, "digest": digest,
        "before": (before["keyword_coverage"], before["composite_score"]),
        "after": (after["keyword_coverage"], after["composite_score"]),
        "control": [round(h["mean"], 3) for h in control],
    }


def verify(result):
    means, control = result["means"], result["control"]
    return [
        practice.Check(
            "ANSWER: temperature 0.7 gives one observation, not five",
            all([result["distinct"] == [1] * ROUNDS, result["spreads"] == [0.0] * ROUNDS]),
            f"each of the {ROUNDS} rounds draws {SAMPLES} samples and gets "
            f"{result['distinct']} distinct reply, spread {result['spreads']}. "
            "`simulate_llm_call` hashes the request and temperature is part of that request, "
            "so sampling buys five sleeps and no spread",
        ),
        practice.Check(
            "ANSWER: three rounds improve the score by exactly 0.000",
            len(set(means)) == 1,
            f"round means {means}, total improvement {means[-1] - means[0]:.3f}. Each "
            "rewrite changes the prompt, the prompt changes the hash, and the hash changes "
            "8 characters of a reply that scores the same",
        ),
        practice.Check(
            "FINDING: 'identify the weakest criteria' returns the same answer every time",
            all([result["weakest"] == ["keyword_coverage"], result["lows"] == [0.0],
                 result["weakest_all"] == ["keyword_coverage"]]),
            f"the weakest criterion is {result['weakest']} at {result['lows']} in all "
            f"{ROUNDS} rounds, and the same criterion in all five tests of the lesson's "
            "suite. The step that is supposed to steer the rewrite has one possible output",
        ),
        practice.Check(
            "FINDING: aim it at a hex-spellable keyword and the optimiser wins",
            all([result["before"] == (0.0, 0.667), result["after"] == (1.0, 1.0)]),
            f"required_keywords ['cafe'] is reachable because the reply's only variable part "
            f"is md5(request)[:8]. The same loop finds it in {result['tries']} candidates "
            f"(digest {result['digest']!r}), taking coverage and composite "
            f"{result['before']} -> {result['after']}. The optimiser is correct, and its "
            "entire search space is eight hex characters",
        ),
        practice.Check(
            "CONTROL: against a reply that reads the prompt, the same loop improves",
            control[-1] > control[0],
            f"a stub that echoes the rendered prompt back lifts the round means "
            f"{control}, improvement {control[-1] - control[0]:+.3f}, using the identical "
            "optimise() and rewrite(). What is flat in the graded run is the objective",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
