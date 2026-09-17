"""Exercise 1 — every pattern ties at 0.000, and the winner is dictionary order.

    Take the 5 test cases in `TEST_SUITE` and add 5 more that cover the
    remaining patterns (meta-prompt, decomposition, critique, audience
    adaptation, boundary). Run the full suite and identify which pattern
    produces the most consistent scores across models.

Reading of the exercise: "most consistent scores across models" is read as the
population standard deviation of `composite_score` over the three models in
`MODEL_CONFIGS`, per test -- the only reading under which "across models" means
anything. The 5 new cases use the lesson's own criteria vocabulary and its own
`build_prompt`, so the suite that runs is the lesson's suite at 10 rows.

**ANSWER: all ten patterns tie, at a spread of exactly 0.000.** Not
approximately: every model scores every test identically, so "most consistent"
has ten joint winners and the question cannot select between patterns.

**MECHANISM: `simulate_llm_call` never reads the prompt.** Its reply is a fixed
per-model sentence plus `md5(request)[:8]`. Two prompts of the same model differ
in 8 characters out of 139 -- and those 8 are hex, so no task keyword can appear
in them. The prompt is an input to the hash and to nothing else.

**FINDING: the composite score is a function of the criteria dict alone.**
`(1 if words <= max_words) + keyword_coverage + (1 if no forbidden) +
(1 if format_valid)`, over the number of criteria keys present. Every canned
reply is 19-21 words, contains no task keyword and is not JSON, so the score of
a test can be computed before it is run: predicted equals measured, 10 of 10.

**FINDING: "gpt-4o: 5 wins out of 5" is `MODEL_CONFIGS` insertion order.**
`compare_models` sorts equal scores, and `sorted` is stable, so rank 1 is
whichever model was declared first. Reverse the declaration and Gemini wins all
ten with no score changing anywhere.

**CONTROL: the suite discriminates as soon as the reply depends on the prompt.**
A stub that echoes the rendered prompt back, truncated to a per-model word
budget, breaks the tie -- the spread goes above zero and a strict ranking
appears. The suite design is sound; the simulator is what flattens it.

Structure: `EXTRA` holds the 5 new cases, `spread` the cross-model stdev,
`predict` the criteria-only formula and `echo_stub` the control simulator.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "01-prompt-engineering"
HEX = set("0123456789abcdef")
BUDGET = {"gpt-4o": 80, "claude-3.5-sonnet": 50, "gemini-1.5-pro": 120}
# what each criteria key contributes for a 21-word reply holding no task word
IMPLIED = {"max_words": lambda v: float(21 <= v), "required_keywords": lambda v: 0.0,
           "forbidden_phrases": lambda v: 1.0,
           "expected_format": lambda v: float(v not in
                                              ("json", "bullet_points", "numbered_list"))}

EXTRA = [
    ("meta_prompt", dict(objective="summarise support tickets", metric="brevity",
                         model="gpt-4o"), dict(required_keywords=["role", "format"])),
    ("decomposition", dict(problem="Plan a zero-downtime database migration."),
     dict(required_keywords=["sub-problem"], max_words=300)),
    ("critique", dict(task="Draft a deprecation notice for an API endpoint."),
     dict(required_keywords=["critique"], forbidden_phrases=["as an AI"])),
    ("audience_adapt", dict(concept="vector embeddings", audience="a product manager",
                            length="150 words", include="one analogy", exclude="equations"),
     dict(required_keywords=["embedding"], expected_format="bullet_points")),
    ("boundary", dict(scope="billing questions", refusal_message="I only handle billing.",
                      user_input="What is the capital of Peru?"),
     dict(required_keywords=["billing"], max_words=100, forbidden_phrases=["Peru"])),
]


def scores(ref, pattern, variables, criteria, models=None):
    """One test row: per-model composite, the rank-1 model, the comparison table."""
    prompt = ref.build_prompt(pattern, variables)
    table, ranked = ref.compare_models(ref.run_prompt_test(prompt, models), criteria)
    return ({m: d["scores"]["composite_score"] for m, d in table.items()},
            ranked[0][0], table)


def spread(per_model):
    return round(statistics.pstdev(per_model.values()), 6)


def predict(criteria):
    """The score implied by the criteria alone, with no model involved."""
    parts = [IMPLIED[k](v) for k, v in criteria.items() if k in IMPLIED]
    return round(sum(parts) / len(parts), 3) if parts else 0.0


def echo_stub(model_name, request):
    """The control: a reply that depends on the prompt, truncated per model."""
    return {"response": " ".join(str(request).split()[:BUDGET[model_name]]),
            "tokens_used": {"total": 0}, "latency_ms": 0, "finish_reason": "stop"}


def hash_window(ref):
    """Where two replies of one model differ, and what sits in that window."""
    reply = [ref.simulate_llm_call("gpt-4o", {"n": n})["response"] for n in (1, 2)]
    differ = [i for i, (a, b) in enumerate(zip(*reply)) if a != b]
    window = {r[differ[0]:differ[-1] + 1] for r in reply}
    return {"span": differ[-1] - differ[0] + 1, "varying": sorted(window),
            "length": len(reply[0]), "hex": all(set(h) <= HEX for h in window)}


def under_stub(ref, rows):
    """The same ten tests against a reply that reads the prompt."""
    original = ref.simulate_llm_call
    try:
        ref.simulate_llm_call = echo_stub
        return [spread(scores(ref, *row)[0]) for row in rows]
    finally:
        ref.simulate_llm_call = original


def report(measured, flipped):
    return {
        "spreads": [spread(s) for s, _, _ in measured],
        "measured": [sorted(set(s.values()))[0] for s, _, _ in measured],
        "coverage": sorted({t["gpt-4o"]["scores"]["keyword_coverage"] for _, _, t in measured}),
        "winners": sorted({w for _, w, _ in measured}),
        "flipped": sorted({w for _, w, _ in flipped}),
        "flipped_scores": [sorted(set(s.values())) for s, _, _ in flipped],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "prompt_engineering")
    rows = [(t["pattern"], t["variables"], t["criteria"]) for t in ref.TEST_SUITE] + EXTRA
    order = list(ref.MODEL_CONFIGS)
    measured = [scores(ref, *row) for row in rows]
    flipped = [scores(ref, *row, models=order[::-1]) for row in rows]
    return {**report(measured, flipped), "order": order,
            "patterns": [r[0] for r in rows], "predicted": [predict(r[2]) for r in rows],
            "window": hash_window(ref), "control": under_stub(ref, rows)}


def verify(result):
    spreads, control, window = result["spreads"], result["control"], result["window"]
    widest = max(zip(control, result["patterns"]))[1]
    return [
        practice.Check(
            "ANSWER: all ten patterns tie -- the cross-model spread is exactly 0.000",
            all([len(spreads) == 10, set(spreads) == {0.0}]),
            f"{len(spreads)} tests, {len(set(spreads))} distinct spread value: "
            f"{sorted(set(spreads))}. 'Most consistent' has ten joint winners, so the "
            "question the exercise asks cannot separate one pattern from another",
        ),
        practice.Check(
            "MECHANISM: the reply is a constant plus md5[:8], and hex holds no keyword",
            all([window["span"] == 8, window["hex"], result["coverage"] == [0.0]]),
            f"two prompts of one model give replies differing only inside an "
            f"{window['span']}-character window of {window['length']} -- the hash prefix "
            f"{window['varying']}, drawn from [0-9a-f]. keyword_coverage is "
            f"{result['coverage']} on all ten tests: no task word is spellable there",
        ),
        practice.Check(
            "FINDING: the composite score is a function of the criteria dict alone",
            result["predicted"] == result["measured"],
            f"predicted from the criteria with no model involved: {result['predicted']}; "
            f"measured by running the suite: {result['measured']}. The suite scores the "
            "shape of the criteria it was handed, not the prompt and not the model",
        ),
        practice.Check(
            "FINDING: the reported winner is MODEL_CONFIGS insertion order, not a result",
            all([result["winners"] == result["order"][:1],
                 result["flipped"] == result["order"][-1:],
                 result["flipped_scores"] == [[m] for m in result["measured"]]]),
            f"every test ranks {result['winners']} first out of {result['order']}; pass "
            f"the same three models in reverse and every test ranks {result['flipped']} "
            "first, with all thirty scores unchanged. `sorted` is stable, so rank 1 is "
            "whichever model was listed first -- the whole content of the lesson's "
            "'gpt-4o: 5 wins out of 5' summary",
        ),
        practice.Check(
            "CONTROL: a prompt-dependent stub breaks the tie and ranks the patterns",
            all([max(control) > 0.0, len(set(control)) > 1]),
            f"replying with the rendered prompt truncated to a per-model word budget "
            f"gives spreads from {min(control)} to {max(control)}, widest on {widest!r}. "
            "The suite discriminates the moment the reply reads the prompt, so what is "
            "flat here is the simulator and not the test design",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
