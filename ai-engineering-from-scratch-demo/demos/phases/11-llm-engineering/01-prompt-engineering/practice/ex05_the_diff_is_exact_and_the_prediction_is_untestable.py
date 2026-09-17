"""Exercise 5 — the diff is exact, and the prediction half has nothing to test against.

    Create a "prompt diff" tool. Given two versions of a prompt, identify what
    changed (added constraints, removed examples, changed role, modified
    format) and predict whether the change will improve or degrade output
    quality. Test your predictions against actual outputs.

Reading of the exercise: a "version of a prompt" is what `build_prompt` returns
-- pattern, variables, temperature, system -- not the rendered string, so the
diff is taken over that structure and the four categories the exercise names are
assigned from the variable that moved. "Actual outputs" are the only outputs
this lesson has: `simulate_llm_call`'s, scored by `score_response`.

**ANSWER: the diff is exact, 6 of 6 pairs.** The prompts are structured, so
there is nothing to infer: one changed key, one category, no ambiguity.

**FINDING: the four categories miss the change that fires most often.**
Switching pattern also switches temperature -- `PROMPT_PATTERNS` carries 4
distinct recommended values across its 10 rows -- and 2 of the 6 pairs move it.
"Added constraints, removed examples, changed role, modified format" has no name
for the one edit that changes the sampler rather than the text.

**FINDING: a text diff over the rendered prompt cannot attribute the change.**
The structured diff names one category per pair; `difflib` over the same pairs
reports between 2 and 15 changed lines, and on the two pattern swaps not one
substantive line survives -- the whole prompt changed, which is true and useless.

**ANSWER: the prediction half is untestable, and the trivial predictor is
perfect.** Both versions of all 6 pairs score identically, so "no change" is
right 6 times out of 6, and the three-rule predictor is right on exactly the 3
pairs it declined to call.

**CONTROL: against a reply that reads the prompt, predictions become testable.**
Under a stub that echoes the rendered prompt 2 of the 6 pairs separate, and the
three-rule predictor is then right on 1 of 6 -- worse than saying "no change"
every time. Which is the point: a prediction rule has to be calibrated against
outputs, and the lesson has no outputs that move.

Structure: `PAIRS` is the labelled edit set and `prompt_diff` the structured
tool. `text_diff` is the difflib baseline, returning the changed lines and the
substantive lines that survived the edit; `score` runs the lesson's own scorer
over the lesson's own reply; `RULE` is the three-rule prediction.
"""

from __future__ import annotations

import difflib

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "01-prompt-engineering"
CATEGORY = {"role": "changed role", "examples": "removed examples",
            "template_structure": "modified format", "additional_rules": "added constraints"}
RULE = {"added constraints": "improve", "modified format": "improve",
        "removed examples": "degrade"}   # the prediction, as three rules
PERSONA = dict(role="a technical writer", experience="10 years", style="precise",
               priority="clarity", task="Explain API rate limits.")
FEW_SHOT = dict(examples='In: "great"\nOut: positive\n\nIn: "awful"\nOut: negative',
                input="the pasta was perfect")
GUARD = dict(role="Python tutor", domain="Python", additional_rules="Give hints only.",
             question="How do I sort dicts?")
FILL = dict(text="John, Google, 5 years.", template_structure="Name: []\nCompany: []")
CRITERIA = {"required_keywords": ["Exclude", "Out:", "step"], "max_words": 40,
            "expected_format": "numbered_list"}

PAIRS = [  # (true category, before, after), one per category plus two pattern swaps
    ("added constraints", ("guardrail", GUARD),
     ("guardrail", GUARD | {"additional_rules": "Give hints only. Never cite URLs."})),
    ("removed examples", ("few_shot", FEW_SHOT),
     ("few_shot", FEW_SHOT | {"examples": 'In: "great"\nOut: positive'})),
    ("changed role", ("persona", PERSONA), ("persona", PERSONA | {"role": "a junior agent"})),
    ("modified format", ("template_fill", FILL),
     ("template_fill", FILL | {"template_structure": '{"name": "", "company": ""}'})),
    ("changed pattern", ("persona", PERSONA), ("chain_of_thought", {"problem": "Rate limits?"})),
    ("changed pattern", ("few_shot", FEW_SHOT), ("decomposition", {"problem": "Rate limits?"})),
]


def prompt_diff(a, b, before_spec, after_spec):
    """The prompt keys that moved, the variables that moved, and the category."""
    structural = [k for k in ("pattern", "temperature", "system") if a[k] != b[k]]
    moved = [k for k in set(before_spec[1]) | set(after_spec[1])
             if before_spec[1].get(k) != after_spec[1].get(k)]
    category = "changed pattern" if "pattern" in structural else CATEGORY[moved[0]]
    return structural, moved, category


def text_diff(a, b):
    delta = list(difflib.ndiff(a["user"].splitlines(), b["user"].splitlines()))
    return (sum(1 for ln in delta if ln[0] in "+-"),
            sum(1 for ln in delta if ln.startswith("  ") and ln.strip()))


def score(ref, prompt):
    return ref.score_response(ref.run_prompt_test(prompt, ["gpt-4o"])["gpt-4o"]["response"],
                              CRITERIA)["composite_score"]


def one_pair(ref, truth, before_spec, after_spec):
    a, b = ref.build_prompt(*before_spec), ref.build_prompt(*after_spec)
    structural, moved, category = prompt_diff(a, b, before_spec, after_spec)
    lines, survived, delta = *text_diff(a, b), score(ref, b) - score(ref, a)
    return {"truth": truth, "category": category, "lines": lines, "whole": survived == 0,
            "keys": len(structural) + len(moved), "temperature": "temperature" in structural,
            "prediction": RULE.get(truth, "no change"), "outcome": "improve" if delta > 0
            else "degrade" if delta < 0 else "no change"}


def echo_stub(model_name, request):
    """The control: a reply that is the prompt, so an edit can move the score."""
    words = " ".join(m["content"] for m in request["messages"]).split()[:90]
    return {"response": " ".join(words), "tokens_used": {"total": 0}, "latency_ms": 0,
            "finish_reason": "stop"}   # the prompt itself, truncated


def scored(rows):   # the prediction half, tallied
    return {"outcomes": sorted({r["outcome"] for r in rows}),
            "spoken": sum(r["prediction"] != "no change" for r in rows),
            "right": sum(r["prediction"] == r["outcome"] for r in rows)}


def report(rows):
    return {"exact": sum(r["category"] == r["truth"] for r in rows), "pairs": len(rows),
            "categories": sorted({r["category"] for r in rows}),
            "temperature_moves": sum(r["temperature"] for r in rows),
            "keys": [r["keys"] for r in rows], "lines": [r["lines"] for r in rows],
            "whole": sum(r["whole"] for r in rows), **scored(rows)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "prompt_engineering")
    rows = [one_pair(ref, *pair) for pair in PAIRS]
    original = ref.simulate_llm_call
    try:
        ref.simulate_llm_call = echo_stub
        control = [one_pair(ref, *pair) for pair in PAIRS]
    finally:
        ref.simulate_llm_call = original
    return {**report(rows), "control_outcomes": [r["outcome"] for r in control],
            "control_right": scored(control)["right"],
            "temperatures": len({p["temperature"] for p in ref.PROMPT_PATTERNS.values()})}


def verify(result):
    lines, keys = result["lines"], result["keys"]
    return [
        practice.Check(
            "ANSWER: the structured diff is exact on every pair",
            result["exact"] == result["pairs"],
            f"{result['exact']} of {result['pairs']} edits are named correctly from the "
            f"prompt structure alone, over {result['categories']} -- `build_prompt` returns "
            "pattern, variables and temperature, so the change is read, not inferred",
        ),
        practice.Check(
            "FINDING: the exercise's four categories have no name for temperature",
            all([result["temperature_moves"] == 2, result["temperatures"] == 4]),
            f"{result['temperature_moves']} of the {result['pairs']} pairs move the sampling "
            f"temperature, because PROMPT_PATTERNS pins one per pattern across "
            f"{result['temperatures']} values. Added constraints, removed examples, changed "
            "role and modified format name none of it",
        ),
        practice.Check(
            "FINDING: a text diff over the rendered prompt cannot attribute the change",
            all([result["whole"] == 2, max(lines) >= 14, min(lines) == 2]),
            f"the structured diff names one category per pair, from {keys} changed keys; "
            f"difflib reports {lines} changed lines, and on the {result['whole']} pattern "
            "swaps not one substantive line survives -- the whole prompt changed, which is "
            "true and names none of the four",
        ),
        practice.Check(
            "ANSWER: nothing can be tested against these outputs",
            all([result["outcomes"] == ["no change"],
                 result["right"] == result["pairs"] - result["spoken"]]),
            f"both versions of all {result['pairs']} pairs score identically, so every "
            f"outcome is {result['outcomes']}. The predictor speaks on {result['spoken']} "
            f"pairs and is right on {result['right']} -- exactly the ones it declined to "
            "call. Always saying 'no change' scores full marks",
        ),
        practice.Check(
            "CONTROL: against a reply that reads the prompt, the pairs separate",
            len(set(result["control_outcomes"])) > 1,
            f"under a stub that echoes the rendered prompt the outcomes become "
            f"{result['control_outcomes']} and the three-rule predictor is right on "
            f"{result['control_right']} of {result['pairs']} -- worse than the trivial one. "
            "A prediction rule needs outputs to calibrate on; this lesson has none",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
