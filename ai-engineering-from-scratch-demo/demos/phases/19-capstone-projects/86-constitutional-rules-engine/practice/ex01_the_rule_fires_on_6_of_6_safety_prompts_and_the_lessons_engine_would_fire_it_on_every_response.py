"""Exercise 1 — the rule fires on 6 of 6 safety prompts, and the lesson's Engine would fire it on every response.

    Add a rule that requires every response to include the phrase "If this is urgent" when the prompt mentions safety. Use composition.

Reading of the exercise: the condition is on the prompt and the requirement
is on the response, but the lesson's `Engine.evaluate(text)` sees only the
response. The rule therefore gets one new key, `applies_when_prompt`, in the
lesson's own predicate grammar. A small `bind_prompt` step evaluates it with
the lesson's `_eval_predicate` against the prompt and, when it is false,
rewrites the rule's `applies_when` to `{not_: {}}` (always false). The
unchanged lesson `Engine` then records it as `not_applicable`, and the
unchanged lesson `Fixer` applies its `prepend_if_missing`. "Mentions safety"
is an `any_of` of two regexes: safe/unsafe/safety, and a
danger/hazard/emergency/injury family. Lookbehinds exclude the programming
compounds thread-safe, memory safety and type-safe. The rule YAML parses
the same under PyYAML and the lesson's `yaml_subset` fallback. The fixture
is 6 safety prompts and 6 others, all harmless.

**ANSWER: with the rule added to the shipped constitution, it reports a
violation on 6/6 safety prompts and `not_applicable` on 6/6 others.** The
fixer prepends "If this is urgent, contact your local emergency number
first." and the second pass is clean. A response that already contains the
phrase passes. The lesson's 5 demo drafts get the same violations as under
the shipped constitution.

**FINDING: the lesson's Engine silently ignores keys it does not know.**
Load the same rule into a plain `Engine(rules=[rule])` and the missing
`applies_when` defaults to "always applies". It reports a violation on 6/6
responses (the 5 demo drafts and a plain answer), with no error.

**FINDING: a plain `\\b(un)?safe(ty)?\\b` trigger fires on 3/6 non-safety
prompts:** "thread-safe", "memory safety" and "type-safe". The hyphen is a
word boundary, so `\\b` does not exclude them. The lookbehinds do.
"""

from __future__ import annotations

import sys

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "86-constitutional-rules-engine"
RULE_YAML = r"""
rules:
  - name: urgent-line-on-safety-prompt
    severity: medium
    applies_when_prompt:
      any_of:
        - contains_regex: '(?<!type[- ])(?<!thread[- ])(?<!memory[- ])\b(un)?safe(ty)?\b'
        - contains_regex: '\b(danger\w*|hazard\w*|emergency|injur\w*)\b'
    must:
      contains_regex: '\bif this is urgent\b'
    explanation: "A response to a safety prompt must say what to do if this is urgent."
    fix:
      prepend_if_missing: "If this is urgent, contact your local emergency number first.\n\n"
"""
NAIVE_TRIGGER = r"\b(un)?safe(ty)?\b"
SAFETY = ["Is it safe to charge my phone overnight?", "What safety gear do I need for cycling?",
          "My smoke alarm keeps beeping. Is that a hazard?", "Is this old ladder unsafe to use?",
          "What should I do in a kitchen fire emergency?", "Can a sprained ankle injury wait until Monday?"]
OTHER = ["Is this Python dict thread-safe?", "Explain Rust's memory safety model.", "Is TypeScript type-safe?",
         "Write a haiku about autumn.", "How do I sort a list in Python?", "Summarise this meeting in three lines."]
ANSWER = "Here are the steps, in order."
ALREADY = "If this is urgent, call your local emergency number. Otherwise, here are the steps."


def load():
    """main.py does `from yaml_subset import load_yaml`; register the lesson's module for that import."""
    sub = parity.load_reference(PHASE, LESSON, "yaml_subset")
    saved = sys.modules.get("yaml_subset")
    sys.modules["yaml_subset"] = sub
    try:
        return sub, parity.load_reference(PHASE, LESSON, "main")
    finally:
        sys.modules.pop("yaml_subset") if saved is None else sys.modules.__setitem__("yaml_subset", saved)


def bind_prompt(rules, prompt, main):
    """Rewrite prompt-scoped rules into the lesson's own grammar for this one call.

    A rule whose `applies_when_prompt` is false for this prompt gets `applies_when: {not_: {}}`
    (always false), so the unchanged lesson Engine records it as not_applicable.
    """
    out = []
    for rule in rules:
        rule = dict(rule)
        scope = rule.pop("applies_when_prompt", None)
        if scope is not None and not main._eval_predicate(scope, prompt)[0]:
            rule["applies_when"] = {"not_": {}}
        out.append(rule)
    return out


def run(main, rules, prompt, response):
    report = main.Engine(rules=bind_prompt(rules, prompt, main)).evaluate(response)
    revised = main.Fixer(rules).apply(response, report.violations())
    again = main.Engine(rules=bind_prompt(rules, prompt, main)).evaluate(revised)
    status = {r.rule_name: r.status for r in report.results}["urgent-line-on-safety-prompt"]
    return status, [v.rule_name for v in again.violations()]


def solve():
    sub, main = load()
    rule = sub.load_yaml(RULE_YAML)["rules"][0]
    rules = main.Engine().rules() + [rule]
    demo = [d["draft"] for d in main._DEMO_DRAFTS]
    return {
        "parsers_agree": sub._parse(RULE_YAML) == sub.load_yaml(RULE_YAML),
        "safety": [run(main, rules, p, ANSWER) for p in SAFETY],
        "other": [run(main, rules, p, ANSWER) for p in OTHER],
        "already": run(main, rules, SAFETY[0], ALREADY),
        "demo_same": [main.Engine(rules=bind_prompt(rules, OTHER[3], main)).evaluate(d).violations()
                      == main.Engine().evaluate(d).violations() for d in demo],
        "plain_engine": sum(r.status == "violation" for d in demo + [ANSWER]
                            for r in main.Engine(rules=[rule]).evaluate(d).results),
        "naive": [bool(main._eval_predicate({"contains_regex": NAIVE_TRIGGER}, p)[0]) for p in OTHER],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the rule fires on 6/6 safety prompts, the fix clears it, 6/6 others not_applicable",
            r["parsers_agree"] and r["safety"] == [("violation", [])] * 6
            and [s for s, _ in r["other"]] == ["not_applicable"] * 6
            and r["already"] == ("pass", []) and all(r["demo_same"]),
            f"safety {r['safety']}; other {[s for s, _ in r['other']]}; already {r['already']}; "
            f"demo verdicts unchanged {r['demo_same']}",
        ),
        practice.Check(
            "FINDING: the lesson Engine ignores the unknown prompt key and fires the rule on 6/6 responses",
            r["plain_engine"] == 6,
            f"Engine(rules=[rule]) violations over 5 demo drafts + 1 plain answer: {r['plain_engine']}/6",
        ),
        practice.Check(
            "FINDING: a plain safe/safety word trigger fires on 3/6 non-safety prompts (thread-/memory-/type-safe)",
            r["naive"] == [True, True, True, False, False, False],
            f"naive trigger on other prompts: {r['naive']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
