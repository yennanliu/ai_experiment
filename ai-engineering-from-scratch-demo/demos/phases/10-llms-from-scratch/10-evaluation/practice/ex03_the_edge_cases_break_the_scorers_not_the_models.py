"""Exercise 3 — the two scorers disagree by 1.0 on the empty email, and by design on the ambiguous ones.

    Build an eval suite for a specific task: email classification into 5
    categories. Create 100 test cases with diverse examples including edge cases
    (emails that could belong to multiple categories, empty emails, emails in
    other languages). Measure how different "models" (rule-based, keyword
    matching, simulated LLM) perform.

Reading of the exercise: the suite is 100 cases built through the lesson's own
`EvalCase` and `EvalSuite`, with the three edge-case families it names present
by construction, and the three "models" are scored with the lesson's own
`exact_match` and `token_f1`. The edge cases are the point of the exercise, so
they are scored separately from the ordinary ones.

**ANSWER: keyword matching wins on the ordinary cases and every model collapses
on the edge cases.** On the 70 unambiguous emails the three models score well
apart; on the 30 edge cases they converge, because the edge cases are where the
*scorer* rather than the model decides the outcome.

**FINDING: the two scorers disagree by the maximum possible amount on an empty
prediction.** `exact_match("", "")` is **1.0** and `token_f1("", "")` is
**0.0** -- `token_f1` returns 0.0 whenever either side has no tokens, before it
compares anything. A model that correctly classifies an empty email as empty is
scored perfect by one shipped scorer and zero by the other.

**FINDING: a single-label suite cannot express a multi-label case.** The
exercise asks for "emails that could belong to multiple categories" and
`EvalCase` holds one `expected` string. A model that picks the other defensible
category scores 0.0 under `exact_match` and 0.0 under `token_f1` -- the same as
a model that answers nonsense. The suite records the ambiguity in `metadata`,
where no scorer reads it.

**FINDING: `token_f1` scores a category name by its words, so two of the five
categories overlap.** `"billing question"` against `"billing complaint"` scores
**0.500** rather than 0, because they share a token. Category labels are atoms
and this scorer treats them as bags of words, which makes partial credit
available for being wrong in the right neighbourhood.

Structure: `build_suite` assembles the 100 cases with their edge-case tags;
`rule_based`, `keyword` and `simulated_llm` are the three models the exercise
asks to compare.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "10-evaluation"
CATEGORIES = ("billing question", "billing complaint", "technical support",
              "sales enquiry", "spam")
ORDINARY, EDGE = 70, 30


def build_suite(ref):
    """100 cases: 70 unambiguous, 30 edge cases across the three families named."""
    cases = []
    for i in range(ORDINARY):
        category = CATEGORIES[i % len(CATEGORIES)]
        cases.append(ref.EvalCase(f"{category.split()[0]} enquiry number {i}", category,
                                  {"kind": "ordinary"}))
    for i in range(EDGE):
        family = ("empty", "ambiguous", "foreign")[i % 3]
        if family == "empty":
            cases.append(ref.EvalCase("", "", {"kind": "empty"}))
        elif family == "ambiguous":
            cases.append(ref.EvalCase(f"billing issue number {i}", "billing complaint",
                                      {"kind": "ambiguous", "also": "billing question"}))
        else:
            cases.append(ref.EvalCase(f"facture numero {i}", "billing question",
                                      {"kind": "foreign"}))
    return ref.EvalSuite("email", cases, {"exact": ref.exact_match, "f1": ref.token_f1})


def rule_based(text):
    if not text.strip():
        return ""
    return "spam" if "number" in text and text.startswith("spam") else "technical support"


def keyword(text):
    lowered = text.lower()
    if not lowered.strip():
        return ""
    for category in CATEGORIES:
        if category.split()[0] in lowered:
            return category
    return "sales enquiry"


def simulated_llm(text):
    """Always guesses a plausible-but-general label, the way a cautious judge would."""
    return "billing question" if "billing" in text or "facture" in text else "sales enquiry"


def by_kind(rows, cases, metric):
    """Mean score split into the ordinary cases and the edge cases."""
    ordinary = [r["scores"][metric] for r, c in zip(rows, cases)
                if c.metadata["kind"] == "ordinary"]
    edge = [r["scores"][metric] for r, c in zip(rows, cases)
            if c.metadata["kind"] != "ordinary"]
    return statistics.fmean(ordinary), statistics.fmean(edge)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    suite = build_suite(ref)
    models = {"rule_based": rule_based, "keyword": keyword, "simulated_llm": simulated_llm}
    runs = {name: suite.run(fn) for name, fn in models.items()}
    return {
        "cases": len(suite.cases),
        "edge": sum(case.metadata["kind"] != "ordinary" for case in suite.cases),
        "split": {name: {metric: by_kind(rows, suite.cases, metric)
                         for metric in ("exact", "f1")}
                  for name, rows in runs.items()},
        "empty": (ref.exact_match("", ""), ref.token_f1("", "")),
        "ambiguous": (ref.exact_match("billing question", "billing complaint"),
                      ref.token_f1("billing question", "billing complaint")),
        "nonsense": (ref.exact_match("purple", "billing complaint"),
                     ref.token_f1("purple", "billing complaint")),
    }


def verify(result):
    split = result["split"]
    exact_empty, f1_empty = result["empty"]
    exact_ambiguous, f1_ambiguous = result["ambiguous"]
    ordinary_spread = max(s["exact"][0] for s in split.values()) - \
        min(s["exact"][0] for s in split.values())
    edge_spread = max(s["exact"][1] for s in split.values()) - \
        min(s["exact"][1] for s in split.values())
    return [
        practice.Check(
            "ANSWER: the models separate on the ordinary cases and converge on the edge cases",
            ordinary_spread > edge_spread,
            f"over {result['cases']} cases, {result['edge']} of them edge cases, exact-match "
            "means split as "
            + ", ".join(f"{name} {s['exact'][0]:.3f} ordinary / {s['exact'][1]:.3f} edge"
                        for name, s in split.items())
            + f". The spread between models is {ordinary_spread:.3f} on the ordinary cases and "
            f"{edge_spread:.3f} on the edge cases: the edge cases are where the scorer rather "
            "than the model decides the outcome",
        ),
        practice.Check(
            "FINDING: the two scorers disagree by 1.0 on an empty prediction",
            exact_empty == 1.0 and f1_empty == 0.0,
            f"exact_match('', '') is {exact_empty:.1f} and token_f1('', '') is {f1_empty:.1f} -- "
            "token_f1 returns 0.0 whenever either side has no tokens, before it compares "
            "anything. A model that correctly classifies an empty email as empty is scored "
            "perfect by one shipped scorer and zero by the other, and the exercise asks for "
            "empty emails by name",
        ),
        practice.Check(
            "FINDING: a single-label suite cannot express a multi-label case",
            result["nonsense"][0] == exact_ambiguous == 0.0,
            "the exercise asks for emails that could belong to multiple categories, and "
            f"EvalCase holds one expected string. A model that picks the other defensible "
            f"category scores {exact_ambiguous:.1f} under exact_match -- the same "
            f"{result['nonsense'][0]:.1f} as a model that answers 'purple'. The suite records "
            "the second category in metadata, where no scorer reads it",
        ),
        practice.Check(
            "FINDING: token_f1 gives partial credit for being wrong in the right neighbourhood",
            f1_ambiguous > 0.4 and result["nonsense"][1] == 0.0,
            f"'billing question' against 'billing complaint' scores {f1_ambiguous:.3f} under "
            f"token_f1 and {result['nonsense'][1]:.3f} for 'purple', because the two categories "
            "share a word. Category labels are atoms and this scorer treats them as bags of "
            "words, so two of the five categories are half-right by spelling",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
