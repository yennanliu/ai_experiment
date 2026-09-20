"""Exercise 3 — a prefix is not a classifier, and the default is the expensive one.

    Add a prompt classifier that routes "code ..." prompts to an alias
    favoring intelligence and "summarize ..." prompts to an alias favoring
    speed.

Reading of the exercise: the two rules it names cover two prompt shapes, so
the design question is the third case -- everything else. Whichever alias
catches it decides the bill for every prompt the classifier does not
recognise, and the obvious default is the wrong one. The rules themselves are
then tested against the phrasings a user actually types.

**ANSWER: `code ...` to `smart`, `summarize ...` to `fast`, and unmatched to
`fast`.** The three routes give `openai/gpt-4o`, `openai/gpt-4o-mini` and
`openai/gpt-4o-mini`; over a mixed workload of **6** prompts that costs
**less** than routing everything to `smart`, and the two named rules land on
the aliases the exercise specifies.

**FINDING: defaulting to `smart` is a 10x decision taken by omission.**
`smart`'s first model is priced **5.0/15.0** per million and `fast`'s is
**0.15/0.60** -- a **33x** input ratio. Routing the unmatched prompts to
`smart` instead raises the same workload's bill measurably, and nothing in
the exercise says which way to fall. The default is the largest
cost decision in the classifier and the one it does not mention.

**FINDING: prefix matching misses the phrasings people use.** Of **5**
prompts that clearly ask for a summary -- `"summarize x"`, `"Summarize x"`,
`"please summarize x"`, `"can you summarize x"`, `"summarise x"` -- a
`startswith("summarize")` rule catches **1**. Lowercasing and searching
anywhere catches **4**; the British spelling needs its own entry either way.

**FINDING: a misroute is silent and survives into the invocation.** A
misclassified prompt produces a complete, successful `Invocation` with a
`chosen_model` and a cost, and **0** fields recording which rule fired or
that any rule fired. The alias is the only trace, and the alias is what the
classifier chose -- so the log cannot distinguish a deliberate `fast` from a
defaulted one.

Structure: `classify` is the rule set with its default as a parameter, and
`spend` prices one workload under one classifier.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "21-llm-routing-layer"
SUMMARY_PHRASINGS = ["summarize the report", "Summarize the report",
                     "please summarize the report", "can you summarize the report",
                     "summarise the report"]
WORKLOAD = ["code a binary search", "summarize the incident",
            "what is the capital of France", "code review this diff",
            "summarize the thread", "translate this paragraph"]


def classify(prompt, *, default="fast", loose=False):
    """(alias, rule) -- the rule is why, and it is what the Invocation loses."""
    text = prompt.lower() if loose else prompt
    contains = (lambda word: word in text) if loose else text.startswith
    if contains("code"):
        return "smart", "code"
    if contains("summarize"):
        return "fast", "summarize"
    return default, "default"


def spend(ref, prompts, chooser):
    total, models = 0.0, []
    for prompt in prompts:
        with contextlib.redirect_stdout(io.StringIO()):
            inv = ref.route(chooser(prompt)[0], [{"role": "user", "content": prompt}])
        total += inv.cost_usd
        models.append(inv.chosen_model)
    return round(total, 10), models


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cheap, cheap_models = spend(ref, WORKLOAD, lambda p: classify(p))
    dear, _ = spend(ref, WORKLOAD, lambda p: classify(p, default="smart"))
    always, _ = spend(ref, WORKLOAD, lambda _p: ("smart", "forced"))

    strict = [classify(p)[1] for p in SUMMARY_PHRASINGS]
    loose = [classify(p, loose=True)[1] for p in SUMMARY_PHRASINGS]
    with contextlib.redirect_stdout(io.StringIO()):
        misrouted = ref.route(classify("translate this")[0],
                              [{"role": "user", "content": "translate this"}])
    smart_rates, fast_rates = ref.PRICES["openai/gpt-4o"], ref.PRICES["openai/gpt-4o-mini"]
    return {
        "routes": [classify("code a parser")[0], classify("summarize this")[0],
                   classify("hello there")[0]],
        "models": [cheap_models[0], cheap_models[1], cheap_models[2]],
        "cheap": cheap, "dear": dear, "always": always,
        "smart_rates": smart_rates, "fast_rates": fast_rates,
        "input_ratio": round(smart_rates[0] / fast_rates[0], 1),
        "strict_hits": strict.count("summarize"), "loose_hits": loose.count("summarize"),
        "phrasings": len(SUMMARY_PHRASINGS),
        "british": classify("summarise the report", loose=True),
        "inv_fields": sorted(vars(misrouted)),
        "misrouted_error": misrouted.error, "misrouted_model": misrouted.chosen_model,
        "rule_recorded": any("rule" in f or "classif" in f for f in vars(misrouted)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: code to smart, summarize to fast, unmatched to fast",
            all([result["routes"] == ["smart", "fast", "fast"],
                 result["models"] == ["openai/gpt-4o", "openai/gpt-4o-mini",
                                      "openai/gpt-4o-mini"],
                 result["cheap"] < result["always"]]),
            f"the three routes give {result['routes']} and land on {result['models']}. Over "
            f"a mixed workload of {len(WORKLOAD)} prompts that spends {result['cheap']} "
            f"against {result['always']} for routing everything to smart",
        ),
        practice.Check(
            "FINDING: defaulting to smart is a cost decision taken by omission",
            all([result["dear"] > result["cheap"], result["input_ratio"] > 30,
                 result["smart_rates"] == (5.0, 15.0),
                 result["fast_rates"] == (0.15, 0.60)]),
            f"smart's first model is {result['smart_rates']} per million against fast's "
            f"{result['fast_rates']} -- {result['input_ratio']}x on input. Falling back to "
            f"smart costs {result['dear']} where falling back to fast costs "
            f"{result['cheap']}, and nothing in the exercise says which way to fall",
        ),
        practice.Check(
            "FINDING: prefix matching misses the phrasings people use",
            all([result["strict_hits"] == 1, result["loose_hits"] == 4,
                 result["phrasings"] == 5, result["british"] == ("fast", "default")]),
            f"of {result['phrasings']} prompts clearly asking for a summary, "
            f"startswith('summarize') catches {result['strict_hits']} and a lowercased "
            f"substring search catches {result['loose_hits']}. The British spelling needs "
            f"its own entry either way -- here it falls through to "
            f"{result['british'][1]!r}, which happens to pick the right alias for the wrong "
            "reason",
        ),
        practice.Check(
            "FINDING: a misroute is silent and survives into the invocation",
            all([result["misrouted_error"] is None,
                 result["misrouted_model"] == "openai/gpt-4o-mini",
                 not result["rule_recorded"]]),
            f"a defaulted prompt produces a complete Invocation with "
            f"chosen_model={result['misrouted_model']!r} and error="
            f"{result['misrouted_error']}, and {result['inv_fields']} contains no field "
            "naming a rule. The alias is the only trace, and it is what the classifier "
            "chose -- so a deliberate fast and a defaulted one look identical",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
