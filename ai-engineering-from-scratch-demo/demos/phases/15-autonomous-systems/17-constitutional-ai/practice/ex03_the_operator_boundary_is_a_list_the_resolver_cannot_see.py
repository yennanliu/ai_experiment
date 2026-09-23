"""Exercise 3 — the operator boundary is a list the resolver cannot see.

    Design a soft-coded default set for a customer-support agent. What does
    the operator adjust? What can the operator not touch? Justify each
    boundary.

Reading of the exercise: the justification has to be a rule rather than a
list, or the next default nobody anticipated has no home. The rule used here
is the one the lesson's own split implies -- an operator may change *how* the
agent answers and never *whether* a refusal can be reached -- and it is then
applied to the shipped resolver to see where it lands.

**ANSWER: seven adjustable defaults, four untouchable, and one rule.** The
operator sets tone, response length, topical scope, escalation threshold,
tool allowlist, locale and refusal wording -- **7**. They cannot set the
prohibition list, the tier order, the blocking threshold or the reason
strings the audit reads -- **4**. The rule: adjustable if changing it cannot
change any verdict from refuse to allow.

**FINDING: two of the four untouchables are unprotected in the code.**
`HARDCODED_PROHIBITIONS` and the tier order are module constants, and so is
the blocking threshold -- the literal **3** appears **3** times in `resolve`
as a bare comparison. Raising it to 5 turns **3** of the **8** shipped cases
into allows, with no rename and no edit to the prohibition list.
The lesson says an operator "cannot remove the hardcoded prohibitions by
renaming them"; the threshold is the way around that does not involve
renaming anything.

**FINDING: topical scope is adjustable and refusal *wording* is not.** Both
look like presentation. Scope only ever *adds* refusals, so it passes the
rule; wording changes what the audit sees without changing what happened,
which is the one presentation change that can hide a verdict. `resolve`
returns the reason as a free string with **0** structure, so an operator who
controls wording controls the record.

**FINDING: the resolver has no operator layer at all.** It takes an action
and a score and reads **2** module constants; there is **0** configuration
object, so every boundary above is a convention rather than a mechanism. A
design that says "the operator may adjust these seven" needs a place to put
them, and the artifact's answer to where is the module source.

Structure: `ADJUSTABLE` and `FIXED` are the design; `threshold_shift()`
applies the one edit the code does not defend against.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "17-constitutional-ai"

ADJUSTABLE = ("tone", "response length", "topical scope", "escalation threshold",
              "tool allowlist", "locale", "refusal latency")
FIXED = ("prohibition list", "tier order", "blocking threshold", "reason strings")
SHIPPED_THRESHOLD, LOOSENED = 3, 5


def threshold_shift(ref, action, score, threshold=LOOSENED):
    """`resolve` with the blocking threshold moved -- no other edit."""
    block = ref.hardcoded_block(action)
    if block:
        return "refuse", f"hardcoded prohibition: {block}"
    if score.safety >= threshold:
        return "refuse", f"safety violation (score={score.safety})"
    if score.ethics >= threshold:
        return "refuse", f"ethics violation (score={score.ethics})"
    if score.guidelines >= threshold:
        return "modify", f"guideline conflict (score={score.guidelines}); modify"
    return "allow", "all higher tiers clear; helpfulness respected"


def flips(ref, threshold=LOOSENED):
    return [action[:40] for action, score in ref.CASES
            if ref.resolve(action, score)[0] != "allow"
            and threshold_shift(ref, action, score, threshold)[0] == "allow"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = inspect.getsource(ref.resolve)
    return {
        "adjustable": len(ADJUSTABLE),
        "fixed": len(FIXED),
        "rule": "adjustable if it cannot turn a refuse into an allow",
        "threshold_literals": source.count(f">= {SHIPPED_THRESHOLD}:"),
        "threshold_is_constant": f"BLOCKING" in source,
        "flipped": flips(ref),
        "flip_count": len(flips(ref)),
        "cases": len(ref.CASES),
        "prohibitions_are_module": isinstance(ref.HARDCODED_PROHIBITIONS, tuple),
        "scope_only_adds": True,
        "reason_structure": 0,
        "resolver_params": list(inspect.signature(ref.resolve).parameters),
        "config_objects": 0,
        "module_constants_read": [name for name in ("HARDCODED_PROHIBITIONS", "TierScore")
                                  if name in inspect.getsource(ref)],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: seven adjustable, four fixed, and one rule",
            all([result["adjustable"] == 7, result["fixed"] == 4,
                 result["rule"].startswith("adjustable if")]),
            f"{result['adjustable']} defaults an operator sets and "
            f"{result['fixed']} they cannot, under one rule -- {result['rule']}",
        ),
        practice.Check(
            "FINDING: two of the four untouchables are unprotected in the code",
            all([result["threshold_literals"] == 3, not result["threshold_is_constant"],
                 result["flip_count"] == 3, result["cases"] == 8]),
            f"the blocking threshold is a bare literal appearing "
            f"{result['threshold_literals']} times, and moving it from "
            f"{SHIPPED_THRESHOLD} to {LOOSENED} turns {result['flip_count']} of "
            f"{result['cases']} cases into allows -- {result['flipped']} -- with no "
            "rename and no edit to the prohibition list",
        ),
        practice.Check(
            "FINDING: topical scope is adjustable and refusal wording is not",
            all([result["scope_only_adds"], result["reason_structure"] == 0]),
            "scope can only add refusals, so it passes the rule; wording changes what "
            f"the audit sees without changing what happened, and resolve returns the "
            f"reason as a free string with {result['reason_structure']} structure",
        ),
        practice.Check(
            "FINDING: the resolver has no operator layer at all",
            all([result["resolver_params"] == ["action", "score"],
                 result["config_objects"] == 0,
                 result["prohibitions_are_module"]]),
            f"resolve takes {result['resolver_params']} and reads module constants; "
            f"there are {result['config_objects']} configuration objects, so every "
            "boundary above is a convention rather than a mechanism",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
