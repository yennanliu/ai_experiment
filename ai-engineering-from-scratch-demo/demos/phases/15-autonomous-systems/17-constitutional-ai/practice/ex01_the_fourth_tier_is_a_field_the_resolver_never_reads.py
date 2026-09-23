"""Exercise 1 — the fourth tier is a field the resolver never reads.

    Run `code/main.py`. Confirm the hardcoded prohibition fires even when
    helpfulness is high. Modify the resolver to weight helpfulness above
    ethics; observe the failure mode.

Reading of the exercise: the first half is confirmable and the second is not
implementable as stated, because `resolve` never reads `score.helpfulness` --
so the modification has to *add* the read before it can reweight it, and what
that exposes is the more interesting result.

**ANSWER: the prohibition fires, and helpfulness could not have stopped it.**
`resolve` calls `hardcoded_block` before touching the score at all, so the
bioweapons case refuses at `helpfulness=5` -- and would refuse at any value,
because the branch returns before the score exists. Of the tier fields,
**3** of **4** are read: `safety`, `ethics` and `guidelines`. `helpfulness`
appears in the dataclass, in the module docstring and in **0** comparisons.

**FINDING: reweighting means adding a rule, not moving one.** Putting
helpfulness above ethics requires writing the first comparison that reads it.
Done -- allow when `helpfulness >= 4` before the ethics check -- **2** of the
**8** shipped cases flip from `refuse` to `allow`: disinformation about a real
person and helping a user hide a lie from their manager. Both score ethics
**4** and helpfulness **4** or **5**, and neither is on the hardcoded list.

**FINDING: the hardcoded floor catches 1 of the 8 cases.** It is substring
matching over the action's *description*, so `provide bioweapons uplift` hits
and `explain how to make a pipe bomb` does not -- the latter is stopped by
`safety=5`, a soft tier. The floor that "never bends" covers **1** case; the
reweightable tiers cover **6**.

**FINDING: the failure mode is quiet, because the reason string still says
tier.** After the reweight, the two flipped cases return
`all higher tiers clear; helpfulness respected` -- the same sentence the haiku
gets. An audit reading verdicts and reasons sees two more allows and no
indication that the hierarchy was inverted, because nothing in the output
names the rule that fired.

Structure: `reweighted()` is the modification the exercise asks for;
`verdicts()` runs the shipped cases through either resolver.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "17-constitutional-ai"

TIERS = ("safety", "ethics", "guidelines", "helpfulness")
BLOCKING, HELPFUL = 3, 4


def reweighted(ref, action, score):
    """`resolve` with helpfulness promoted above ethics -- which needs a new read."""
    block = ref.hardcoded_block(action)
    if block:
        return "refuse", f"hardcoded prohibition: {block}"
    if score.safety >= BLOCKING:
        return "refuse", f"safety violation (score={score.safety})"
    if score.helpfulness >= HELPFUL:
        return "allow", "all higher tiers clear; helpfulness respected"
    if score.ethics >= BLOCKING:
        return "refuse", f"ethics violation (score={score.ethics})"
    if score.guidelines >= BLOCKING:
        return "modify", f"guideline conflict (score={score.guidelines}); modify"
    return "allow", "all higher tiers clear; helpfulness respected"


def verdicts(ref, resolver):
    return [resolver(action, score) for action, score in ref.CASES]


def flipped(ref):
    """(action, before, after, new reason) for every case the reweight changes."""
    shipped = verdicts(ref, ref.resolve)
    changed = verdicts(ref, lambda a, s: reweighted(ref, a, s))
    return [(action, before[0], after[0], after[1])
            for (action, _s), before, after in zip(ref.CASES, shipped, changed)
            if before[0] != after[0]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = inspect.getsource(ref.resolve)
    shipped = verdicts(ref, ref.resolve)
    changes = flipped(ref)
    after = verdicts(ref, lambda a, s: reweighted(ref, a, s))
    hardcoded = [action for action, _s in ref.CASES if ref.hardcoded_block(action)]
    return {
        "tiers": list(TIERS),
        "read": [tier for tier in TIERS if f"score.{tier}" in source],
        "block_before_score": source.index("hardcoded_block") < source.index("score."),
        "shipped": [verdict for verdict, _reason in shipped],
        "flipped": [action[:38] for action, _b, _a, _r in changes],
        "flips": len(changes),
        "cases": len(ref.CASES),
        "hardcoded_cases": hardcoded,
        "soft_refusals": sum(1 for (action, _s), (verdict, _r) in zip(ref.CASES, shipped)
                             if verdict != "allow" and not ref.hardcoded_block(action)),
        "prohibitions": len(ref.HARDCODED_PROHIBITIONS),
        "flipped_reasons": sorted({reason for _a, _b, _v, reason in changes}),
        "haiku_reason": after[0][1],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the prohibition fires, and helpfulness could not have stopped it",
            all([result["read"] == ["safety", "ethics", "guidelines"],
                 result["block_before_score"], len(result["read"]) == 3,
                 len(result["tiers"]) == 4]),
            f"resolve reads {result['read']} of {result['tiers']} and calls "
            f"hardcoded_block before touching the score, so the prohibition returns "
            "before helpfulness exists -- it is not outranked, it is unread",
        ),
        practice.Check(
            "FINDING: reweighting means adding a rule, not moving one",
            all([result["flips"] == 2, result["cases"] == 8,
                 all("disinformation" in text or "hide a lie" in text
                     for text in result["flipped"])]),
            f"promoting helpfulness above ethics needs the first comparison that reads "
            f"it, and flips {result['flips']} of {result['cases']} cases from refuse to "
            f"allow -- {result['flipped']}",
        ),
        practice.Check(
            "FINDING: the hardcoded floor catches 1 of the 8 cases",
            all([len(result["hardcoded_cases"]) == 1, result["soft_refusals"] == 5,
                 result["prohibitions"] == 6]),
            f"{len(result['hardcoded_cases'])} of {result['cases']} cases matches the "
            f"{result['prohibitions']}-entry prohibition list -- substring matching over "
            f"the description -- while {result['soft_refusals']} are stopped by "
            "reweightable tiers",
        ),
        practice.Check(
            "FINDING: the failure mode is quiet, because the reason still says tier",
            all([result["flipped_reasons"] == [result["haiku_reason"]]]),
            f"the flipped cases return {result['flipped_reasons'][0]!r} -- the same "
            "sentence the haiku gets -- so an audit sees two more allows and no sign "
            "that the hierarchy was inverted",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
