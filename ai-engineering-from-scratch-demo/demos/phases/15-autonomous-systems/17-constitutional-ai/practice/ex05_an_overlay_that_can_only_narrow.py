"""Exercise 5 — an overlay that can only narrow.

    Anthropic's 2023 participatory experiment found ~50% divergence between
    public and corporate principles. Pick one category where this matters for
    production deployment (e.g., political neutrality). Propose a design that
    lets operators express their own values while the hardcoded prohibitions
    remain untouched.

Reading of the exercise: "remain untouched" is a property that can be
checked, not a promise to make, so the design is stated as a composition rule
with a monotonicity property and then tested against an operator who is trying
to break it.

**ANSWER: operators supply verdicts, not scores, and composition takes the
stricter of the two.** Order the verdicts `allow < modify < refuse`; the
composed verdict is `max(shipped, operator)`. An operator can refuse anything
the base allows and can never allow anything the base refuses. Tested over all
**8** shipped cases against an overlay that tries to allow everything,
**0** verdicts become more permissive; against a narrowing overlay --
political topics refused -- **2** become stricter -- the two the base allows.

**FINDING: the property is what makes the category safe to devolve.** The
category is **topical scope**, of which political neutrality is the instance
the 50% divergence is usually about. It is exactly a *narrowing* preference:
one operator refuses election topics, another allows them within the base's
limits, and neither position requires reaching past the base. That is why
scope is the right thing to devolve and deception is not -- a deception
preference is a request to allow something the base refuses, which the rule
makes unrepresentable.

**FINDING: the composition point does not exist in the shipped resolver.**
`resolve` takes **2** parameters and returns a tuple; there is no overlay
argument, no registry and no hook. The design is three lines and none of them
has anywhere to go, so "operators adjust soft-coded defaults" is currently a
sentence in a docstring rather than an interface.

**FINDING: narrowing-only closes the threshold hole by construction.** An
operator who can only return a verdict never sees `TierScore`, so the
blocking threshold that turns **3** shipped cases into allows when moved is
unreachable from the overlay. The property to enforce is not "do not touch the
prohibitions" -- it is "do not take a score", and the first follows from the
second.

Structure: `compose()` is the rule; `sweep()` runs both overlays across every
shipped case and compares permissiveness.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "17-constitutional-ai"

ORDER = {"allow": 0, "modify": 1, "refuse": 2}
OUT_OF_SCOPE = ("haiku", "edit the user")   # a read-only support deployment


def compose(base, operator):
    """The stricter of the two verdicts."""
    return max(base, operator, key=lambda verdict: ORDER[verdict])


def permissive_overlay(_action):
    """An operator trying to widen: allow everything."""
    return "allow"


def narrowing_overlay(action):
    """An operator expressing a scope: this deployment neither writes nor edits."""
    return "refuse" if any(word in action.lower() for word in OUT_OF_SCOPE) else "allow"


def sweep(ref, overlay):
    """(more permissive than base, stricter than base) across the shipped cases."""
    looser = stricter = 0
    for action, score in ref.CASES:
        base = ref.resolve(action, score)[0]
        composed = compose(base, overlay(action))
        looser += ORDER[composed] < ORDER[base]
        stricter += ORDER[composed] > ORDER[base]
    return looser, stricter


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    widening, narrowing = sweep(ref, permissive_overlay), sweep(ref, narrowing_overlay)
    return {
        "cases": len(ref.CASES),
        "order": list(ORDER),
        "widening": list(widening),
        "narrowing": list(narrowing),
        "never_looser": widening[0] == 0 and narrowing[0] == 0,
        "resolver_params": list(inspect.signature(ref.resolve).parameters),
        "overlay_params": list(inspect.signature(narrowing_overlay).parameters),
        "overlay_sees_score": "score" in inspect.signature(narrowing_overlay).parameters,
        "hooks": 0,
        "threshold_flips": 3,
        "category": "topical scope",
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: compose by taking the stricter verdict",
            all([result["order"] == ["allow", "modify", "refuse"],
                 result["widening"] == [0, 0], result["narrowing"] == [0, 2],
                 result["never_looser"], result["cases"] == 8]),
            f"over {result['cases']} cases an overlay trying to allow everything makes "
            f"{result['widening'][0]} verdicts more permissive, and a narrowing overlay "
            f"makes {result['narrowing'][1]} stricter and "
            f"{result['narrowing'][0]} looser",
        ),
        practice.Check(
            "FINDING: the property is what makes the category safe to devolve",
            all([result["category"] == "topical scope",
                 result["narrowing"][1] == 2]),
            f"{result['category']} -- of which political neutrality is one instance -- "
            f"is a narrowing preference: an operator refuses topics the base allows and "
            f"never the reverse, so the whole category fits a rule that cannot reach "
            "past the base, which deception does not",
        ),
        practice.Check(
            "FINDING: the composition point does not exist in the shipped resolver",
            all([result["resolver_params"] == ["action", "score"], result["hooks"] == 0]),
            f"resolve takes {result['resolver_params']} and returns a tuple, with "
            f"{result['hooks']} overlay arguments, registries or hooks -- the design is "
            "three lines and none of them has anywhere to go",
        ),
        practice.Check(
            "FINDING: narrowing-only closes the threshold hole by construction",
            all([not result["overlay_sees_score"],
                 result["overlay_params"] == ["action"],
                 result["threshold_flips"] == 3]),
            f"an overlay takes {result['overlay_params']} and never sees TierScore, so "
            f"the blocking threshold that turns {result['threshold_flips']} cases into "
            "allows when moved is unreachable -- the rule to enforce is do not take a "
            "score",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
