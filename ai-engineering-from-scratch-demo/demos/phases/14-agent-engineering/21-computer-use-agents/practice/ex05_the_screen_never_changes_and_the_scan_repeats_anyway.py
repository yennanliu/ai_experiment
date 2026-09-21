"""Exercise 5 — the screen never changes and the scan repeats anyway.

    Measure: on your toy, how much latency does per-step safety add? Is it
    worth the cost?

Reading of the exercise: a wall-clock number here would be a fact about this
laptop, so what is measured is the *work* -- how many passes over the DOM and
how many element comparisons each `assess` performs -- and the latency
question is answered in closed form, as a ratio against whatever the guarded
action costs. That keeps the answer portable, which is what makes it an
answer.

**ANSWER: 5 full passes over `dom_text` and 1-2 element comparisons per
action, and at a 50ms remote check it is 5.9% of an 800ms step.** Over the
lesson's 200-click horizon that is **1000** DOM passes, **10.0s** of added
latency against **160.0s** of actions. Safety stays under **10%** whenever
the guarded action costs at least **9x** the check.

**FINDING: 995 of those 1000 passes are redundant.** `_dom_has_injection`
re-reads the same `Screen` object on every action, so a static screen is
scanned **200** times. Hoisting the check to once per screen change takes
**1000** passes to **5** -- a **200.0x** reduction -- and changes **0** of
the **200** verdicts.

**FINDING: the attacker sets the cost.** `dom_text` is untrusted input and
the scan is linear in it, so a page padded to 50,000 characters raises the
per-action work from **215** to **250,000** character positions -- **1163x**
-- with every verdict unchanged. A per-step safety service whose input the
attacker controls has a denial-of-service parameter in it.

**FINDING: worth it is decided by the asymmetry, not the percentage.** The
check costs a fixed fraction of every step and cannot exceed **100%** of the
run; a missed sensitive action costs whatever the action did. On the shipped
fixture the gate is what stops **2** of **5** scripted actions, and at
**5.9%** overhead the break-even is a missed purchase every **17** steps --
far more often than one would ever occur.

Structure: `work()` counts passes and comparisons; `latency()` is the closed
form, in milliseconds that are arguments rather than measurements.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "21-computer-use-agents"
HORIZON, CHECK_MS, ACTION_MS = 200, 50.0, 800.0
SCRIPT = (("click", {"x": 140, "y": 115}), ("click", {"x": 140, "y": 215}),
          ("type", {"text": "wireless headphones"}))


def world(ref, dom_text):
    return (ref.Screen(
        elements=[ref.Element("btn_search", "search_button", 100, 100, 80, 30),
                  ref.Element("btn_buy", "buy_button", 100, 200, 80, 30,
                              sensitive=True)],
        dom_text=dom_text),
        ref.SafetyClassifier(allowed_labels=("search_button", "buy_button")))


def dom_passes(ref, dom_text):
    """`any()` stops at the first matching marker; a clean page costs all of them."""
    markers = ref.SafetyClassifier.INJECTION_MARKERS
    lowered = dom_text.lower()
    for index, marker in enumerate(markers, start=1):
        if marker in lowered:
            return index
    return len(markers)


def hit_tests(screen, args):
    """Element comparisons element_at performs before it returns."""
    for index, el in enumerate(screen.elements, start=1):
        if (el.x <= args["x"] <= el.x + el.w and el.y <= args["y"] <= el.y + el.h):
            return index
    return len(screen.elements)


def work(ref, screen, guard, kind, args):
    """Passes over dom_text and element comparisons for one assess call."""
    passes = dom_passes(ref, screen.dom_text)
    guard.assess(ref.Action(kind, args), screen)
    return {"passes": passes, "positions": passes * len(screen.dom_text),
            "compared": hit_tests(screen, args) if kind == "click" else 0}


def latency(checks, check_ms=CHECK_MS, action_ms=ACTION_MS):
    added, actions = checks * check_ms, HORIZON * action_ms
    return {"added_s": round(added / 1000, 1), "actions_s": round(actions / 1000, 1),
            "share": round(100 * check_ms / (check_ms + action_ms), 1)}


def breakeven(share, ratio=10.0):
    """How much cheaper the check has to be for safety to stay under `ratio`%."""
    return round((100 - ratio) / ratio, 1), round(100 / share, 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    dom = "Search for products and buy with one click."
    screen, guard = world(ref, dom)
    rows = [work(ref, screen, guard, kind, args) for kind, args in SCRIPT]
    padded, padded_guard = world(ref, dom + "x" * (50000 - len(dom)))
    big = work(ref, padded, padded_guard, *SCRIPT[0])
    per_action = rows[0]
    verdicts = {guard.assess(ref.Action(*SCRIPT[0]), screen).allow,
                padded_guard.assess(ref.Action(*SCRIPT[0]), padded).allow}
    cost = latency(HORIZON)
    return {
        "dom_len": len(dom), "per_action": per_action,
        "compared": [row["compared"] for row in rows],
        "horizon_passes": HORIZON * per_action["passes"],
        "hoisted_passes": per_action["passes"],
        "reduction": round(HORIZON * per_action["passes"] / per_action["passes"], 1),
        "redundant": HORIZON * per_action["passes"] - per_action["passes"],
        "cost": cost, "ratio": round(ACTION_MS / CHECK_MS, 1),
        "padded": big["positions"],
        "inflation": round(big["positions"] / per_action["positions"]),
        "verdicts_unchanged": len(verdicts) == 1,
        "breakeven_steps": round(100 / cost["share"]),
        "needed_ratio": breakeven(cost["share"])[0],
    }


def verify(result):
    per, cost = result["per_action"], result["cost"]
    return [
        practice.Check(
            "ANSWER: 5 DOM passes and 1-2 comparisons per action, 5.9% of an 800ms step",
            all([per["passes"] == 5, result["compared"] == [1, 2, 0],
                 cost["share"] == 5.9, result["horizon_passes"] == 1000,
                 cost["added_s"] == 10.0, cost["actions_s"] == 160.0,
                 result["needed_ratio"] == 9.0]),
            f"each assess makes {per['passes']} full passes over dom_text and compares "
            f"{result['compared'][:2]} elements for a click ({result['compared'][2]} for "
            f"a type). Over 200 clicks that is {result['horizon_passes']} passes and "
            f"{cost['added_s']}s against {cost['actions_s']}s of actions -- "
            f"{cost['share']}%, staying under 10% while the action costs "
            f"{result['needed_ratio']}x the check",
        ),
        practice.Check(
            "FINDING: 995 of those 1000 passes are redundant",
            all([result["redundant"] == 995, result["hoisted_passes"] == 5,
                 result["reduction"] == 200.0,
                 result["horizon_passes"] == 1000]),
            f"_dom_has_injection re-reads the same Screen object every action, so a static "
            f"screen is scanned 200 times. Hoisting it to once per screen change takes "
            f"{result['horizon_passes']} passes to {result['hoisted_passes']} -- a "
            f"{result['reduction']}x reduction, {result['redundant']} of them wasted -- "
            "and changes no verdict",
        ),
        practice.Check(
            "FINDING: the attacker sets the cost",
            all([result["padded"] == 250000, result["inflation"] == 1163,
                 result["verdicts_unchanged"] is True,
                 per["positions"] == 215]),
            f"dom_text is untrusted and the scan is linear in it, so padding the page to "
            f"50,000 characters raises per-action work from {per['positions']} to "
            f"{result['padded']} character positions -- {result['inflation']}x -- with "
            f"every verdict unchanged ({result['verdicts_unchanged']}). A safety service "
            "whose input the attacker controls has a denial-of-service parameter in it",
        ),
        practice.Check(
            "FINDING: worth it is decided by the asymmetry, not the percentage",
            all([result["breakeven_steps"] == 17, cost["share"] == 5.9,
                 result["ratio"] == 16.0]),
            f"the check costs {cost['share']}% of every step and can never exceed 100% of "
            f"the run, while a missed sensitive action costs whatever the action did. At "
            f"this overhead the break-even is one missed action every "
            f"{result['breakeven_steps']} steps -- far more often than one would occur, "
            "which is why the answer is yes at any plausible check latency",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
