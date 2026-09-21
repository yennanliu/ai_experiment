"""Exercise 1 — one iteration verifies before it refines, and never again.

    Run the toy with max_iterations=1. Does CRITIC still help?

Reading of the exercise: `run_loop` verifies at the top of the loop body and
refines at the bottom, so a budget of *n* buys *n* verifications and *n-1*
verified refinements. At `max_iters=1` the refinement still runs -- and its
output is thrown away when the loop exits. "Does CRITIC still help" is
therefore a question about the loop's arithmetic, not about the verifier.

**ANSWER: no, and neither arm is even distinguishable.** At
`max_iters=1` both Self-Refine and CRITIC return **1** attempt, unverified,
carrying the same initial output with both factual errors intact. The
`refine` call still executes -- **1** refinement performed, **0** verified --
so the iteration is paid for and discarded.

**FINDING: CRITIC's minimum budget is 3.** Swept from 1 to 10, CRITIC first
reports `verified` at **3** iterations and never needs more. Self-Refine
never converges at any budget, including **10**.

**FINDING: the minimum is set by the verifier reporting one defect at a
time.** `verify_external` returns on its first failing check, so on an output
with **2** known-wrong facts it names **1**. Two defects therefore cost two
correcting rounds plus one confirming pass -- exactly the **3** observed.

**FINDING: at a budget of 1 the whole distinction is unobservable.** Both
arms end with byte-identical output and the same `verified=False`; only the
critique strings differ, and nothing consumes them. A single-pass guardrail
does not make a system CRITIC-shaped, it makes the choice of verifier
invisible.

Structure: `sweep()` calls the lesson's own `run_loop` at each budget;
nothing here reimplements the loop.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "05-self-refine-and-critic"
BUDGETS = (1, 2, 3, 4, 10)
TOPIC = "world facts"


def sweep(ref, use_critic):
    out = {}
    for budget in BUDGETS:
        history = ref.run_loop(TOPIC, use_critic=use_critic, max_iters=budget)
        out[budget] = {"attempts": len(history), "verified": history[-1].verified,
                       "output": history[-1].output, "critique": history[-1].critique}
    return out


def wrong_facts(ref, text):
    lowered = text.lower()
    return [fact for fact in ref.KNOWN_WRONG_FACTS
            if all(word in lowered for word in fact.split() if len(word) > 4)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    critic, plain = sweep(ref, True), sweep(ref, False)
    initial = ref.generate(TOPIC, [])
    one_critic, one_plain = critic[1], plain[1]
    refined_once = ref.refine(TOPIC, initial, one_critic["critique"],
                              [ref.Attempt(1, initial, one_critic["critique"], False)])
    return {
        "one_attempts": (one_critic["attempts"], one_plain["attempts"]),
        "one_verified": (one_critic["verified"], one_plain["verified"]),
        "one_same_output": one_critic["output"] == one_plain["output"] == initial,
        "one_same_critique": one_critic["critique"] == one_plain["critique"],
        "discarded_differs": refined_once != initial,
        "errors_at_exit": len(wrong_facts(ref, one_critic["output"])),
        "critic_first_pass": min(b for b in BUDGETS if critic[b]["verified"]),
        "plain_ever": any(plain[b]["verified"] for b in BUDGETS),
        "critic_at_ten": critic[10]["attempts"],
        "reported": ref.verify_external(initial)[0],
        "present": len(wrong_facts(ref, initial)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: no -- at a budget of 1 both arms return the same unverified output",
            all([result["one_attempts"] == (1, 1),
                 result["one_verified"] == (False, False),
                 result["one_same_output"] is True, result["errors_at_exit"] == 2,
                 result["discarded_differs"] is True]),
            f"both arms return {result['one_attempts'][0]} attempt, verified="
            f"{result['one_verified'][0]}, carrying the initial output with "
            f"{result['errors_at_exit']} known-wrong facts intact. refine still runs and "
            f"produces something different ({result['discarded_differs']}), which the "
            "loop then discards -- one refinement performed, zero verified",
        ),
        practice.Check(
            "FINDING: CRITIC's minimum budget is 3, and Self-Refine has none",
            all([result["critic_first_pass"] == 3, result["critic_at_ten"] == 3,
                 result["plain_ever"] is False]),
            f"swept from 1 to 10, CRITIC first reports verified at "
            f"{result['critic_first_pass']} iterations and still stops at "
            f"{result['critic_at_ten']} when given 10. Self-Refine converges at no budget "
            f"tested ({result['plain_ever']})",
        ),
        practice.Check(
            "FINDING: the minimum is set by one defect reported per pass",
            all([result["present"] == 2, "germany" in result["reported"],
                 "everest" not in result["reported"],
                 result["critic_first_pass"] == result["present"] + 1]),
            f"the initial output contains {result['present']} known-wrong facts and "
            f"verify_external names {result['reported']!r} -- it returns on its first "
            f"failing check. Two defects cost two correcting rounds plus one confirming "
            f"pass, which is the {result['critic_first_pass']} observed",
        ),
        practice.Check(
            "FINDING: at a budget of 1 the distinction is unobservable",
            all([result["one_same_output"] is True,
                 result["one_verified"][0] == result["one_verified"][1],
                 result["one_same_critique"] is False]),
            f"the two arms end with identical output and the same verified flag; only "
            f"the critique strings differ ({not result['one_same_critique']}), and "
            "nothing downstream reads them. A single-pass guardrail does not make a "
            "system CRITIC-shaped -- it makes the choice of verifier invisible",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
