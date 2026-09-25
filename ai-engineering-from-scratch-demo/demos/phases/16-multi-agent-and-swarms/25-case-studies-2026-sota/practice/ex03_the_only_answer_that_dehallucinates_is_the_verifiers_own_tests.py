"""Exercise 3 — the only answer that dehallucinates is the verifier's own tests.

    Read ChatDev (arXiv:2307.07924). Identify the mechanism of "communicative
    dehallucination." Implement it in one of your existing multi-agent
    systems.

Reading of the exercise: "one of your existing multi-agent systems" is Lesson
08's planner / executor / critic / verifier pipeline, imported, not copied;
dehallucination is added between executor and planner, and every artifact is
still judged by the lesson's own critic and verifier.

**ANSWER: role reversal -- the assistant asks the instructor for the missing
detail before it answers.** ChatDev §3.2: the assistant is "proactively
seeking more specific information ... before delivering a conclusive
response", turning instruct -> respond into instruct -> (ask -> answer)* ->
respond; §4 activates it during code completion, review and testing. In the
Lesson 08 pipeline the executor does not know which operation the wish
means. Guessing through + * - max, starting from the lesson's own buggy
`a * b`, takes **4** artifacts and **3** rejections -- 7 messages. Asking the
planner first takes a question, an answer and **1** artifact -- 3 messages.
The critic approves all 4 guesses, so only the verifier can drive the loop.

**FINDING: the only answer that dehallucinates is the verifier's test data.**
What can the planner say? The signature `add_two(a: int, b: int) -> int`
admits all 4 candidates. The description is the user's wish, verbatim. Only a
test case, ((1, 2), 3), leaves 1 of 4 -- and an executor holding the tests can
return a lookup table, which the reference verifier passes. The detail that
removes the guess is the one that removes the verifier's independence.

**FINDING: dehallucination faithfully relays a wrong spec.** `planner` returns
the same signature and tests for every wish. Ask it about "a function that
returns the product of two integers" and it answers with ((1, 2), 3); the
dehallucinating executor ships `a + b`, which the critic approves and the
verifier passes. The question stops at the planner, and the planner never
read the wish -- in ChatDev terms, the instructor is itself hallucinating.

Structure: `guessing()` is the loop without the mechanism, `asking()` the loop
with it; `ask_planner()` is the instructor's side of the role reversal.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "25-case-studies-2026-sota"
PIPELINE = "08-role-specialization"
CANDIDATES = ["a * b", "a - b", "max(a, b)", "a + b"]  # starts at executor_buggy's guess
SUM, PRODUCT = ("A function that returns the sum of two integers.",
                "A function that returns the product of two integers.")


def artifact(ref, op):
    return ref.Artifact(code=f"def add_two(a, b):\n    return {op}\n")


def judged(ref, spec, op):
    art = artifact(ref, op)
    return ref.critic(spec, art).approved, ref.verifier(spec, art).passed


def guessing(ref, spec):
    """Without the mechanism: produce, get rejected, guess again."""
    messages, verdicts = 0, []
    for op in CANDIDATES:
        verdicts.append(judged(ref, spec, op))
        messages += 1
        if verdicts[-1][1]:
            return {"artifacts": len(verdicts), "messages": messages, "verdicts": verdicts}
        messages += 1  # the rejection sent back
    return {"artifacts": len(verdicts), "messages": messages, "verdicts": verdicts}


def ask_planner(spec, detail):
    """The instructor's side of the role reversal: answer from the spec it holds."""
    return {"signature": spec.signature, "description": spec.description,
            "example": spec.tests[0]}[detail]


def run(op, a, b):
    return eval(op, {"max": max}, {"a": a, "b": b})  # noqa: S307 - fixed candidate strings


def consistent(example):
    (a, b), want = example
    return [op for op in CANDIDATES if run(op, a, b) == want]


def asking(ref, spec):
    """With the mechanism: one question, one answer, then produce from what it says."""
    options = consistent(ask_planner(spec, "example"))
    return {"op": options[0], "messages": 2 + 1, "options": len(options),
            "verdict": judged(ref, spec, options[0])}


def solve():
    ref = parity.load_reference(PHASE, PIPELINE, "main")
    spec = ref.planner(SUM)
    lookup = ref.Artifact(code="def add_two(a, b):\n    return {(1, 2): 3, (10, 20): 30, "
                               "(-5, 5): 0}.get((a, b), a * b)\n")
    product = ref.planner(PRODUCT)
    return {
        "guess": guessing(ref, spec), "ask": asking(ref, spec),
        "signature": ask_planner(spec, "signature"),
        "by_signature": sum(isinstance(run(op, 3, 4), int) for op in CANDIDATES),
        "by_example": len(consistent(spec.tests[0])),
        "description_is_wish": ask_planner(spec, "description") == SUM,
        "lookup_passes": ref.verifier(spec, lookup).passed,
        "same_tests": product.tests == spec.tests,
        "product_ship": asking(ref, product),
    }


def verify(result):
    guess, ask, product = result["guess"], result["ask"], result["product_ship"]
    return [
        practice.Check(
            "ANSWER: role reversal -- ask the instructor before answering",
            all([guess["artifacts"] == 4, guess["messages"] == 7, ask["messages"] == 3,
                 ask["verdict"] == (True, True),
                 all(approved for approved, _ in guess["verdicts"])]),
            f"guessing takes {guess['artifacts']} artifacts and {guess['messages']} "
            f"messages; asking the planner takes {ask['messages']} and ships "
            f"{ask['op']!r}; the critic approved all {guess['artifacts']} guesses",
        ),
        practice.Check(
            "FINDING: the only answer that dehallucinates is the verifier's test data",
            all([result["by_signature"] == 4, result["by_example"] == 1,
                 result["description_is_wish"], result["lookup_passes"]]),
            f"{result['signature']} admits {result['by_signature']} candidates, the "
            f"description is the wish verbatim, one test case leaves "
            f"{result['by_example']} -- and a lookup table over the tests passes the "
            "reference verifier",
        ),
        practice.Check(
            "FINDING: dehallucination faithfully relays a wrong spec",
            result["same_tests"] and product["op"] == "a + b"
            and product["verdict"] == (True, True),
            f"asked about the product wish the planner answers with the sum's tests, and "
            f"the executor ships {product['op']!r} -- approved and passed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
