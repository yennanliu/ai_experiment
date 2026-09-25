"""Exercise 2 — the loop has no way to stop early because nothing can approve.

    Modify the pipeline so the reviewer can send feedback back to the coder for
    a revision loop (max 2 rounds)

Reading of the exercise: build the loop, then look for the exit it implies --
"max 2 rounds" is a cap, and a cap only matters if something else could stop
the loop sooner. In this simulator nothing can, and the reason is two lines of
type declaration.

**ANSWER: the loop is coder and reviewer alternating behind a round counter,
and it always runs to the cap.** Each round is two calls and costs exactly
**1140** tokens. Two rounds take the pipeline from **1652** to **3932** tokens
and from **3** calls to **7** -- **125%** more than the **1745** the single
agent spends, on the same task the lesson used to argue the pipeline is
cheaper.

**FINDING: nothing in the simulator can say "approved".** `LLMResponse`
declares **3** fields -- output, tokens, calls -- and none of them carries a
verdict. `fakeLLMCall` has exactly **1** output expression, the constant
template `[Response to: ${userMessage.slice(0, 80)}...]`, so the reviewer's
reply is a formatting of its own input. A revision loop needs a predicate over
the review and there is no review to predicate over, which is why "max 2
rounds" is not a safety cap here but the entire stopping rule.

**FINDING: the context cannot grow.** Every message in every round is exactly
**98** characters -- 80 characters of sliced input plus 18 of framing. Round
two's review is the same length as round zero's. The context-window pressure
this lesson exists to demonstrate is truncated out of its own simulator, which
is why the single agent's variable cost is **245** tokens against a flat
**1500**.

**FINDING: a revision is fixed price.** Because message length is constant,
each round costs the same **1140** tokens whether the code under review is one
line or a thousand. The one number a revision loop should be sensitive to --
how much there is to revise -- is the one number this model holds constant.

Structure: `llm()` is fakeLLMCall's arithmetic; `loop()` runs the revision
rounds the exercise asks for.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "01-why-multi-agent"
TEMPLATE = "[Response to: ${userMessage.slice(0, 80)}...]"
ROUNDS = 2
FLAT = 500


def lesson():
    """The lesson's only code file and the constants it declares -- read, not imported."""
    src = (parity.lesson_dir(PHASE, LESSON) / "code" / "single_vs_multi.ts").read_text("utf-8")
    start = src.index("const systemPrompt = `") + len("const systemPrompt = `")
    return (src, src[start:src.index("`;", start)],
            dict(re.findall(r'createSpecialist\(\s*"(\w+)",\s*"([^"]+)"', src)),
            re.search(r'const task = "([^"]+)"', src).group(1))


def llm(system, user):
    """fakeLLMCall's arithmetic: floor((len + len) / 4) + 500, and its output."""
    return (len(system) + len(user)) // 4 + FLAT, f"[Response to: {user[:80]}...]"


def single(prompt, task):
    """The single-agent arm, for the comparison the lesson prints."""
    context, total = [], 0
    for message in (f"Research: {task}",
                    "Given this research:\n{}\n\nNow write code for: " + task,
                    "Given all previous context:\n{}\n\nReview the code."):
        spent, out = llm(prompt, message.format("\n".join(context)))
        context.append(out)
        total += spent
    return total, 3, sum(len(entry) for entry in context)


def loop(specialists, task, rounds):
    """Researcher, coder, reviewer, then `rounds` of reviewer-to-coder revision."""
    total, out = llm(specialists["researcher"], task)
    spent, code = llm(specialists["coder"], f"[From researcher]: {out}")
    total += spent
    spent, review = llm(specialists["reviewer"], f"[From coder]: {code}")
    total, calls, sizes, milestones = total + spent, 3, [len(review)], [total + spent]
    for index in range(1, rounds + 1):
        spent, code = llm(specialists["coder"],
                          f"[From reviewer]: {review}\n[Revise round {index}]: {code}")
        total += spent
        spent, review = llm(specialists["reviewer"], f"[From coder]: {code}")
        total, calls = total + spent, calls + 2
        sizes.append(len(review))
        milestones.append(total)
    return {"total": total, "calls": calls, "sizes": sizes, "milestones": milestones}


def solve():
    src, prompt, specialists, task = lesson()
    run = loop(specialists, task, ROUNDS)
    lone, lone_calls, variable = single(prompt, task)
    steps = [b - a for a, b in zip(run["milestones"], run["milestones"][1:])]
    fields = src[src.index("type LLMResponse"):src.index("type AgentResult")]
    return {
        **run,
        "base": run["milestones"][0],
        "per_round": sorted(set(steps)),
        "single": lone, "single_calls": lone_calls,
        "single_variable": lone - FLAT * lone_calls,
        "single_flat": FLAT * lone_calls,
        "response_fields": len(re.findall(r"\w+: \w+;", fields)),
        "output_expressions": src.count(TEMPLATE),
        "verdict_fields": len(re.findall(r"(?m)^\s*(?:approved|verdict|status|accept)\b", src)),
    }


def verify(result):
    over = 100 * (result["total"] - result["single"]) / result["single"]
    return [
        practice.Check(
            "ANSWER: the loop alternates coder and reviewer, and always runs to the cap",
            all([result["per_round"] == [1140], result["base"] == 1652,
                 result["total"] == 3932, result["calls"] == 7, round(over) == 125]),
            f"each round is two calls costing {result['per_round'][0]} tokens; "
            f"{ROUNDS} rounds take the pipeline from {result['base']} to "
            f"{result['total']} tokens and {result['calls']} calls -- {over:.0f}% above "
            f"the single agent's {result['single']} on the same task",
        ),
        practice.Check(
            "FINDING: nothing in the simulator can say 'approved'",
            all([result["response_fields"] == 3, result["output_expressions"] == 1,
                 result["verdict_fields"] == 0]),
            f"LLMResponse declares {result['response_fields']} fields -- output, tokens, "
            f"calls -- and {result['verdict_fields']} carry a verdict; fakeLLMCall has "
            f"{result['output_expressions']} output expression, a constant template of "
            "its own input, so there is no review to write a stopping predicate over",
        ),
        practice.Check(
            "FINDING: the context cannot grow",
            all([set(result["sizes"]) == {98}, len(result["sizes"]) == ROUNDS + 1,
                 result["single_variable"] == 245, result["single_flat"] == 1500]),
            f"every message in every round is {result['sizes'][0]} characters -- 80 "
            f"sliced plus 18 of framing -- across all {len(result['sizes'])} reviews, so "
            f"the single agent's variable cost is {result['single_variable']} tokens "
            f"against a flat {result['single_flat']}",
        ),
        practice.Check(
            "FINDING: a revision is fixed price",
            all([len(result["per_round"]) == 1, result["per_round"] == [1140]]),
            f"all {ROUNDS} rounds cost the same {result['per_round'][0]} tokens because "
            "message length is constant -- the one quantity a revision loop should scale "
            "with, how much there is to revise, is the one this model holds fixed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
