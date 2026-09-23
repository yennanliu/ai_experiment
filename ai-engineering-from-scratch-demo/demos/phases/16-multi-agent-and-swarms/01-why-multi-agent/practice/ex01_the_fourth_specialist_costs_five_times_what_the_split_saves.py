"""Exercise 1 — the fourth specialist costs five times what the split saves.

    Add a fourth specialist: a "tester" agent that receives code from the coder
    and review feedback from the reviewer, then writes tests

Reading of the exercise: add it, then price it against the comparison the
lesson prints -- because the lesson has no Python to import, its token model is
eight characters of arithmetic, and once that arithmetic is written down the
exercise turns out to reverse the lesson's own conclusion.

**ANSWER: the tester is a fourth `createSpecialist` fed the two messages
addressed to it, and it makes multi-agent lose.** With the tester the pipeline
spends **2238** tokens against the single agent's **1745** -- **493** more, or
**28%** worse. Without it the pipeline spends **1652**, which is **93** tokens
better. The whole advantage the lesson demonstrates is smaller than one fifth
of the cost of the agent the exercise asks for.

**FINDING: 86% of the compared number is a constant.** `fakeLLMCall` returns
`floor((systemPrompt.length + userMessage.length) / 4) + 500`. Across the
single agent's three calls that flat **500** contributes **1500** of **1745**
tokens; across the pipeline's three it contributes **1500** of **1652**. The
context pollution the lesson dramatises is the remaining **245** against
**152** -- a **93**-token difference inside numbers near 1700, which is why
adding any fourth call swamps it.

**FINDING: the single agent is told to do four things and does three.** Its
system prompt numbers **4** steps and the fourth is "Write tests". The
function contains **3** `await fakeLLMCall` lines: research, code, review. So
the tester this exercise adds to the multi-agent side is a step the
single-agent side was already charged for in its prompt and never ran.

**FINDING: the third printed metric is a random number.** `calls` is
`Math.floor(Math.random() * 5) + 1`, uniform on 1..5 and independent of the
task, the prompt and the architecture. "Tool calls" therefore compares **3**
draws against **3** -- and against **4** once the tester lands, so the metric
moves with the agent count and with nothing else.

Structure: `lesson()` reads the prompts out of the TypeScript and `llm()` is
its token arithmetic; `solve()` prices the three pipelines.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "01-why-multi-agent"
TESTER = ("You are a test engineer. Given implementation code and review "
          "feedback, write the tests that cover both. Nothing else.")
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
    """Three calls into one growing context window."""
    context, total = [], 0
    for message in (f"Research: {task}",
                    "Given this research:\n{}\n\nNow write code for: " + task,
                    "Given all previous context:\n{}\n\nReview the code."):
        spent, out = llm(prompt, message.format("\n".join(context)))
        context.append(out)
        total += spent
    return total, 3


def pipeline(specialists, task, tester=False):
    """Researcher, coder, reviewer -- and optionally the tester this exercise adds."""
    total, out = llm(specialists["researcher"], task)
    cost, code = llm(specialists["coder"], f"[From researcher]: {out}")
    total += cost
    cost, review = llm(specialists["reviewer"], f"[From coder]: {code}")
    total += cost
    if not tester:
        return total, 3
    cost, _ = llm(TESTER, f"[From coder]: {code}\n[From reviewer]: {review}")
    return total + cost, 4


def solve():
    """Every pipeline the exercise compares, priced with the lesson's own model."""
    src, prompt, specialists, task = lesson()
    start = src.index("async function singleAgentApproach")
    body = src[start:src.index("function createSpecialist", start)]
    lone, lone_calls = single(prompt, task)
    plain, plain_calls = pipeline(specialists, task)
    tested, tested_calls = pipeline(specialists, task, tester=True)
    return {
        "single": lone, "single_calls": lone_calls, "pipeline": plain,
        "pipeline_calls": plain_calls, "tested": tested, "tested_calls": tested_calls,
        "numbered_steps": len(re.findall(r"(?m)^\d\.", prompt)),
        "single_llm_calls": body.count("await fakeLLMCall"),
        "random_calls": bool(re.search(r"calls: Math\.floor\(Math\.random\(\) \* 5\) \+ 1", src)),
        "single_flat": FLAT * lone_calls, "pipeline_flat": FLAT * plain_calls,
        "saved": lone - plain, "added": tested - lone,
    }


def verify(result):
    share = 100 * result["single_flat"] / result["single"]
    return [
        practice.Check(
            "ANSWER: the tester is a fourth specialist, and it makes multi-agent lose",
            all([result["tested"] == 2238, result["tested_calls"] == 4,
                 result["added"] == 493, result["saved"] == 93]),
            f"with the tester the pipeline spends {result['tested']} tokens against the "
            f"single agent's {result['single']} -- {result['added']} more, "
            f"{100 * result['added'] / result['single']:.0f}% worse -- where without it it "
            f"was {result['saved']} ahead",
        ),
        practice.Check(
            "FINDING: 86% of the compared number is a constant",
            all([result["single_flat"] == 1500, result["pipeline_flat"] == 1500,
                 round(share) == 86, result["saved"] == 93]),
            f"the flat +{FLAT} contributes {result['single_flat']} of {result['single']} "
            f"tokens ({share:.0f}%) and {result['pipeline_flat']} of "
            f"{result['pipeline']}; the context pollution is the remaining "
            f"{result['single'] - result['single_flat']} against "
            f"{result['pipeline'] - result['pipeline_flat']}",
        ),
        practice.Check(
            "FINDING: the single agent is told to do four things and does three",
            all([result["numbered_steps"] == 4, result["single_llm_calls"] == 3]),
            f"the system prompt numbers {result['numbered_steps']} steps, the fourth "
            f"'Write tests', and the function makes {result['single_llm_calls']} "
            "fakeLLMCall invocations -- the step this exercise adds as an agent was "
            "charged for in the single arm's prompt and never run",
        ),
        practice.Check(
            "FINDING: the third printed metric is a random number",
            all([result["random_calls"], result["pipeline_calls"] == 3,
                 result["tested_calls"] == 4]),
            "calls is Math.floor(Math.random() * 5) + 1, uniform on 1..5 and independent "
            f"of task, prompt and architecture, so Tool calls compares "
            f"{result['pipeline_calls']} draws against {result['single_calls']}, and "
            f"{result['tested_calls']} once the tester lands",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
