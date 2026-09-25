"""Exercise 3 — the fan-out already exists, and throws the review away.

    Convert the sequential pipeline into a fan-out: run the researcher and a
    "requirements analyzer" agent in parallel, then merge their outputs before
    passing to the coder

Reading of the exercise: write it, then diff it against `multiAgentFanOut`,
which is the function this exercise describes and which the lesson already
ships forty lines above the exercise list. Writing it independently is how the
difference becomes visible, and there is one.

**ANSWER: four calls, 2238 tokens, and the same wall clock as the three-call
pipeline.** Researcher and requirements analyst run under one `Promise.all`,
their two messages merge into the coder's input, and the reviewer follows.
That is **2238** tokens against the pipeline's **1652** -- **586** more, or
**35%** -- for **3** sequential `setTimeout(50)` ticks either way, because the
parallel pair still has to finish before the coder starts. The fan-out buys
one more opinion and no time at all.

**FINDING: the shipped fan-out pays for a review it does not return.**
`multiAgentFanOut` makes **4** agent calls and performs **3** `messages.push`.
The reviewer's result is the one that is never pushed: it is passed
`codeResult.content` directly and then discarded. The returned `content` is
built by mapping over `messages`, so it contains researcher, requirements and
coder, and not the review -- **550** tokens are spent and counted, and the
output does not contain them.

**FINDING: the two arms are not plumbed the same way.** The pipeline routes
every hop through `messages.filter((m) => m.to === ...)` -- **2** filters for
its **3** calls. The fan-out uses **1** filter for **4** calls and hands the
reviewer a raw string. So the comparison the lesson prints is between an
architecture and a shortcut through it, not between two architectures.

**FINDING: the requirements analyst is constructed inside the call.** The
three named specialists are module constants; the fourth is a
`createSpecialist(...)` expression inside `Promise.all`, rebuilt on every
invocation. It is the only agent in the file with no name in scope, which is
also why it never appears in the comparison summary that `main` prints.

Structure: `fanout()` is the exercise; `shipped()` counts what the lesson's
own version does.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "01-why-multi-agent"
FLAT = 500
TICK = 50


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


def pipeline(specialists, task):
    """The sequential arm, for the comparison this exercise is a conversion of."""
    total, out = llm(specialists["researcher"], task)
    spent, code = llm(specialists["coder"], f"[From researcher]: {out}")
    total += spent
    spent, _ = llm(specialists["reviewer"], f"[From coder]: {code}")
    return {"tokens": total + spent, "calls": 3, "ticks": 3}


def fanout(specialists, task):
    """The exercise: researcher and requirements in parallel, merged for the coder."""
    parallel = [llm(specialists["researcher"], f"Research technical approach for: {task}"),
                llm(specialists["requirements"], f"Analyze requirements for: {task}")]
    merged = "\n".join(f"[From {name}]: {out}" for (_, out), name
                       in zip(parallel, ("researcher", "requirements")))
    spent, code = llm(specialists["coder"], merged)
    review_cost, _ = llm(specialists["reviewer"], code)
    return {"tokens": sum(cost for cost, _ in parallel) + spent + review_cost,
            "calls": 4, "ticks": 3, "review_cost": review_cost}


def shipped(src):
    """What `multiAgentFanOut` and `multiAgentPipeline` actually do, counted."""
    def body(start, end):
        index = src.index(start)
        return src[index:src.index(end, index)]

    fan = body("async function multiAgentFanOut", "async function main")
    pipe = body("async function multiAgentPipeline", "async function multiAgentFanOut")
    return {
        "fan_runs": fan.count(".run("), "fan_pushes": fan.count("messages.push"),
        "fan_filters": fan.count(".filter((m) => m.to ==="),
        "pipe_runs": pipe.count(".run("), "pipe_pushes": pipe.count("messages.push"),
        "pipe_filters": pipe.count(".filter((m) => m.to ==="),
        "returns_messages": "messages\n      .map((m)" in fan,
        "inline_specialist": fan.count("createSpecialist("),
        "named_specialists": len(re.findall(r"(?m)^const \w+ = createSpecialist\(", src)),
        "in_summary": src[src.index("=== COMPARISON ==="):].count("requirements"),
    }


def solve():
    src, _, specialists, task = lesson()
    mine, plain = fanout(specialists, task), pipeline(specialists, task)
    return {**shipped(src), "mine": mine, "plain": plain,
            "extra": mine["tokens"] - plain["tokens"], "tick_ms": TICK}


def verify(result):
    mine, plain = result["mine"], result["plain"]
    share = 100 * result["extra"] / plain["tokens"]
    return [
        practice.Check(
            "ANSWER: four calls, 2238 tokens, and the same wall clock as the pipeline",
            all([mine["tokens"] == 2238, mine["calls"] == 4, plain["tokens"] == 1652,
                 mine["ticks"] == plain["ticks"] == 3, result["extra"] == 586]),
            f"{mine['tokens']} tokens against {plain['tokens']} -- {result['extra']} "
            f"more, {share:.0f}% -- for {mine['ticks']} sequential {result['tick_ms']}ms "
            "ticks either way: the parallel pair still finishes before the coder starts",
        ),
        practice.Check(
            "FINDING: the shipped fan-out pays for a review it does not return",
            all([result["fan_runs"] == 4, result["fan_pushes"] == 3,
                 result["returns_messages"], mine["review_cost"] == 550]),
            f"multiAgentFanOut makes {result['fan_runs']} agent calls and "
            f"{result['fan_pushes']} messages.push; the reviewer's result is the one "
            f"never pushed, and content is built by mapping over messages -- so "
            f"{mine['review_cost']} tokens are spent, counted, and absent from the output",
        ),
        practice.Check(
            "FINDING: the two arms are not plumbed the same way",
            all([result["pipe_filters"] == 2, result["pipe_runs"] == 3,
                 result["fan_filters"] == 1, result["fan_runs"] == 4]),
            f"the pipeline routes every hop through messages.filter -- "
            f"{result['pipe_filters']} filters for {result['pipe_runs']} calls -- while "
            f"the fan-out uses {result['fan_filters']} for {result['fan_runs']} and "
            "hands the reviewer a raw string",
        ),
        practice.Check(
            "FINDING: the requirements analyst is constructed inside the call",
            all([result["named_specialists"] == 3, result["inline_specialist"] == 1,
                 result["in_summary"] == 1]),
            f"{result['named_specialists']} specialists are module constants and the "
            f"fourth is {result['inline_specialist']} createSpecialist expression inside "
            "Promise.all, rebuilt every invocation -- the only agent with no name in "
            "scope, and the summary mentions it once, in prose",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
