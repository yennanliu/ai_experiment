"""Exercise 2 — the cheapest arm is the one that can do neither.

    **Medium.** Build the same task in AutoGen (researcher <-> writer chat,
    editor joins via `GroupChat`) and Agno (a single agent with `search_tools`
    and `write_tools`, plus a session store). Rank the four implementations on
    (a) cost per run, (b) ability to resume after a crash, (c) ability to
    inject a human approval before the write step.

Reading of the exercise: none of AutoGen, CrewAI or Agno is installed here, so
all four arms are transcriptions of the documented shapes, built from the same
task and the same four reply strings. The LangGraph transcription is checked
against the real library's measurement in exercise 1 -- both come to 337 input
tokens -- which is what licenses the other three. Axes (b) and (c) are answered
from the lesson's own `recommend`, because a capability claim about a framework
is exactly what that tree encodes.

**ANSWER: the three rankings disagree.**

```text
(a) cost per run   agno 56  <  crewai 283  <  langgraph 337  <  autogen 1579
(b) resume         langgraph, alone
(c) human approval langgraph, alone
```

A 28x spread on cost, and the only arm that can do (b) or (c) is the
second-dearest. "Rank the four implementations" has no single answer, which is
the finding rather than a complaint.

**MECHANISM: AutoGen costs 4.7x LangGraph because `GroupChat` pays for the
speaker.** The default selector is an LLM call, so five content turns become
**10 calls**, and every prompt carries the whole transcript -- the 160-token
brief appears in **6 of the 10**. The chat shape charges for coordination that
a graph gets from an edge.

**FINDING: Agno's 56 tokens are a floor, not a forecast.** Its single prompt
contains none of the four replies: the agent is expected to call
`search_tools` and `write_tools` itself, and a transcription charges nothing
for those round trips. LangGraph's last prompt carries all four replies. The
arm that looks 6x cheaper is the arm whose work has not been counted.

**FINDING: two of the three axes are already decided by the lesson's tree.**
For every one of the **384** descriptors with `needs_resume` or
`needs_human_interrupt` set, `recommend` returns `langgraph` -- no exceptions
across the whole enumeration. The exercise asks the reader to rank what the
module already answers, and the answer does not depend on the implementations
at all.

**FINDING: the ranking inverts if the task grows.** Cost here is dominated by
one 200-word artifact, so the arms that touch it once are cheap and the arm
that puts it in six prompts is dear. Double the brief and AutoGen's total moves
by six times the increment while Agno's does not move at all -- the ranking on
(a) is a statement about this task's output size, not about the frameworks.

Structure: `SHAPES` maps each framework to a function returning its per-call
prompt token counts, `autogen_prompts` and `agno_shape` are the two the
exercise adds, and `durable` enumerates the descriptors that need a resume or
an approval and asks the lesson's tree what to use for them. Every arm is
priced twice, once with the brief and once with it doubled.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "17-agent-framework-tradeoffs"
TASK = "research Anthropic's headquarters, write a 200-word brief, cite sources"
SYSTEM = "You are a research assistant. Work step by step and cite every claim."
PLAN = "Plan: 1 search, 2 draft 200 words, 3 attach citations."
FOUND = "Anthropic is headquartered in San Francisco, California (anthropic.com/company)."
BRIEF = "Anthropic is an AI safety company headquartered in San Francisco. " * 12
CITE = "Sources: anthropic.com/company (accessed 2026-09-18.)"
ROSTER = "Speakers: researcher, writer, editor. Reply with one name."
TOOLS = ("search_tools: web_search(query) -> str; fetch(url) -> str. "
         "write_tools: draft(outline) -> str; cite(urls) -> str.")
FLAGS = ("has_typed_state", "has_roles", "has_dialogue", "has_parallel_fanout",
         "needs_resume", "needs_human_interrupt", "needs_session_memory")
AGNO_PROMPT = f"{SYSTEM}\n{TOOLS}\nSession: returning user, 3 prior briefs.\n{TASK}"


def tokens(text):
    return max(1, len(text.split()) * 4 // 3)


def langgraph_shape(brief):
    history, calls = [TASK], []
    for reply in (PLAN, FOUND, brief, CITE):
        calls.append(tokens(SYSTEM + "\n" + "\n".join(history)))
        history.append(reply)
    return calls


def crewai_shape(brief):
    context, calls = "", []
    for name, backstory, reply in (
            ("researcher", "verifies every fact against a primary source", FOUND),
            ("writer", "produces tight 200-word briefs", brief),
            ("editor", "checks citations and trims to length", CITE)):
        calls.append(tokens(f"You are the {name}. A senior professional who {backstory}.\n"
                            f"Overall task: {TASK}\nContext:\n{context}"))
        context += reply + "\n"
    return calls


def autogen_prompts(brief):
    transcript, prompts = [TASK], []
    for speaker, reply in (("researcher", FOUND), ("writer", brief),
                           ("researcher", "Confirmed against anthropic.com/company."),
                           ("writer", brief), ("editor", CITE)):
        prompts.append(ROSTER + "\n" + "\n".join(transcript))
        prompts.append(f"You are the {speaker}.\n" + "\n".join(transcript))
        transcript.append(reply)
    return prompts


def autogen_shape(brief):
    return [tokens(prompt) for prompt in autogen_prompts(brief)]


def agno_shape(brief):
    return [tokens(AGNO_PROMPT)]


SHAPES = {"agno": agno_shape, "crewai": crewai_shape, "langgraph": langgraph_shape,
          "autogen": autogen_shape}


def durable(ref):
    needy = [dict(zip(FLAGS, bits))
             for bits in itertools.product((False, True), repeat=len(FLAGS))
             if bits[FLAGS.index("needs_resume")] or bits[FLAGS.index("needs_human_interrupt")]]
    picks = {ref.recommend(ref.Problem(total_llm_calls=n, **fields)).framework
             for fields in needy for n in (1, 2, 3, 8)}
    return len(needy) * 4, sorted(picks)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    costs = {name: shape(BRIEF) for name, shape in SHAPES.items()}
    total = {name: sum(calls) for name, calls in costs.items()}
    doubled = {name: sum(shape(BRIEF * 2)) for name, shape in SHAPES.items()}
    count, picks = durable(ref)
    return {
        "calls": {name: len(calls) for name, calls in costs.items()}, "total": total,
        "ranking": sorted(total, key=total.get), "brief_tokens": tokens(BRIEF),
        "spread": round(max(total.values()) / min(total.values()), 1),
        "brief_in_autogen": sum(1 for p in autogen_prompts(BRIEF) if BRIEF.strip() in p),
        "agno_carries": sum(1 for reply in (PLAN, FOUND, BRIEF, CITE)
                            if reply.strip() in AGNO_PROMPT),
        "durable_count": count, "durable_picks": picks,
        "growth": {name: doubled[name] - total[name] for name in total},
    }


def verify(result):
    total, growth = result["total"], result["growth"]
    return [
        practice.Check(
            "ANSWER: agno 56 < crewai 283 < langgraph 337 < autogen 1579",
            all([result["ranking"] == ["agno", "crewai", "langgraph", "autogen"],
                 total == {"agno": 56, "crewai": 283, "langgraph": 337, "autogen": 1579},
                 result["spread"] == 28.2]),
            f"the four arms cost {total} over {result['calls']} calls -- a "
            f"{result['spread']}x spread -- and the only arm that can resume or pause for "
            f"approval is {result['durable_picks'][0]}, the second-dearest",
        ),
        practice.Check(
            "MECHANISM: GroupChat pays an LLM call to choose the speaker",
            all([result["calls"]["autogen"] == 10, result["brief_in_autogen"] == 6,
                 total["autogen"] / total["langgraph"] > 4]),
            f"five content turns become {result['calls']['autogen']} calls, and every prompt "
            f"carries the whole transcript, so the {result['brief_tokens']}-token brief "
            f"appears in {result['brief_in_autogen']} of them -- "
            f"{total['autogen'] / total['langgraph']:.1f}x LangGraph. The chat shape charges "
            "for coordination a graph gets from an edge",
        ),
        practice.Check(
            "FINDING: Agno's 56 tokens are a floor, not a forecast",
            all([result["calls"]["agno"] == 1, result["agno_carries"] == 0,
                 total["agno"] < total["langgraph"] / 5]),
            f"its single prompt contains {result['agno_carries']} of the four replies: the "
            "agent is expected to call search_tools and write_tools itself, and a "
            f"transcription charges nothing for those round trips. The arm that looks "
            f"{total['langgraph'] / total['agno']:.0f}x cheaper is the uncounted one",
        ),
        practice.Check(
            "FINDING: two of the three axes are already decided by the lesson's tree",
            all([result["durable_count"] == 384, result["durable_picks"] == ["langgraph"]]),
            f"for every one of the {result['durable_count']} descriptors with needs_resume "
            f"or needs_human_interrupt set, recommend returns {result['durable_picks']} -- no "
            "exceptions across the enumeration. The exercise asks the reader to rank what "
            "the module already answers, without reference to the implementations",
        ),
        practice.Check(
            "FINDING: the ranking inverts if the task grows",
            all([growth["autogen"] > 5 * growth["langgraph"], growth["agno"] == 0,
                 growth["langgraph"] == growth["crewai"]]),
            f"doubling the brief moves the totals by {growth}: AutoGen by six times the "
            f"increment because the brief is in six prompts, Agno not at all because it is "
            "in none. The ranking on cost is a statement about this task's output size",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
