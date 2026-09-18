<!-- generated:start -->
# 11-llm-engineering / 17-agent-framework-tradeoffs

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/17-agent-framework-tradeoffs/) · upstream spec
`phases/11-llm-engineering/17-agent-framework-tradeoffs/docs/en.md`

```bash
uv run demo practice run 17-agent-framework-tradeoffs --ex 1
uv run demo explain 17-agent-framework-tradeoffs --ex 1
uv run pytest demos/phases/11-llm-engineering/17-agent-framework-tradeoffs
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Take the same task — "research Anthropic's headquarters, write a 200-word brief, cite s… | code | T1 | `ex01_the_extra_node_runs_when_the_transcript_is_shortest.py` |
| 2 | Medium. Build the same task in AutoGen (researcher ↔ writer chat, editor joins via `GroupChat… | code | T0 | `ex02_the_cheapest_arm_is_the_one_that_can_do_neither.py` |
| 3 | Hard. Build a decision-tree script `pick_framework.py` that takes a short problem description… | code | T0 | `ex03_the_tree_returns_langgraph_for_ninety_three_percent_of_its_input_space.py` |
<!-- generated:end -->

## Answers

Exercise 1's LangGraph arm is the real library, so it is **T1** in the `agents`
group and CI skips it; exercises 2 and 3 are stdlib and **T0**. CrewAI, AutoGen
and Agno are not installed here, so their arms are transcriptions of the
documented shapes, labelled as such — and the LangGraph transcription in
exercise 2 is checked against the real library's measurement in exercise 1,
which is what licenses the other three.

### 1 — the extra node runs when the transcript is shortest

| | calls | input tokens | lines |
|---|---:|---:|---:|
| LangGraph (real, 4 nodes) | 4 | **337** | 9 |
| CrewAI shape (3 roles) | 3 | **283** | 7 |

Per-call prompts, recorded from inside the node functions: **29, 42, 53, 213**.

**MECHANISM: 160 of those tokens are the 200-word brief** — 47% of the
LangGraph total and 57% of the CrewAI one, and both arms carry it into their
last prompt. The cost of this task is set by the artifact, and the two numbers
the exercise asks for are the least sensitive to the choice it is asking about.

**FINDING: the extra call is nearly free.** The `plan` node runs when the
transcript is one line long — 29 tokens of 337 — so **33% more calls buys 19%
more input**. Call count and prompt size are inversely ordered along a chain,
so counting calls ranks frameworks backwards.

**MEASUREMENT: the real graph hands every node the whole accumulated list.**
The prompts increase monotonically and the last is 7.3× the first, because
`add_messages` appends and nothing prunes.

**FINDING: "cite sources" has no channel to be cited in.** The state schema has
one key, `messages`; CrewAI's context is one accumulating string. The citation
is a sentence inside a reply, and neither shape can check it against the source
it names.

### 2 — the cheapest arm is the one that can do neither

| arm | calls | input tokens | resume | human approval |
|---|---:|---:|---|---|
| Agno (1 agent + session store) | 1 | **56** | — | — |
| CrewAI (3 roles) | 3 | 283 | — | — |
| LangGraph (4 nodes) | 4 | 337 | **yes** | **yes** |
| AutoGen (`GroupChat`) | 10 | **1,579** | — | — |

**ANSWER: the three rankings disagree.** A 28.2× spread on cost, and the only
arm that can resume or pause is the second-dearest.

**MECHANISM: `GroupChat` pays an LLM call to choose the speaker.** Five content
turns become **10 calls**, every prompt carries the whole transcript, and the
160-token brief appears in **6 of the 10** — 4.7× LangGraph. The chat shape
charges for coordination that a graph gets from an edge.

**FINDING: Agno's 56 tokens are a floor, not a forecast.** Its single prompt
contains **0** of the four replies: the agent is expected to call
`search_tools` and `write_tools` itself, and a transcription charges nothing
for those round trips. The arm that looks 6× cheaper is the uncounted one.

**FINDING: two of the three axes are already decided by the lesson's tree.**
For every one of the **384** descriptors with `needs_resume` or
`needs_human_interrupt` set, `recommend` returns `langgraph` — no exceptions
across the whole enumeration.

**FINDING: the ranking inverts if the task grows.** Doubling the brief moves
AutoGen by **1,280** tokens, LangGraph and CrewAI by 160 each, and Agno by
**0**. The ranking on cost is a statement about this task's output size.

### 3 — the tree returns langgraph for 93% of its input space

7 boolean fields × 4 call counts = **512 descriptors**, all enumerated:

| answer | descriptors |
|---|---:|
| `langgraph` | **476** (93.0%) |
| `autogen` | 16 |
| `plain python` | 8 |
| `crewai` | 8 |
| `agno` | 4 |

**ANSWER: a recommender that gives the same answer to 93% of its inputs is a
default with an eight-field questionnaire in front of it.**

**FINDING: the smallest-first branch ignores two of the eight fields.** Every
descriptor reaching `plain python` agrees on five flags and on nothing else, so
the branch never reads `has_typed_state` or `needs_session_memory` — and a
two-call task with a state schema and durable memory is told to use the one
answer that cannot hold a session.

**FINDING: declaring a state schema removes two frameworks from
consideration.** `crewai` and `autogen` both require `has_typed_state == False`,
so the **256** descriptors with a schema answer only `langgraph`, `agno` or
`plain python`. A CrewAI pipeline with a typed hand-off is unreachable by
construction.

**FINDING: the fallback is one equivalence class** — reachable for exactly
**2** of the 512 descriptors (every flag false, 3 or 8 calls), out of 7 distinct
reasons. The `has_typed_state` branch one line above catches everything else.

**ANSWER: on six cases designed against those seams, the tree agrees 3 times.**

| case | tree | argued here |
|---|---|---|
| two calls, typed state, durable memory | `plain python` | `agno` |
| role pipeline + a state schema | `langgraph` | `langgraph` |
| proposer-critic over a typed scratchpad | `langgraph` | `langgraph` |
| role pipeline that must remember the user | `crewai` | `agno` |
| fanout to three retrievers inside a debate | `langgraph` | `langgraph` |
| one call that needs a human to approve it | `langgraph` | `plain python` |

It agrees wherever a state schema decides the answer. The shipped suite is 7
for 7 because each of its cases exercises one branch in isolation.
