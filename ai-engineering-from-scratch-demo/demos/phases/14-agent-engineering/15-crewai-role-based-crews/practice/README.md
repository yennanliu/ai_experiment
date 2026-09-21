<!-- generated:start -->
# 14-agent-engineering / 15-crewai-role-based-crews

Solutions to all 7 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/15-crewai-role-based-crews/) · upstream spec
`phases/14-agent-engineering/15-crewai-role-based-crews/docs/en.md`

```bash
uv run demo practice run 15-crewai-role-based-crews --ex 1
uv run demo explain 15-crewai-role-based-crews --ex 1
uv run pytest demos/phases/14-agent-engineering/15-crewai-role-based-crews
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Convert the Sequential crew to a Flow. Count the touchpoints where variability drops. Note wh… | code | T1 | `ex01_the_declared_dependency_falls_back_to_the_previous_task.py` |
| 2 | Add entity memory to the crew: facts about a customer persist across kickoffs. Verify retriev… | code | T1 | `ex02_the_entity_store_exists_and_no_crew_ever_writes_to_it.py` |
| 3 | Implement a Hierarchical process where the manager refuses to route to the editor until the w… | code | T1 | `ex03_the_manager_is_handed_a_set_of_names_and_never_the_output.py` |
| 4 | Wire a `BaseTool` subclass for a (mocked) web search. Compare the trace shape vs the `@tool`… | code | T1 | `ex04_the_agent_finds_its_tool_by_searching_for_the_word_search.py` |
| 5 | Add `output_pydantic=Brief` to the editor task, where `Brief` has `title`, `summary`, `sectio… | code | T1 | `ex05_expected_output_is_a_string_that_nothing_reads.py` |
| 6 | Read CrewAI's docs intro. Port the toy to the real `crewai` API. Which guarantees did the std… | explain | T0 | prose, below |
| 7 | Wire AgentOps or Langfuse (Lesson 24) to a real run. Which traces did you miss in the stdlib… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Five ship code at T1 (the lesson's `code/main.py` imports numpy, so the
`math` group is a build input); exercises 6 and 7 read documentation and wire
a hosted tracer, so they are answered in prose.

The recurring finding is that **every contract in the toy is a string that
nothing reads**. `Task.expected_output` is `3 sources`, `3 paragraphs`,
`800 words` — and `SequentialCrew.kickoff` mentions the field **0** times.
`_writer` returns text claiming `3 paragraphs` and containing **0** paragraph
breaks. The researcher finds its tool by substring-matching `search` in a
human-readable display name. In each case the system runs, produces output,
and is wrong in a way nothing in the run can detect.

The second thread is that **the stores and the hooks exist and are
unwired**. `Memory.entity` and `write_entity` are shipped and called by
neither crew. `recall_long_term` seeds a random vector from the SHA-256 of
the whole string, so an exact match scores **1.0**, a one-word edit scores
**-0.12**, and an unrelated sentence scores **0.14** — *higher* than the
near-duplicate. It is an exact-match lookup wearing cosine's clothes.

What holds: the four primitives really are small, and the Crew/Flow contrast
really does land. Exercise 1 converts one to the other with no behaviour
change, which is the strongest thing you can say about a framework's
factoring.

### 1 — the declared dependency falls back to the previous task

**ANSWER: the same three outputs, three decisions made literal, four steps to
pay for them.** The Flow reproduces the crew's output verbatim — **3** of
**3** strings — with the per-task input decision written into each listener
and the routing written as topics. The readability cost is shape: **3**
`Task` rows become **4** decorated functions plus a terminator, and the
pipeline stops being readable as a list.

**FINDING: a declared dependency silently becomes an implicit one.**
`kickoff` computes `agent_input = joined or prior`, so a task whose declared
`context` produced nothing falls back to the *previous* task's output. A task
declaring a dependency the crew was never given runs anyway, on the wrong
input, with **0** warnings — and the output is byte-identical to the correct
run, so no test written against outputs would catch it.

**FINDING: the Flow has no step cap.** `HierarchicalCrew` carries
`max_steps=5`; `Flow.kickoff` loops `while topic in self.listeners` with
**0** counters. The shape the lesson recommends for production — "start
production with a Flow" — is the one with no bound on it.

**FINDING: the trace shapes are not comparable.** A crew returns `list[str]`
(role, text) and a flow returns `(step_name, topic, output)`. The extra field
is the topic, which is precisely the thing that made the routing literal —
so the flow is more traceable for the same reason it is less readable.

### 2 — the entity store exists and no crew ever writes to it

**ANSWER: entity writes wired in, surviving two kickoffs.** **2** customers,
**5** facts, **3** of **3** retrieved for the right one and **0** leaked from
the other, while long-term memory grows to **6** entries over the same runs.

**FINDING: the long-term embedding is a hash, so retrieval is not
similarity.** `_embed` seeds `np.random.default_rng` from the SHA-256 of the
whole string. Exact match **1.0**; one-word edit **-0.12**; unrelated
sentence **0.14**. The near-duplicate scores *below* the unrelated one. The
lesson's own description — "retrieved by similarity to the current task" — is
true of CrewAI and false of this stand-in, and the failure is invisible
because the demo only ever queries with a word that appears in nothing.

**FINDING: entity memory is the only store keyed by subject.** `short_term`
and `long_term` are keyed by *role*: **6** entries, **3** distinct roles,
**0** customer identifiers. "What do we know about this customer" has no
answer in either, which is the whole reason the entity store is a separate
type rather than a tag.

**FINDING: the two crews disagree about what to persist.**
`SequentialCrew.kickoff` writes short *and* long term; `HierarchicalCrew`
writes short term only. The same three agents doing the same three jobs leave
**6** long-term entries one way and **0** the other — so switching process
type silently changes what survives the run.

### 3 — the manager is handed a set of names, never the output

**ANSWER: a manager that sees the output, and one retry in the trace.** The
first draft has **0** paragraph breaks, so the manager routes back to
`writer`; the second has **3** and the run advances. **5** trace lines for a
**3**-role crew, `writer` twice.

**FINDING: the draft says three paragraphs and has none.** `_writer` returns
`draft (3 paragraphs) from sources: …` — the claim is *in the text* — with
**0** newlines in the string. A gate that reads the wording passes it; a gate
that counts breaks fails it. This is the lesson's "brittle handoffs" pitfall
in its purest form: the contract is prose and the prose is the output.

**FINDING: `done` is a set, so a retry cannot be expressed in it.** `kickoff`
adds each pick to a set and the shipped `_manager` returns the first role not
in it — so once `writer` is in `done` there is no value the manager can
return to run it again. The retry has to live in the manager's own state,
which means a hierarchical process cannot be made retrying without changing
the manager's *type*, not just its prompt.

**FINDING: the retry budget is the whole-run budget.** `max_steps=5` counts
manager turns, so a writer that never reaches three paragraphs consumes the
entire budget and the run ends with **0** editor output and no error — the
loop falls out of `range()`. A caller sees a shorter trace and a missing
result, with nothing saying which.

### 4 — the agent finds its tool by searching for the word "search"

**ANSWER: a `BaseTool`-shaped class with a name, a description, an args
schema and a call log.** Both forms return identical results on **3** of
**3** probes; only the class records the **4** calls and their arguments.

**FINDING: discovery is a substring match on the human-readable name.**
`_researcher` looks for `"search" in tool.tool_name.lower()`, and the
decorated tool is named `Search the web`. Rename it to `Find sources` and the
agent silently falls back to hard-coded sources with **0** tool calls and no
error. The tool's *display name* is load-bearing for routing — which is the
same failure mode as exercise 3's prose contract, one layer down.

**FINDING: the decorator has nowhere to put the schema.** `@tool` sets **2**
attributes, `tool_name` and `is_tool`. The argument schema is the Python
signature and the description is a docstring that is present and read by
nothing. The class carries **4**, which is the concrete answer to "when do I
need `BaseTool`": when anything other than the function body has to be
inspectable.

**FINDING: the trace shapes differ by where the record lives.** A crew run
returns **3** lines — one per task — and **0** mentioning the tool, so the
search is invisible in the trace while the tool's own log holds **4**
entries. Tool-level observability is a property of the *tool* here rather
than of the runtime, which is exactly backwards and is what exercise 7 is
about.

### 5 — `expected_output` is a string that nothing reads

**ANSWER: a typed `Brief`, one malformed output, one retry.** The writer
emits invalid JSON on attempt 1 and valid JSON on attempt 2; the validating
crew retries once and produces a `Brief` with **3** fields and **2**
sections, recording the failure as `invalid json` rather than raising.

**FINDING: the shipped crew passes the malformed output downstream.** With no
validation the editor receives the broken JSON verbatim and the run completes
in **3** lines whose final output still contains it and does not parse. There
is no retry to verify because there is no failure to trigger it — a pipeline
that cannot fail cannot retry, and that is the honest answer to the
exercise's last clause.

**FINDING: `expected_output` is documentation.** Three tasks, three strings —
`3 sources`, `3 paragraphs`, `800 words` — read **0** times. They also name
three different *units*: a count, a structure and a length. Even a generic
checker would have nothing uniform to check, which is why `output_pydantic`
exists as a separate field rather than as an interpretation of this one.

**FINDING: retrying is only safe because the agents are pure.**
`SequentialCrew` writes to memory after *every* task, so a retry inside the
crew would append to `short_term` and `long_term` twice: the validating
wrapper leaves **2** entries where an in-crew retry leaves **3**.
Idempotence is a precondition the shipped crew never states and the exercise
quietly assumes.

### 6 — the guarantees skipped are the ones with a schema behind them

*Cites "Four primitives".*

**The toy implements all four primitives and skips every guarantee that
depends on a declared type.** `Agent`, `Task`, `Crew`, `Process` are all
present as the lesson names them, and the ported surface would be nearly a
rename. What does not port is the set of promises CrewAI makes *about* those
fields, and each one is measured above.

**Five guarantees, in the order the exercises found them missing.**

1. **`expected_output` is a contract.** The lesson's own primitive list says
   so. In the toy it is read **0** times (exercise 5), so the "contract"
   is a comment. Real CrewAI feeds it to the model as part of the task
   prompt — which is a weaker guarantee than validation but is *not nothing*,
   and the stdlib version does not even do that.
2. **`output_pydantic` validates and retries.** Exercise 5 had to build both
   halves. This is the one guarantee whose absence changes outcomes rather
   than ergonomics: the shipped crew hands broken JSON to the next agent and
   reports success.
3. **`context` is a declared dependency.** Exercise 1: a task whose declared
   context produced nothing silently falls back to the previous task's
   output, with output identical to the correct run. Real CrewAI resolves
   context against the crew's own task list, so an orphan is a construction
   error rather than a silent substitution.
4. **Tools have a schema and an identity.** Exercise 4: discovery is a
   substring match on a display name, so renaming a tool disables it. A
   `BaseTool` has `name`, `description` and `args_schema` as separate fields
   precisely so that none of the three has to double as another.
5. **Memory is configured, not hand-wired.** Exercise 2: the entity store
   exists and neither crew writes to it, and the two crews disagree about
   whether long-term memory is written at all. `memory=True` on a real Crew
   is one flag that enables four stores consistently — the guarantee is
   *uniformity across process types*, which is exactly what the toy loses.

**Two more that no exercise could reach**, and worth naming because they are
the ones a port discovers late: **LLM configuration per agent**
(`manager_llm` versus the specialists' model, which is where the lesson's
"manager-LLM token tax" lives) and **failure semantics** — what happens when
an agent raises rather than returns. The toy has no try/except anywhere in
`kickoff`, so a raising agent takes down the crew, and there is no
guarantee to skip because there is no behaviour at all.

**What the stdlib version keeps, which is more than it sounds:** the shape.
Agents do not see each other, tasks reference agents, the crew sequences
tasks, the process picks who is next. Exercise 1's conversion of a
Sequential crew to a Flow with byte-identical output is evidence that the
factoring is real and not an artifact of the framework's plumbing.

### 7 — the missing traces are the ones between the tasks

*Cites "Where this pattern goes wrong".*

**The toy traces tasks. Everything interesting happens between them.**

A `SequentialCrew` run returns `list[str]`: **3** lines for **3** tasks,
with **1** fact each (role) plus the text. Exercise 4 measured what that
omits — the tool call inside the researcher appears **0** times, and the only
reason it is countable at all is that the `BaseTool` class keeps its own log.
A hosted tracer's value is that it makes the *runtime* the thing that
records, not the tool.

**Six spans a real tracer would show that the stdlib version does not.**

1. **Tool calls.** Exercise 4: **0** of **3** crew lines mention the tool.
   Name, arguments, result, duration, and whether the agent used the result.
   This is the single biggest hole, because it is where latency and cost
   actually go.
2. **Per-agent token and cost attribution.** There are no LLM calls to
   attribute, but the shape matters: the lesson's "prompt-bloat from
   backstories" pitfall is only visible if the trace reports prompt size per
   agent. A crew whose backstories are 2000 words each looks identical in
   this trace to one whose backstories are 20.
3. **The manager's turns.** Exercise 3: a hierarchical run with a gate takes
   **5** manager turns for **3** specialists, and the shipped trace records
   one line per *specialist* call. The "manager-LLM token tax" the lesson
   warns about is the difference between those two numbers, and this trace
   cannot express it.
4. **Retries.** Exercise 5's retry is visible only because the wrapper wrote
   a line for it. The shipped crew has no retry, so there is no span; a real
   one would show attempt 1 failing validation and attempt 2 succeeding as
   two children of one task span — and the *ratio* of those is the number
   that tells you a prompt is drifting.
5. **Memory reads and writes.** Exercise 2: a Sequential run writes **6**
   long-term entries and a Hierarchical run writes **0**, and neither shows
   up in the output at all. Retrieval hit rate — how often
   `recall_long_term` returned something that changed the output — is
   unmeasurable here, and given that exercise 2 also shows the embedding is a
   hash, it is the metric that would have caught it.
6. **Task boundaries with inputs.** The trace records outputs only, so
   exercise 1's silent context fallback is invisible: a run where task 2 read
   the wrong input looks exactly like one where it read the right one.

**And the one thing the trace does have that is worth keeping:** it is
deterministic and diffable. The lesson's "crew-as-prod" pitfall is that "on
call cannot diff a bad run against a good one", and a three-line
`list[str]` diffs perfectly. A hosted tracer adds the six spans above and
usually removes that property; the version worth shipping keeps both — a
stable textual artifact per run *and* the spans underneath it.
