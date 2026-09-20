<!-- generated:start -->
# 14-agent-engineering / 01-the-agent-loop

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/) · upstream spec
`phases/14-agent-engineering/01-the-agent-loop/docs/en.md`

```bash
uv run demo practice run 01-the-agent-loop --ex 1
uv run demo explain 01-the-agent-loop --ex 1
uv run pytest demos/phases/14-agent-engineering/01-the-agent-loop
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `max_tool_calls_per_turn` cap. What breaks if the model issues three calls but you only… | code | T0 | `ex01_the_cap_has_nowhere_to_attach_and_the_drop_leaves_no_trace.py` |
| 2 | Implement a `no_tool_calls → done` stop path. Contrast with `finish` as an explicit tool. Whi… | code | T0 | `ex02_absence_is_the_default_state_of_a_broken_parse.py` |
| 3 | Extend `ToyLLM` so it sometimes returns an `Action` with a malformed argument dict. Make the… | code | T0 | `ex03_the_loop_already_recovers_and_the_policy_cannot.py` |
| 4 | Replace `ToyLLM` with a real Responses API call. Move the thought trace from inline strings t… | explain | T0 | prose, below |
| 5 | Add a `tool_use_id` correlator like the Anthropic schema so parallel tool calls can return ou… | code | T0 | `ex05_positional_pairing_is_right_until_it_is_silently_wrong.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 4 needs a live provider
call, so it is answered in prose against the lesson's own text.

The recurring finding is that **`code/main.py` makes one call per turn, and
four of the five exercises are about what that identity hides**. A turn is a
call, so the turn budget is a call budget, position is a correct correlator,
an observation always belongs to the call above it, and a cap on calls per
turn has nothing to attach to. Every one of those stops being true at the
first turn that carries two calls — which is every turn of every 2026
provider schema.

The parts that hold are the ones the lesson named as ingredients.
`ToolRegistry.dispatch` really does turn every failure into an observation
string rather than a crash, and `AgentLoop` really does bound the run. What
neither of them has is a *type* for the difference between outcomes: one
`kind='final'` for three terminations, one `error:` prefix for two layers,
one `except TypeError` for two different faults.

### 1 — the cap has nowhere to attach, and the drop leaves no trace

**ANSWER: `max_tool_calls_per_turn` belongs in the flattening adapter.** The
shipped loop reads one `action` per reply, so three-calls-in-a-turn is not a
state it can hold — `ToolCall` has **2** fields, `Turn` has **4**, and the
reply's `action` is a `str`. At `cap=2` over a first turn of **3** calls,
**4** of **5** calls dispatch.

**FINDING: the dropped call leaves nothing behind.** The history is written
by the loop, and the loop never saw the third call — **0** mentions of the
dropped `kv_set` anywhere in the transcript. A reader cannot tell a capped
turn from a two-call turn.

**FINDING: the drop is read back as data.** `kv_get(key='tax')` returns
`missing:tax`, and `dispatch` marks failures only by an `error:` prefix. **0**
of the **4** observations look like errors, while **1** of them is wrong, and
the agent's next thought is conditioned on it.

**FINDING: `max_turns` stops being a tool-call budget.** With one call per
turn it bounded calls at **10**; with a cap of *c* it bounds them at *10c* and
starts truncating turns mid-flight — uncapped at `max_turns=4` the run
dispatches **4** of **5** and ends `budget exhausted`. The exercise's
breakage arrives from the budget instead of the cap.

### 2 — absence is the default state of a broken parse

**ANSWER: both stop paths, over the same six replies.** One reply is empty.
`no_tool_calls → done` reads it as a decision and stops having dispatched
**3** calls, returning `''`. `finish`-as-a-tool reads it as a malformed turn,
dispatches `error: unknown tool ''`, and carries on to dispatch **6** and
return `138.0`.

**Which is safer: `finish` as an explicit tool.** Every mechanical failure
that produces no parsed call — a truncated stream, an unrecognised schema, a
refusal, a network cut mid-block — looks exactly like a deliberate stop under
the first design. Under the second, stopping is a positive act that has to
name a registered tool and bind its arguments. The asymmetry is not about
which is more elegant; it is that one design's failure mode is *silence*, and
silence is what every error already produces.

**FINDING: the shipped `finish` is neither.** It is a third reply kind. The
registry holds **3** names and `finish` is not one of them, so `dispatch` has
no path to a final answer. It is also the one place the loop indexes instead
of `.get`-ing: `reply["content"]` raises `KeyError` on a finish reply with no
content, while `reply.get("args", {})` tolerates an action reply with none.

**FINDING: three terminations, one turn kind.** A real finish, an exhausted
script and an exhausted budget all append `Turn(kind='final')` — contents
`the total including 15% tax is 138.0`, `no more actions`, `budget exhausted`.
Anything that branches on the kind sees one outcome where there are three,
and two of the three are failures.

### 3 — the loop already recovers, and the policy cannot

**ANSWER: a malformed script plus a critic that reads the observation back.**
**3** of **6** planned calls carry an argument dict that does not bind.
Without recovery the loop survives all **6**, records **3** `error:`
observations and finishes reading `missing:tax`. With recovery the same
script dispatches **9**, still **3** errors, and reads `18.0`.

The "make the loop recover" half was already shipped: `dispatch` catches
`TypeError` and returns a string, and `AgentLoop` stores it like any other
observation. Nothing in the loop needed changing.

**FINDING: the shipped policy has no path to recovery.** `ToyLLM.respond`
takes `history` and names it **0** times in its body. The script advances
identically whatever comes back, so the feedback channel the lesson built has
no reader. The critic retries **3** times; `ToyLLM` retries **0**.

**FINDING: `bad args` also reports faults the arguments did not cause.**
`calculator(expression=...)` fails to bind; `calculator(expr=120)` binds fine
and then raises `TypeError` inside the tool. Both come back prefixed
`error: bad args for calculator`, because `dispatch` has one
`except TypeError`. The model is told to fix its arguments for a bug in the
tool — which is Lesson 26's cascading failure in miniature.

**FINDING: `error:` is produced at two layers.** `calculator` returns
`error: illegal character in expr` through a *successful* dispatch; an unknown
tool returns `error: unknown tool` through a failed one. Both land as
`Turn(kind='action')` with the string in `observation`, so telling a rejected
call from an executed one that disliked its input means matching a prefix.

### 4 — the transcript loses the thoughts and gains a second budget

*Cites "The 2026 shift: native reasoning".*

**Three things change, and the control flow is not one of them.**

**The thought stops being something the loop wrote.** `AgentLoop.run` does
`thought = reply.get("thought", "")` and appends it as
`Turn(kind='thought', content=thought)` — a string `ToyLLM` made up, which
nothing ever reads again (exercise 3 measures this: `ToyLLM.respond` names
`history` **0** times in its body). Under the Responses API lineage the
reasoning arrives on its own channel as items the provider emits, and the
lesson is explicit that "that channel is passed through turns (encrypted
across providers in production)". So the transcript changes role: it stops
being a log the loop writes for humans and becomes an *input* the loop is
obliged to carry back unmodified. Dropping the shipped `thought` strings
changes nothing; dropping reasoning items changes the model's next decision.

**The transcript splits in two.** `pretty_trace` prints `turn.content` for a
thought turn, which presumes the thought is readable text. Encrypted
reasoning items are not, so the human-facing trace and the model-facing
history stop being the same object. What is left for a human is the shape the
lesson calls the loop — user, action, observation, final — with a hole where
the reasoning was. This is why Lesson 23's observability is a separate
subject: once the interesting part of the turn is opaque, "debugging step 38"
needs spans, not a printout.

**The budget gains a dimension the shipped loop has no field for.**
Ingredient 4 is "a turn budget to prevent infinite loops", and `max_turns` is
the whole of it. A real call adds tokens, latency and money per turn, and
reasoning tokens are charged while being invisible in the transcript — so
`max_turns=10` no longer describes the cost of a run. Exercise 1 finds the
same gap from the other side: the shipped budget counts the one thing that
used to be a proxy for everything.

One further change, which is exercise 2's contrast arriving as a fact: the
Responses API has no `finish` kind. An assistant turn with no tool calls *is*
the stop condition, so moving to a real provider means adopting the less safe
of the two stop paths unless a `finish` tool is declared on purpose.

### 5 — positional pairing is right until it is silently wrong

**ANSWER: a `tool_use_id` correlator over three calls returned in reverse.**
Keyed by id, **3** of **3** observations land on the call that produced them.
Keyed by position, **1** of **3** does.

**Why all three providers require it:** because the alternative fails
silently. Both transcripts here hold **3** action turns, **3** non-empty
observations and **0** errors. Nothing is missing; two things are wrong. A
protocol that let results return unlabelled would make every out-of-order
batch a well-formed lie, and neither the loop nor the model can detect it —
all three results are numeric strings, so a type check catches **0** of the
**2** swaps and a "looks like a number" check catches **0**.

**FINDING: the id has to be minted before the tool runs.**
`ToolRegistry.dispatch` takes a `ToolCall` and returns a bare `str` — no
channel for an identifier on the way out — and `ToolCall` carries `name` and
`args`, neither an id. So the correlator cannot be retrofitted at the tool
layer. It belongs on the request, echoed back by the result block, which is
exactly where Anthropic's `tool_use_id`, OpenAI's `call_id` and Bedrock's
`toolUseId` all sit.

**FINDING: the shipped loop hides the need for it.** The module names
`tool_use_id` **0** times, and the sequential run pairs **3** of **3**
correctly with no correlator at all, because `AgentLoop.run` dispatches and
appends in one statement. Position is a correct key for exactly as long as a
turn holds one call.
