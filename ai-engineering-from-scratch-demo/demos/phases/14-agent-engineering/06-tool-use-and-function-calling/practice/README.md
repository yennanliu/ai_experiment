<!-- generated:start -->
# 14-agent-engineering / 06-tool-use-and-function-calling

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/06-tool-use-and-function-calling/) · upstream spec
`phases/14-agent-engineering/06-tool-use-and-function-calling/docs/en.md`

```bash
uv run demo practice run 06-tool-use-and-function-calling --ex 1
uv run demo explain 06-tool-use-and-function-calling --ex 1
uv run pytest demos/phases/14-agent-engineering/06-tool-use-and-function-calling
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a "no-op" tool that lets the model explicitly refuse to use any other tool. Measure on a… | code | T0 | `ex01_declining_and_inventing_a_tool_name_are_the_same_event.py` |
| 2 | Implement argument coercion for int-as-string and float-as-string. Where does coercion start… | code | T0 | `ex02_coercion_uses_pythons_number_parser_not_the_wires.py` |
| 3 | Add a per-tool timeout and a circuit breaker (refuse the tool for 60s after 3 consecutive fai… | code | T0 | `ex03_the_breaker_counts_ok_false_and_a_bad_argument_is_ok_false.py` |
| 4 | Read BFCL V4 description. Pick one category (e.g. "multi-turn") and run 10 example prompts th… | code | T0 | `ex04_every_turn_picks_the_right_tool_and_six_answers_are_wrong.py` |
| 5 | Port the stdlib validator to Pydantic or Zod. What did Pydantic/Zod catch that the toy missed? | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 asks for a port to a
third-party schema library, so it is answered in prose rather than dragging
`pydantic` into a T0 lesson's dependency group.

The recurring finding is that **`ToolResult` has three fields and the
registry asks them to carry five different meanings**. A validation error, an
unknown tool name, an execution exception, a correct refusal, and — once
exercise 3 adds them — a timeout and an open circuit all arrive as
`ok=False` with free text in `content`. Every exercise in this lesson ends up
somewhere in that collapse: declining is indistinguishable from inventing a
tool name (ex 1), the circuit breaker cannot tell a broken tool from a
malformed argument (ex 3), and a silently wrong multi-turn answer is
`ok=True` with no error at all (ex 4).

The second thread is that **coercion is where types stop being checked**.
`_coerce` runs Python's `int()` and `float()` on strings, so the validator
accepts `5_0`, `  7  `, an Arabic-Indic `٥` and `nan` — and `nan` clears a
`minimum`/`maximum` range because every comparison with NaN is false.
Underneath that, nested objects and array items are not validated at all.

What holds: the dispatch shape. Every failure really does come back as an
observation rather than an exception, correlation IDs really are threaded
through, and `catalog()` really is the model-facing surface. Those are the
parts the lesson set out to demonstrate and they are correct.

### 1 — declining and inventing a tool name are the same event

**ANSWER: a `no_tool` tool taking a `reason` string, and the score moves from
5/10 to 10/10.** Over a ten-case fixture split five tool / five no-tool, a
model that must always call something scores **5/10** overall and **0/5** on
the hallucination half. With `no_tool` registered, a correct decliner scores
**10/10** and **5/5**, and all **10** dispatches return `ok=True` carrying
the reason it declined.

**FINDING: before the no-op exists, the category cannot be scored at all.**
`dispatch` on `no_tool` in the bare registry returns
`error: unknown tool 'no_tool'` — the same `ok=False`, the same message
shape, the same `ToolResult` as the demo's deliberately bogus `subtract`
call. A correct refusal and a hallucinated tool name are one event class. No
metric can separate them, which is the concrete reason BFCL needed a
hallucination category rather than an accuracy number.

**FINDING: refusing in the runtime instead leaves holes in the trace.** A
threshold that suppresses the call produces **5** `ToolResult`s for **10**
prompts while `catalog()` still advertises **3** tools. The model was never
told that declining was permitted — the refusal is a fact about the harness
that never reached the prompt — and nothing records why five prompts produced
nothing.

**FINDING: the no-op's cost is that refusal becomes a success.** A correct
decliner and a model that declines *everything* both return **10/10**
`ok=True` results, while scoring **10/10** and **5/10** on the task. Any
dashboard built on dispatch success rate ranks them equal. Adding the no-op
fixes the scoring problem and creates a monitoring one.

### 2 — coercion uses Python's number parser, not the wire's

**ANSWER: both coercions are already shipped, and they disagree with each
other.** `_coerce` has `int(value)` and `float(value)` on its `str` branches.
`"5"` becomes `5`, `"5.5"` becomes `5.5`, and `"5.0"` is a valid `number` and
an invalid `integer` — the same token accepted or rejected by the schema's
type rather than by its own shape.

**Where coercion starts to hide real bugs: when it converts and discards.**
A tool handed `"5_0"` is called with `50`, the dispatch reports `ok=True`,
and nothing anywhere records the original token. The tool cannot reject it,
the log cannot show it, and a post-mortem cannot find it.

**FINDING: `"nan"` passes a bounded range.** Against
`{"type": "number", "minimum": 0, "maximum": 100}`, the string `nan` coerces
to `float('nan')` and clears both bounds, because every comparison with NaN
is false. `"inf"` — which is the value people worry about — gets a clean
`inf > maximum 100`. The dangerous one is the one that fails *silently*, and
it is the one JSON cannot even express.

**FINDING: the accepted set is Python's, not JSON's.** Of the **7** probe
strings the validator accepts, only **3** are JSON number tokens. `"5_0"`
becomes **50** on Python's underscore separators, `"  7  "` becomes **7** on
whitespace stripping, and `"٥"` becomes **5** because `int()` reads Unicode
decimal digits. No provider's JSON could have produced any of those three as
a number, which means accepting them cannot be robustness — the only thing
that can send them is something that is already wrong.

### 3 — the breaker counts `ok=False`, and a bad argument is `ok=False`

**ANSWER: a per-tool timeout and a 60-second breaker over the lesson's own
registry.** A tool with `timeout_s=1.0` called at a declared cost of **3.0**
returns `timeout after 1.0s`; **3** consecutive timeouts open the breaker and
the fourth call is refused without reaching the tool — **0** executions
across **4** calls. Advance the clock **60** seconds and the next call runs
and returns `3`. (Duration is a declared cost against a virtual clock, not a
measurement, so nothing here depends on the host.)

**FINDING: `timeout_s` was declared and never read.** `ToolDef` carries it as
its fifth field and the source of `dispatch` mentions it **0** times. A field
on a public type is a promise; this one had no enforcement anywhere in the
module, which is the most common shape of this bug in real registries.

**FINDING: the breaker fires on failures the tool never saw.** Consecutive
failures are counted from `ok=False`, and `ok=False` also covers validation
errors. **3** calls missing a required argument execute the tool **0** times
and open the breaker anyway — after which a *correct* call is refused for 60
seconds. The model's own malformed output takes a healthy tool offline, and
the fix is not a better breaker but a failure taxonomy.

**What it changes about how the model recovers: it adds a third instruction
and no field to carry it.** Validation errors mean *fix the arguments*,
timeouts mean *retry smaller*, an open circuit means *wait*. All three arrive
as `ok=False` with the difference only in free text, and `ToolResult` has
three fields — `tool_use_id`, `ok`, `content` — none of them a reason code.
A model that has learned to answer failure by editing arguments will keep
editing arguments at an open circuit, burning the cooldown on edits that
cannot matter.

### 4 — every turn picks the right tool, and six answers are wrong

**ANSWER: the multi-turn pass rate is 4/10 without carried state and 10/10
with it.** Ten two-turn prompts: add two numbers, then multiply the result by
a third. **6** refer to the first answer as "that"; **4** restate it. The
agent that does not carry state passes exactly the **4** it was handed the
number for, and the **6** failures are exactly the **6** implicit ones.

**FINDING: the failures are silent.** All **20** turns dispatch with
`ok=True` and **0** validation errors. A missing carried value becomes a
plausible default rather than a malformed argument, so six wrong answers
arrive as successful tool results. There is no error anywhere in the trace.

**FINDING: AST-style scoring would call it perfect.** Scored on the tool name
chosen each turn, the stateless agent matches **20/20** — identical to the
carrying agent. Scored on the final value it matches **4/10**. This is BFCL
V3's change reproduced at ten cases: matching the shape of the call tree
cannot see a wrong argument that is well-typed, and the failures that survive
2026 single-turn training are exactly of that kind.

**FINDING: the registry has nowhere to put the carry.** `ToolCall` carries
`tool_use_id`, `name`, `args`; `ToolResult` carries `tool_use_id`, `ok`,
`content`; **0** session or state handles between them, and `dispatch_many`
maps a list to a list with no accumulator. Multi-turn state is the caller's
job by construction. That is a defensible design — but it means "we have a
tool registry" says nothing about whether multi-turn works, which is why the
agentic and multi-turn categories are 70% of BFCL V4 while single-turn
calling is described as near-solved.

### 5 — what a real schema library catches that the subset cannot

*Cites "Argument validation".*

**The port is short, and the list of things it fixes is not.** Three of the
four items in the lesson's own validation list are unimplemented or
implemented at one level only, and the measurements from exercises 2 and 4
are all symptoms of the same two root causes: the validator does not recurse,
and its coercion is a language's parser rather than a declared policy.

**What it catches, in order of how quietly the toy fails.**

1. **Nested structure — the big one.** `_coerce` on `{"type": "object"}` is
   `isinstance(value, dict)` and nothing else; on `{"type": "array"}` it is
   `isinstance(value, list)`. So a schema declaring
   `user: {age: {type: integer, minimum: 0}, required: [age]}` accepts
   `{"user": {"age": "not a number", "extra": 1}}` with **0** errors, and
   accepts `{"user": {}}` — a missing *nested* required field — with **0**
   errors too. An `items` schema is never consulted, so
   `{"xs": ["a", {}, null]}` validates clean against
   `array of integer`. Pydantic and Zod validate the whole tree because a
   schema *is* a tree; the toy validates the first level because it is a
   loop over `args.items()`. Every tool in this lesson happens to take flat
   integer arguments, which is why the demo never surfaces it.
2. **Error paths.** The toy's errors are flat strings — `"a: expected
   integer, got str"` — with no location. Pydantic returns a `loc` tuple per
   error and Zod a `path` array, so a nested failure says *which* field.
   Since the toy cannot produce a nested failure at all, it has never needed
   one; adding recursion without adding paths would produce errors the model
   cannot act on, which matters because the lesson's whole point is that a
   validation failure is an observation the model retries against.
3. **Finiteness.** Exercise 2's sharpest finding: `"nan"` clears
   `minimum: 0, maximum: 100`. A real library treats this as a declared
   policy — Zod has `.finite()`, Pydantic has `allow_inf_nan` — and JSON
   Schema itself has no NaN literal, so the correct default is to reject.
   The toy has no policy; it has whatever `float()` does.
4. **A declared coercion policy, with a strict mode.** The difference is not
   that a library coerces better — it is that coercion is a *choice you
   write down*. Zod does not convert a string to a number unless you ask for
   `z.coerce`, and Pydantic distinguishes strict from lax validation. The
   toy has exactly one mode and it is Python's: `int("5_0") == 50`,
   `int("  7  ") == 7`, `int("٥") == 5`. Ported to either library the same
   schema either rejects those outright (strict) or runs them through a
   different parser with different quirks (lax) — and in both cases the
   behaviour is a line in the schema rather than an accident of the host
   language. Exact per-version behaviour differs; the point that survives
   any version is that the toy's behaviour is not written down anywhere.
5. **The keywords the subset does not have.** `minLength`, `maxLength`,
   `pattern`, `format`, `const`, `anyOf`, `oneOf`, `$ref`, `additionalProperties`,
   `minItems`, `uniqueItems`. The lesson's fourth validation bullet — "dates,
   emails, URLs — validate with concrete parsers, not regex" — has no
   implementation at all: `{"type": "string", "minLength": 3}` accepts `""`.
   And `additionalProperties` is not configurable in the other direction
   either: unknown fields are always an error, so a schema that wants to
   accept passthrough keys cannot say so.
6. **The raw value survives.** Exercise 2's finding again, from the port's
   side: the toy returns only `out[name] = coerced`, so the original token is
   gone by the time `executor(**validated)` is called. `safeParse` returns
   parsed data beside the input, and a Pydantic model keeps the raw payload
   available to the caller. That is what makes "coerced `5_0` to `50`"
   loggable instead of invisible.

**What the toy gets right and a naive port would lose.** Every failure
returns a *string the model can read and retry against* rather than raising —
`dispatch` catches `Exception` and wraps it. Pydantic raises
`ValidationError` and Zod's `parse` throws; a port that keeps `.parse()`
instead of `.safeParse()` turns every malformed tool call into a crash in the
agent loop, which is the one property the lesson's closing line insists on.
The port is `safeParse` plus a formatter from the library's structured errors
back to a single observation string — and that formatter is where the
`loc`/`path` information from item 2 earns its keep.
