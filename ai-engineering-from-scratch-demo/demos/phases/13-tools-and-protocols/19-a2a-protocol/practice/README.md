<!-- generated:start -->
# 13-tools-and-protocols / 19-a2a-protocol

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/19-a2a-protocol/) · upstream spec
`phases/13-tools-and-protocols/19-a2a-protocol/docs/en.md`

```bash
uv run demo practice run 19-a2a-protocol --ex 1
uv run demo explain 19-a2a-protocol --ex 1
uv run pytest demos/phases/13-tools-and-protocols/19-a2a-protocol
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Trace the full Task lifecycle, including the input-required pause where t… | code | T0 | `ex01_the_clarification_asks_for_a_key_the_resume_path_ignores.py` |
| 2 | Add a signed Agent Card. Sign with HMAC over the card's canonical JSON. Write a verifier and… | code | T0 | `ex02_canonical_json_is_the_signature_and_the_card_has_no_slot_for_it.py` |
| 3 | Implement task streaming: the writer agent emits three incremental artifact chunks over SSE a… | code | T0 | `ex03_the_chunks_carry_no_index_so_order_is_the_transports_promise.py` |
| 4 | Design an A2A agent that wraps an MCP server. Map each MCP tool to an A2A skill. Note the tra… | code | T0 | `ex04_the_wrapper_publishes_the_schemas_opacity_was_hiding.py` |
| 5 | Read the A2A v1.0 announcement and identify the one feature that is not yet implemented by an… | code | T0 | `ex05_multi_hop_delegation_has_no_field_to_travel_in.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code — including exercises 1 and
5, which read as observation and reading tasks. Exercise 5 asks what no
framework had shipped as of April 2026; that is not checkable from inside this
repo, so it is answered as the question the code *can* answer: what would
multi-hop delegation need, and how much of it exists in the lesson's model.

A2A's premise is that agents stay opaque to each other, and four of the five
exercises land on the same consequence: **the card and the task carry less
than the interaction needs, so the missing part ends up in prose.**

### 1 — the clarification asks for a key the resume path ignores

**ANSWER: `submitted` → `working` → `input_required` → `working` →
`completed`.**

**FINDING: the message names `target_length` and the code reads
`targetLength`.**

| reply key | outcome | length in the artifact |
|---|---|---|
| `target_length` (as printed) | completed | **short** — defaulted |
| `targetLength` (as coded) | completed | long |

One character apart, and no error anywhere.

**FINDING: the pause requires a key the resume does not.** `send` pauses
unless `targetLength` is present; `reply` resumes on *any* data part.

**FINDING: a prose reply is absorbed silently, and `state` has no transition
table.** `completed` → `submitted` is a plain assignment.

### 2 — canonical JSON is the signature, and the card has no slot for it

**ANSWER: HMAC-SHA256 over `sort_keys` JSON, detached.** Verifies; survives
reordering; fails on a flipped capability.

**FINDING: the signature cannot live in the card it signs.** Embedding it
changes the bytes it covers, so verification would need a stripping rule the
card does not state.

**FINDING: a mutation anywhere is caught, including inside a skill.** There
is no cosmetic region.

**FINDING: signing makes a claim attributable, not true.**
`capabilities.streaming` is `True` and the module has **0** streaming
functions.

### 3 — the chunks carry no index, so order is the transport's promise

**ANSWER: three chunks over SSE accumulate to the single-shot artifact, byte
for byte.**

**FINDING: nothing in the payload orders the chunks.** `Part` is
`(kind, payload)`; `Artifact` is `(name, mimeType, parts)`. Shuffled frames
produce a different document the accumulator cannot detect.

**FINDING: the terminal frame is the only thing that says "done".** No chunk
count is ever announced.

**FINDING: the card advertises streaming and the module implements none of
it.** The flag was set before anything implemented it.

### 4 — the wrapper publishes the schemas opacity was hiding

**ANSWER: one skill per tool, in the Agent Card's own shape.**

**WHAT IS LOST: the argument surface.** An A2A skill has **0** schema fields;
`inputModes` says what *kinds* of part it takes. Mapping faithfully turns
**4** property names from a machine-readable contract into prose — and mapping
*without* them is not opacity either, because the tools' own descriptions
already leak `query` and `destination`, **2** of the **4**, by accident. The
real choice is between a contract in prose and half a contract by accident.

**FINDING: the lesson's own agent already pays this price.** `draft_report`
declares three input modes and requires a `targetLength` key stated in **0**
machine-readable fields — exercise 1's failure mode exactly.

**FINDING: the reverse direction loses the lifecycle.** A skill backed by a
one-shot tool can never reach `input_required`, and the card cannot say which
kind it is.

### 5 — multi-hop delegation has no field to travel in

**ANSWER: delegation chaining, and `Task` has no field for it.** `id`,
`state`, `messages`, `artifact` — **0** can name a parent, a delegating agent
or a depth.

**FINDING: chaining by hand loses the relation on the first hop.** The link
becomes prose inside a text part, and no field would refuse a cycle.

**FINDING: the Agent Card cannot say an agent delegates.** An agent fanning
out to three others advertises exactly what one answering alone does.

**FINDING: the lifecycle has no state for waiting on a sub-task.** A blocked
parent is indistinguishable from a computing one, and `input_required` means
the *user*. Timeouts and cancellation have nothing to propagate along.

The lesson's own "Opacity preservation" section is why a card does not
describe internals — so a delegation chain is in tension with the property
A2A exists to protect, which is a likelier explanation for the gap than
nobody getting to it.
