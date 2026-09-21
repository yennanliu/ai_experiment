<!-- generated:start -->
# 14-agent-engineering / 08-memory-blocks-sleep-time-compute

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/08-memory-blocks-sleep-time-compute/) · upstream spec
`phases/14-agent-engineering/08-memory-blocks-sleep-time-compute/docs/en.md`

```bash
uv run demo practice run 08-memory-blocks-sleep-time-compute --ex 1
uv run demo explain 08-memory-blocks-sleep-time-compute --ex 1
uv run pytest demos/phases/14-agent-engineering/08-memory-blocks-sleep-time-compute
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `block_summarize` tool that replaces the block value with a model-generated summary whe… | code | T0 | `ex01_the_summarizer_returns_a_full_stop_when_nothing_fits.py` |
| 2 | Implement sleep-time dedup over archival: two records whose text has >90% token overlap colla… | code | T0 | `ex02_off_the_critical_path_is_a_convention_not_a_boundary.py` |
| 3 | Version blocks. On every write record the old value and a diff. Expose `block_history(label)`… | code | T0 | `ex03_the_history_is_shipped_and_the_writer_is_not_in_it.py` |
| 4 | Treat sleep-time agents as untrusted writers. When they touch the Persona or Safety block, re… | code | T0 | `ex04_the_protected_block_is_the_one_the_sleep_pass_rewrites.py` |
| 5 | Port the example to use the Letta API (`letta_v1_agent`). What changes in the block schema, a… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 is a port to the Letta
API, so it is answered in prose.

The recurring finding is that **`_summarize` returns `"."` whenever no whole
sentence fits the target**, and everything the sleep pass touches runs through
it. The lesson's own `task` block — **119** characters with no sentence break
— consolidates to a single full stop, a **99.2%** loss that `rewrite`
reports as `task v2 rewritten (1/220)`. Three of the lesson's four exercises
walk into this from different directions: the threshold sweep (ex 1), the
version history where exactly one entry has a negative delta (ex 3), and the
review gate whose whole purpose turns out to be stopping this one write
(ex 4). The lesson's own "silent drift" pitfall is not hypothetical here; it
is the default behaviour of the shipped consolidation path.

The second thread is that **every boundary in the design is a convention**.
`Block.append` reports `(39/10)` and stores 39 — the limit is a number in a
label. `Archival.invalidate` takes a `rid` and no caller, so "only in the
sleep pass" is enforced by where the code was typed. And the three write
methods take **0** actor parameters between them, so "treat sleep-time agents
as untrusted writers" cannot be expressed as a property of the writer at all.

What holds: the version counter and the history list. `Block` really does
record the previous value on all three write paths, which is what makes
exercise 3 a reader rather than a rewrite — and what makes the destructive
consolidation recoverable, if anyone thinks to look.

### 1 — the summarizer returns a full stop when nothing fits

**ANSWER: the frontier is 0.60, and it is exactly `1 - longest/limit`.** Over
12 appends against a 200-character block whose longest append is **80**
characters, thresholds **0.50** and **0.60** overflow **0** times and **0.70**
onward overflow **3** times each. Calls fall from **11** at 0.50 to **7** at
0.60 to **3** at 0.95, so **0.60** is the cheapest threshold that never
overflows — and the shipped default of **0.8** overflows three times. The
check runs *after* the append, so the trigger has to leave room for one more
of the largest writes; the closed form and the measurement agree to the step.

**FINDING: `_summarize` returns `"."` when no whole sentence fits.** It
splits on `"."`, keeps sentences while they fit the target, and ends with
`". ".join(picked) + "."` — an empty `picked` still gets the trailing full
stop. Any block written as key-value facts rather than prose is destroyed by
its first consolidation, and the failure is reported as a success.

**FINDING: the summarizer can grow its input.** When the whole text fits
under the target it is returned with a `"."` appended, so a **72**-character
value summarizes to **73**. A consolidation pass triggered on a block just
under the cap makes it bigger — which, at threshold 0.95, is a loop.

**FINDING: overflow is not a state the code can reach.** `Block.append`
returns `x v1 (39/10)` and stores all 39 characters. The block prints both
numbers and compares neither, so "block overflow" is something the caller
measures, not something the block refuses.

### 2 — off the critical path is a convention, not a boundary

**ANSWER: dedup collapses 3 pairs, leaving 9 of 12 records valid.** The scan
is pairwise Jaccard over token sets, keeping the earliest record of each
group, and every collapsed record is still in `all_records()` with
`valid=False` — auditable rather than destructive, which is the right default
for a pass that runs unattended.

**FINDING: nothing records *why* a record was invalidated.**
`ArchivalRecord` has **3** fields — `rid`, `text`, `valid`. A record dropped
as a duplicate and one dropped as contradicted by the `human` block are the
same object afterwards. Without a `superseded_by`, the pass cannot be undone
selectively and an operator debugging a lost fact cannot tell staleness from
redundancy.

**FINDING: 0.9 on token sets is a knife edge.** Against a ten-token record,
one extra word scores **0.909** and collapses; two extra words score
**0.833** and survive. The dedup rate is therefore a property of how verbose
the writer was, not of how redundant the facts are — and the primary agent
writing "fast and raw" is exactly the writer whose verbosity varies.

**FINDING: the boundary is a comment.** `PrimaryAgent` holds `.archival`,
and `Archival.invalidate` takes a `rid` and no caller. A primary turn can
invalidate mid-conversation and nothing notices; **0** of the store's **4**
public methods restrict the writer. "Never on the critical path" is enforced
by where the code was typed, which survives exactly as long as nobody is in a
hurry.

### 3 — the history is shipped, and the writer is not in it

**ANSWER: `block_history(label)` over the shipped `history` and `version`.**
`Block` already appends the previous value on all three write paths, so the
exercise is a reader plus a diff. Three writes to a `task` block give **3**
entries with deltas **+52**, **+69**, **-120**; exactly **1** is negative,
and it is the consolidation that left the block holding `"."`. That one row
is the answer to "why did the agent forget X".

**FINDING: nothing in the record says who wrote.** `append`, `replace` and
`rewrite` take **0** actor parameters between them, and both agents hold the
same `BlockStore`. The destructive entry is distinguishable from the benign
ones only by its sign — had the primary agent truncated the block, the
history would be identical. For the lesson's "silent drift" pitfall, the
diff answers *what* changed and never *who*.

**FINDING: one of the three write paths can lose data.** `append` can only
grow the value; `replace` fails loudly when `old` is absent; `rewrite`
replaces outright. So the single lossy path is the one the sleep pass calls
automatically, with no argument saying how much loss is acceptable. A
`max_loss` parameter on `rewrite` would have made exercise 4 unnecessary.

**FINDING: the history is full copies and unbounded.** After three writes the
block holds **1** character of value and **173** of history, and nothing
trims it. A block rewritten every sleep pass keeps a full copy of its
pre-consolidation value forever — the storage the consolidation was run to
reclaim, retained under a different name.

### 4 — the protected block is the one the sleep pass rewrites

**ANSWER: a reviewed store, and the reviewer rejects 1 of 3 writes.** Two of
three proposed sleep-time writes touch protected labels. The `persona`
consolidation is **rejected** for dropping all **4** of the block's declared
facts; the `safety` edit is **approved** because it keeps them; the `task`
write is unprotected and commits unreviewed. The reviewer reads only the
before and after values — it needs no model and no trust in the writer.

**FINDING: the write the policy exists to stop is the one the sleep pass
issues by default.** With `persona` near its limit, `SleepTimeAgent.run`
rewrites it to `"."` — **4** declared facts to **0** — and reports
`persona v2 rewritten (1/140)`. Unreviewed, the agent's self-concept is one
consolidation pass from empty, and the trace line reads like success.

**FINDING: there is no writer to distrust.** The three write methods take
**0** actor parameters, and `BlockStore.create` takes `label`, `description`,
`limit` with no way to mark a block. "Untrusted sleep-time agent" cannot be a
property of the caller in this design, so the gate has to key on the *label*
— which is a weaker policy, because it protects names rather than actions.

**FINDING: the gate's scope is the whole policy.** The unprotected `task`
block takes the identical destructive rewrite and is the one write of three
that skipped review entirely. The review does not make consolidation safe; it
makes two labels safe. Every other block has to be enumerated in advance, and
the failure mode of forgetting one is silent.

### 5 — the thought leaves the trace, and the block gains a type

*Cites "Native reasoning".*

**Two changes, and only one of them is about memory.**

**The block schema gains identity and description as first-class fields.**
The lesson states Letta's block shape: `id`, `label`, `value`, `limit`,
`description` — "so the model knows when to edit it". The toy has `label`,
`value`, `limit`, `description`, `version`, `history`, and the difference
that matters is not the field list but which fields are *reachable*. Three
things move:

1. **Blocks become addressable by `id`, not by label.** In `BlockStore`,
   `create(label, …)` keys the dict by label and `get(label)` is the only
   lookup, so two agents sharing a store share a namespace where a label
   collision is a silent overwrite. An `id` separates "which block" from
   "what it is called", which is what lets a sleep-time agent hold a
   reference to a block whose label the primary agent is free to change.
2. **`description` starts doing work.** In the toy it is set once in
   `create` and read by nothing — `render()` prints label, version, length
   and value, and the sleep agent branches only on `near_limit()`. Under the
   Letta surface the description is what the model reads to decide *whether*
   a fact belongs in this block, which is the decision exercise 4 had to fake
   with a hard-coded `PROTECTED` tuple.
3. **`block_summarize(label)` is a declared tool rather than an inline
   branch.** The lesson lists it in the block tool surface; in the toy it is
   three lines inside `SleepTimeAgent.run`. Exercise 1 is exactly that
   promotion, and doing it is what makes the trigger threshold a parameter
   anyone can measure — which is how the frontier at **0.60** became visible
   at all.

What the port does *not* fix on its own is the thing all four exercises kept
hitting: `limit` is still advisory unless the runtime enforces it, and
`_summarize`'s `"."` is a property of the summarizer, not of the schema. The
lesson's own "block bloat" advice — "wire a block summarizer before the write
that pushes over the cap" — is the enforcement, and it belongs on the write
path, not in a background pass.

**Native reasoning changes the trace, not the loop.** The lesson is explicit:
`letta_v1_agent` deprecates `send_message`/heartbeat and inline `Thought:`
tokens, the reasoning arrives on its own channel passed through turns
(encrypted across providers in production), and "the control loop is still
ReAct. The thought trace is structural, not prompt-shaped." Three consequences
for this lesson specifically:

- **The trace stops being a string list.** `PrimaryAgent.trace` and
  `SleepTimeAgent.trace` are `list[str]` built with f-strings — a log written
  for a human. Structural reasoning items are typed records the runtime must
  carry back unmodified, so the trace becomes an *input* as well as an
  output. Anything that truncates it changes behaviour.
- **The two traces stop being comparable.** Today the primary and sleep-time
  agents produce the same kind of artifact, which is why exercise 3 can find
  a destructive write only by its sign. With reasoning on a separate channel
  per agent, the runtime knows which agent produced which item — the writer
  identity exercises 3 and 4 both had to go without arrives for free, from
  the transport rather than from a new parameter.
- **Sleep-time compute gets cheaper to justify.** The lesson's argument for
  it is that the sleep agent can be "a more expensive, slower model because
  it is not latency-constrained". Reasoning tokens are the expensive part of
  a 2026 turn, and they are charged whether or not they are shown — so moving
  reasoning-heavy work off the critical path is a latency *and* a token
  argument, and the block history from exercise 3 is what lets you audit
  whether the more expensive model actually wrote better blocks.
