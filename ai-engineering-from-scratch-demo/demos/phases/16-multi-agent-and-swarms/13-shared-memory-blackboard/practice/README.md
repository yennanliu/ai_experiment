<!-- generated:start -->
# 16-multi-agent-and-swarms / 13-shared-memory-blackboard

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/13-shared-memory-blackboard/) · upstream spec
`phases/16-multi-agent-and-swarms/13-shared-memory-blackboard/docs/en.md`

```bash
uv run demo practice run 13-shared-memory-blackboard --ex 1
uv run demo explain 13-shared-memory-blackboard --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/13-shared-memory-blackboard
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm run 1 propagates the hallucination and run 2 catches it. | code | T0 | `ex01_run_two_flags_the_lie_and_ships_the_same_verdict.py` |
| 2 | Add a second hallucination: agent B invents a dataset size. The verifier should catch both wi… | code | T0 | `ex02_a_number_check_catches_both_and_the_shipped_verifier_cannot_see_b.py` |
| 3 | Switch the full pool to a blackboard with topic partitions (`prices`, `summaries`, `analyses`… | code | T0 | `ex03_partitions_stop_poison_flowing_upstream_and_not_down.py` |
| 4 | Read Hayes-Roth (1985, "A Blackboard Architecture for Control"). Identify two control pattern… | explain | T0 | prose, below |
| 5 | Read CA-MCP (arXiv:2601.11595). Map its Shared Context Store to either the MessagePool or Bla… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — run two flags the lie and ships the same verdict

**Run 1 propagates the hallucination. Run 2 flags it and then ships it
anyway.** Run 1's analyst writes "Recommend adoption" on the 42% figure, and
nothing is flagged. In run 2 the verifier returns **1** finding and entry 0
is flagged. The analyst then writes a verdict that is **byte-identical** to
run 1's. `analyst_agent` never reads `flags`, and no code writes the
retraction the lesson promises: both pools hold 3 entries.

| entry | writer | repeats 42% | has `source_uri` | flagged in run 2 |
|---|---|---|---|---|
| 0 | retriever | yes | yes | **yes** |
| 1 | summarizer | yes | no | no |
| 2 | analyst | yes | no | no |

**The flag stops one hop short.** The verifier only checks entries that have
a `source_uri`. `ProvenanceEntry` has no field naming what an entry was
derived from. So the two entries that repeat the lie cannot be flagged, by
construction.

**The honest pipeline corrupts the true figure.** The summarizer keeps
`latest.split('.')[0]`, and the decimal point in "4.2%" is a period. So the
honest summary ends with "The study reports a 4." The hallucinated 42% has no
decimal point and gets through intact.

Two more things show up in the same runs. Provenance cannot tell the runs
apart: `prompt_hash` hashes the prompt rather than the output, so all 3
hashes are the same in the honest and the poisoned run. And the analyst's
rule is `"42%" in latest`, so it recommends adoption exactly when the lie is
present.

### 2 — a number check catches both, and the shipped verifier cannot see B

**Rule: every number in every entry must appear in some fetched source.**
The verifier re-fetches each cited `source_uri` and collects the numbers the
sources actually contain, {4.2, 12500}. Then it flags any entry that states
a number outside that set. The rule names no figure, no source and no agent.

In the test, the retriever drops paper-1's decimal and the summarizer
appends "on 50,000 examples" (paper-2 says 12,500). The rule flags entries
**0, 2 and 3**: the retriever's 42, the summarizer's 42 and 50,000, and the
analyst's echo of both. It leaves the honest paper-2 write alone.

The shipped verifier flags only entry 0, and it would flag only entry 0
whatever B wrote. The summarizer writes `source_uri=None`, and those are the
entries the verifier skips. Its test is also `content != truth`, which is
tuned to verbatim copying: a faithful paraphrase ("Accuracy rose 4.2% over
the baseline.") counts as a mismatch, while the number check passes it.

On the honest pipeline the number check flags exactly one entry: the summary
truncated to "reports a 4.", which really is wrong.

Its limit is lineage. No entry records which source it came from, so the
only thing to check against is the union of all sources. With an honest
retriever, a summary claiming paper-1 showed a "12,500% improvement" gets
**0** flags, because 12,500 appears in paper-2.

### 3 — partitions stop poison flowing upstream, and not down

Wire the lesson's chain onto the reference `Blackboard`: retriever →
`prices` → summarizer → `summaries` → analyst → `analyses`. Then inject a
poisoned entry into each topic in turn:

| poisoned topic | agents reached |
|---|---|
| `prices` | summarizer, analyst |
| `summaries` | analyst |
| `analyses` | none |

**Partitioning makes it harder to poison from downstream. It does not help
when the poison starts at the root.** The lesson's scenario is a retriever
hallucination, which is root poison. The partitioned run still ends in
"Recommend adoption", because partitions keep exactly the edges the lie
travels along.

Even the downstream protection depends on something the class does not have.
`publish` takes `writer` and `topic` as free arguments, so any agent can write
to `prices`, and the summarizer will consume it. A four-line ownership check
rejects that.

Two further holes are unrelated to partitioning:

- **A subscriber can rewrite history and frame the writer.** The callback
  receives the stored entry itself, not a copy. A subscriber that edits 4.2%
  to 42% changes what `read_topic` returns to everyone after it, and the
  entry still says `writer="retriever"`. `MessagePool.read_all` copies the
  list but not the entries, so the pool has the same hole.
- **One failing subscriber leaves a topic out of sync.** Callbacks run in
  order, and if the first raises, the second never receives the entry, even
  though it is stored. Readers of the same topic then disagree about what it
  contains.

### 4 — control as a blackboard problem, and triggers that wait

*Draws on "Blackboard precedent (Hayes-Roth, 1985)".*

The lesson borrows the domain half of the architecture: knowledge sources
contributing to a shared board. The paper's contribution is the other half.
It separates domain problems, knowledge and solutions from **control**
problems, knowledge and solutions, so that a system can reason about its own
behaviour. BB1, built on it, keeps two blackboards. Two patterns from it are
missing here.

**1. Control decisions as entries on their own board.** Strategy and focus
decisions — what to work on now, which heuristics apply — are written by
control knowledge sources onto a **control blackboard**. They can be read,
revised and explained like any other partial solution. A 2026 orchestrator
keeps "why agent X ran next" inside a prompt or a code path. Writing it to a
board, with the same provenance as domain writes, makes a poisoned *plan* as
auditable as a poisoned *fact*. It also lets a meta-level source notice that
the current strategy is not working and switch.

**2. Triggering separated from execution.** In the blackboard tradition an
event does not run a knowledge source. It creates an activation record
(KSAR), which waits on an agenda. A scheduler picks one per cycle, rating
candidates on criteria such as the source's reliability and **the
credibility of the triggering information** against the current focus.
`code/main.py` is the degenerate case. `Blackboard.publish` runs every
subscriber immediately, in the publisher's thread, and the analyst runs no
matter what the verifier found. A scheduler that rates activations on the
credibility of the entry that triggered them is exactly the missing link
between exercise 1's flag and exercise 1's verdict. A flagged entry would
lower the rating of everything it triggers.

### 5 — CA-MCP's store is a message pool that calls itself a blackboard

*Draws on "The two main topologies".*

The paper describes the Shared Context Store as "a centralized blackboard
for coordination, providing a single source of truth for task state,
constraints, and intermediate outputs" (§2.1). It is **one store, not
topic-keyed**. MCP servers are "stateful reactors" that "monitor the SCS for
relevant triggers" and then read the current context, compute, and write
back. The paper names no topics and no subscribe or notify call.

Structurally, then, it maps to **`MessagePool`**: one global store that every
participant reads. The lesson's introduction files CA-MCP under
"blackboard with subscription"; the paper describes condition-triggered
polling of a single store.

**What CA-MCP adds on top:**

- **A seeded plan.** The central LLM writes "an initial contextual blueprint
  (e.g., goals, constraints, execution outline)" into the store before any
  server runs, so the plan lives in shared state.
- **Condition triggers.** Servers wake on the store's contents, not on
  explicit dispatch.
- **Readiness signals.** A server "signals readiness for subsequent tasks"
  when it finishes.
- **A final read.** The central LLM reads the completed state back to
  synthesise the answer.

The reported gains come from removing LLM round-trips: 5 → 2 calls on
TravelPlanner and 2 → 1 on REALM-Bench.

**What it does not have is this lesson's defence.** The paper specifies no
provenance, no versioning or supersession, no access control, and no
verifier, and it does not discuss hallucinated or conflicting writes. A
server that writes a wrong intermediate value into the "single source of
truth" poisons every server triggered after it, and the central LLM
summarises it at the end. That is exercise 1's run 1, with nothing in the
design that would make it run 2.
