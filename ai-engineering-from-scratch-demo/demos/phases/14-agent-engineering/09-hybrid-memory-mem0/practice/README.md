<!-- generated:start -->
# 14-agent-engineering / 09-hybrid-memory-mem0

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/09-hybrid-memory-mem0/) · upstream spec
`phases/14-agent-engineering/09-hybrid-memory-mem0/docs/en.md`

```bash
uv run demo practice run 09-hybrid-memory-mem0 --ex 1
uv run demo explain 09-hybrid-memory-mem0 --ex 1
uv run pytest demos/phases/14-agent-engineering/09-hybrid-memory-mem0
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace the toy vector similarity with a real embedding model (sentence-transformers, Ollama,… | code | T0 | `ex01_the_drift_question_is_answerable_without_an_embedding_model.py` |
| 2 | Add a temporal query: `search(query, as_of=timestamp)`. Return only records valid at or befor… | code | T0 | `ex02_the_graph_knows_when_a_fact_started_and_not_when_it_stopped.py` |
| 3 | Implement a conflict detector: if an incoming fact contradicts a graph edge, invalidate the o… | code | T0 | `ex03_the_graph_updates_and_the_other_two_stores_never_hear.py` |
| 4 | Port the fusion scorer to include a `user_feedback` dimension (thumbs-up on retrieved records… | code | T0 | `ex04_normalising_by_impressions_does_not_fix_exposure_bias.py` |
| 5 | Read the Mem0 docs (`docs.mem0.ai`). Port the toy to `mem0` client calls. Compare retrieval q… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 needs the hosted Mem0
client, so it is answered in prose.

The recurring finding is that **the three stores never talk to each other**.
`GraphStore.add_edge` invalidates the superseded edge correctly — the
conflict detector exercise 3 asks for is half shipped — and then the vector
store still ranks `ava lives in Berlin` first for `where does ava live`, and
the KV store still holds both cities on two different keys. A hybrid memory
whose arms disagree is not three views of one truth; it is three truths, and
the fusion scorer picks between them by weight.

The second thread is that **`Mem0.search`'s KV arm never looks at the
query**. It appends every KV record for the user at a fixed relevance of
**0.4** — a floor of **0.24** on the fused score — with no query term and no
scope check, so a query with **0** vector matches still returns **3**
results, all about other subjects. Combined with a recency term that has
decayed to **0.0016** of the score after a week, a long-lived store answers
most questions with importance plus whatever was written down.

What holds: the scope taxonomy. `user_id` and `scope` really do keep Bob's
refund out of Ava's vector results, and the split across three stores really
is the shape Mem0 describes. The gaps are in the seams.

### 1 — the drift question is answerable without an embedding model

**ANSWER: recall@10 is 19/20 for trigram cosine and 0/20 for token
overlap.** Over **1020** records, each query is a single word in the
inflection the record does not use — `drifted` against `drifting`. The
shipped scorer shares **0** tokens with its target and returns nothing at
all; CRC-hashed character trigrams share `dri`, `rif`, `ift` and find **19**.
The one miss is the honest part: trigrams are robust to a word ending, not to
meaning, so a stem that collides with the filler vocabulary is still lost.
That is the boundary a real embedding model moves, and naming it is more
useful than pretending a hash is a sentence encoder.

**FINDING: it drifts once and then freezes, which is worse than drifting.**
For one fixed query the shipped top-10 at 100, 250, 500 and 1000 writes
overlaps its predecessor by **0.43**, then **1.0**, then **1.0**. Between 250
and 1000 writes, **750** new memories arrive and **0** of them enter the top
ten. Nothing in the scorer is unstable — the best reachable score was simply
already taken, and after a few hundred writes the store stops being able to
surface anything new for that query. The lesson's "embedding drift" pitfall
describes results degrading; this is the other failure, where they stop
moving at all.

**FINDING: the KV arm ignores the query.** Every KV record for the user is
appended at relevance `0.4` with no query term and no scope check.

**FINDING: recency stops contributing within a week.** At the shipped
half-life of **86400** seconds a 7-day-old record scores **0.0078** on
recency and contributes **0.0016** of a fused score. The recency dimension is
a tie-breaker for today and nothing at all for anything older, which for a
memory system is most of the corpus.

### 2 — the graph knows when a fact started, not when it stopped

**ANSWER: `as_of` across all three stores, and the vector arm is a filter.**
With Berlin written at T1 and Lisbon at T3, filtering `Record.ts <= as_of`
and breaking ties toward the newest answers **3** of **3** probe times
correctly. The graph arm answers **1** of **3**.

**Which store needs the most work: the graph.** `Edge` carries `subject`,
`relation`, `obj`, `valid`, `ts` — **0** fields recording when an edge
stopped being valid. `valid` is a snapshot, so at T2 the query returns the
empty list: the only edge old enough has since been marked invalid, and the
store cannot say it used to be true. The lesson's own description of Mem0g —
"temporal queries traverse the valid-at-time subgraph" — requires an
interval, and this `Edge` has a start and a boolean.

**FINDING: the KV arm has lost the question, not the time.** `by_user`
returns `list[Record]`, and `Record` has **0** fields naming a fact type —
the type lives in `KVKey`, which the public read path never returns. "What
was ava's city at T2" is unanswerable even at T = now: the two city records
come back as **2** untyped rows among **3**. The KV tier's whole advantage is
that it is keyed, and the read API drops the key.

**FINDING: the two city facts never collide.** `KVKey` includes `entity`, so
`(ava, city, Berlin)` and `(ava, city, Lisbon)` are different keys and `put`
overwrites neither. That is the design the lesson's own "KV schema creep"
pitfall warns about, seen from the inside: the key is granular enough that
nothing is ever replaced.

### 3 — the graph updates and the other two stores never hear

**ANSWER: a detector that names both sides.** Writing `lives_in Lisbon` over
`lives_in Berlin` leaves **1** invalid edge and **1** valid one, with a
**1**-entry log naming both objects, and `neighbors("ava")` returns exactly
`Lisbon`. The invalidation itself was already shipped; the log is the new
part, and it is what makes the next two findings visible.

**FINDING: the vector store still answers Berlin.** Asked
`where does ava live`, the vector arm returns Berlin first at **0.143** —
tied with Lisbon, broken by insertion order. The conflict was detected in the
graph, and the arm that actually feeds text to the model is one the detector
never touches. Any fix has to either propagate invalidation into the vector
store or teach the fusion scorer to consult the graph, and the shipped
`Mem0.search` does neither.

**FINDING: the KV store holds both cities**, on two keys, and `Mem0.search`
injects both at its relevance floor — so the stale fact is not merely present
but *guaranteed* to be retrieved.

**FINDING: the invalidation does not look at the object.** `add_edge`
invalidates on `(subject, relation)` alone, so writing the same fact twice
leaves **2** edges, **1** invalid, both pointing at `Lisbon` — while the
detector correctly logs **0** conflicts. An idempotent write churns the
graph, which matters directly for the lesson's "graph explosion" pitfall: a
noisy extractor that re-asserts stable facts doubles the edge list without
changing a single answer.

### 4 — normalising by impressions does not fix exposure bias

**ANSWER: a fourth fusion term, and the loop it creates.** An incumbent with
**0** query relevance but high importance scores **0.24** and wins rounds
1–24, collecting **24** thumbs-up. A genuinely better record arrives at round
25 scoring **0.30** on the base fusion — and under a naive count it wins
**0** of the remaining **26** rounds. The gaming the exercise asks about does
not require a malicious agent: only retrieved records can be rated, so
ratings measure exposure and exposure follows ratings.

**FINDING: normalising by impressions does not help.** The incumbent's rate
is **1.0** because every impression it ever had ended in a thumbs-up, and the
newcomer has **0** impressions to divide by. Rate normalisation fixes volume
bias *between two rated records*; it cannot touch the bias between a rated
record and an unrated one, and the lock-in stays at **0/26**. This is worth
stating plainly because impressions-normalisation is the first mitigation
everyone reaches for.

**FINDING: capping the term below the merit gap is what works.** The base gap
is **0.06**; capping the feedback contribution at **0.05** gives the newcomer
the top slot on the round it arrives and every one of the **26** after. The
cap encodes a policy: feedback may break ties and may not overturn evidence.
It also sets a design rule — the cap has to be smaller than the smallest
score difference you want feedback *not* to erase, which means you have to
know your scorer's scale.

**FINDING: exploration alone fixes discovery, not ranking.** Forcing an
unrated record into the top slot every **5th** round surfaces the newcomer at
round **25** — and it wins exactly **1** of the remaining **26**. One
impression makes it rated, after which it re-enters a ranking the incumbent's
**24** thumbs-up still dominate. Exploration answers "was it ever seen"; only
the cap answers "may feedback outvote evidence". The two are complements, not
alternatives.

### 5 — what the benchmark numbers measure, and what the toy cannot

*Cites "Benchmark numbers".*

**The numbers are a statement about fusion, and the toy reproduces the
architecture without the thing being measured.**

The lesson reports LoCoMo **91.6**, LongMemEval **93.4**, BEAM 1M **64.1**,
with full-context 128k, flat vector and flat KV baselines all losing by 10+
points — and adds the right caveat: "benchmarks alone don't justify choice —
operational shape does". Porting the toy to the `mem0` client and rerunning
20 queries would compare two things that differ in more than one variable, so
the useful part of the exercise is being explicit about *which* variable.

**Three differences, in descending order of how much they would move a
score.**

1. **The retrieval function.** The toy's vector arm is Jaccard over token
   sets. Exercise 1 measures the gap this leaves: **0/20** recall@10 on
   queries that differ from their target by a word ending, where even a
   non-semantic trigram hash reaches **19/20**. A real embedding closes the
   morphological gap *and* the paraphrase gap; the second is the one the
   benchmarks are built on, since LoCoMo and LongMemEval ask questions in the
   user's words rather than the transcript's. Any comparison on 20 queries
   would be dominated by this single term.
2. **The fusion, which the toy implements and then bypasses.** The lesson's
   own design is relevance + importance + recency, and in the shipped code
   the KV arm enters at a *constant* relevance of 0.4 regardless of the query
   (exercise 1), recency is worth **0.0016** after a week (exercise 1), and
   `importance` is a float the writer chose. So on any corpus older than a
   day the fusion reduces to relevance-or-0.24-plus-importance. The hosted
   client's advantage here is not a better weighting, it is that all three
   dimensions still carry signal.
3. **Temporal correctness, which is scored and which the toy gets wrong.**
   LongMemEval is explicitly long-horizon episodic memory, and exercise 2
   measures the shipped graph answering **1** of **3** as-of probes and
   returning the empty list for a time when the fact *was* true. Exercise 3
   measures the same failure from the write side: the graph invalidates
   Berlin and the vector arm keeps returning it. A benchmark with updated
   facts — which is what "long-horizon" means — would charge the toy for both
   on the same question.

**What a fair comparison would look like.** Hold the corpus and the 20
queries fixed, and swap one arm at a time: shipped scorer, then trigram
cosine, then a hosted embedding, with the rest of the fusion unchanged.
Exercise 1's harness is already that shape, and the two arms it runs differ
by exactly one function. Reporting "mem0 scored higher" without that
decomposition tells you the hosted product is better assembled, which was
never in question, and not which of the three stores earned it.

**What the toy is still good for.** The architecture is right, and the seams
are where the interesting failures live — which is why the four code
exercises above find real bugs in code that has no model in it at all. The
benchmark numbers justify the *shape*; the exercises show the shape is not
self-enforcing.
