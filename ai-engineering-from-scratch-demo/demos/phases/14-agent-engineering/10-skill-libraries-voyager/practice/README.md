<!-- generated:start -->
# 14-agent-engineering / 10-skill-libraries-voyager

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/10-skill-libraries-voyager/) · upstream spec
`phases/14-agent-engineering/10-skill-libraries-voyager/docs/en.md`

```bash
uv run demo practice run 10-skill-libraries-voyager --ex 1
uv run demo explain 10-skill-libraries-voyager --ex 1
uv run pytest demos/phases/14-agent-engineering/10-skill-libraries-voyager
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a dependency-cycle detector to `compose()`. What happens when skill A depends on B which… | code | T0 | `ex01_a_cycle_returns_an_order_and_the_order_depends_on_the_ask.py` |
| 2 | Implement per-skill version pinning. When a parent skill composes child `crafting@1`, a refin… | code | T0 | `ex02_refinement_mutates_the_object_every_parent_is_holding.py` |
| 3 | Replace token-overlap retrieval with sentence-transformers embeddings (or a BM25 stdlib impl)… | code | T0 | `ex03_only_the_description_is_indexed_and_the_code_is_the_api.py` |
| 4 | Add a "curriculum" agent: given the current library and a domain description, propose 5 missi… | code | T0 | `ex04_the_only_machine_readable_gap_is_a_dependency_that_does_not_exist.py` |
| 5 | Read Anthropic's Claude Agent SDK skill docs. Port the toy library to the SDK's skill schema.… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 reads the Claude Agent
SDK's skill docs, so it is answered in prose.

The recurring finding is that **the library's safety properties are absences
rather than checks**. There is no `compose()` — the ordering step is
`topo_order`, whose `visited` set makes a dependency cycle terminate instead
of raise, so a cycle produces a plausible order that depends on which skill
you asked for. There is no version in `depends_on`, and `register` edits the
existing `Skill` object in place, so a refinement is not a new version a
parent could decline — it is the same identity with different behaviour.
There is no failure record on `Skill`, so a weekly curriculum agent cannot
see what was tried.

The second thread is that **`search` reads `description` and nothing else**.
`Skill` also carries `code` and `tags`, and the thing a caller usually knows
is the function they want to call. Exercise 3 separates the two variables and
finds the field set worth **20** points of retrieval@5 and the scorer worth
**0** on that query shape — the swap the exercise asks for is the less
important half of it.

What holds: the refinement loop itself. Register, run, read the failure,
register again — the demo's v1-to-v2 pickaxe story works exactly as written,
and `history` and `version` do advance. They just cannot be used to go back.

### 1 — a cycle returns an order, and the order depends on the ask

**ANSWER: a detector over the lesson's own walk, and it must be an error.**
Across a library holding one valid DAG, one two-skill cycle and one
self-dependency, the detector flags **2** and leaves the DAG at
`['leaf', 'mid', 'root']`.

**Error, not warning — because the shipped order is ambiguous.**
`topo_order("skill_a")` returns `['skill_b', 'skill_a']` and
`topo_order("skill_b")` returns `['skill_a', 'skill_b']`. Same library, same
cycle, opposite answers. A warning would imply there is a correct order being
approximated; there is not one.

**FINDING: `execute` runs the cycle without complaint.** Composing the
cyclic pair produces **2** log lines and `failed=False`. Whichever skill the
caller entered at runs *last*, so the entry point silently decides which half
of the cycle observes the other's effects on the shared context.

**FINDING: a self-dependency is invisible.** A skill depending on itself
composes to **1** entry, with no repetition and no warning. The `visited` set
absorbs it exactly as it absorbs a real cycle — so the two bugs are
indistinguishable from outside, and neither looks like a bug.

**FINDING: a missing dependency is caught late and as a run failure.**
`topo_order` returns the unknown name, and `execute` notices on arrival:
`missing skill: nowhere`, `failed=True`, after the earlier skills have
already mutated the context. Composition-time validation would catch it
before anything runs — which is exactly what exercise 4 turns into a weekly
report.

### 2 — refinement mutates the object every parent is holding

**ANSWER: `name@version` in `depends_on`, resolved against an archive.**
After `crafting` is refined to v2 the pinned parent still logs
`ran crafting v1: crafted with 3 ore` while the unpinned parent, composed
from the same library at the same instant, logs `v2: crafted with 5 ore`.
The archive is the new part: without somewhere to keep v1, a pin has nothing
to point at.

**FINDING: the refinement edits the object in place.** `register(dedup=True)`
assigns `existing.fn = skill.fn` on the `Skill` already in the dict, so
before and after are the *same instance* at versions 1 and 2. This is the
lesson's own "composed-skill drift" pitfall at its sharpest: the parent does
not pick up v2 because it re-resolved a name, it picks it up because the
object it names changed under it.

**FINDING: `history` is documentation, not a rollback.** It holds **1** entry
of type `str` — the old `code` — and **0** callables. `fn` is what runs, and
no copy of it survives, so after a refinement the previous behaviour is
unrecoverable from the library. A version number that cannot be checked out
is a changelog.

**FINDING: `dedup=False` rewinds the version counter.** Registering the same
name a third time with `dedup=False` replaces the entry with a fresh `Skill`
at version **1** and **0** history entries, after it had reached version 2.
Two supported ways to write a skill, one of which makes v3 look like v1 — and
the lesson's own dedup advice is about the *other* direction (the same skill
added ten times), which this flag is how you get.

### 3 — only the description is indexed, and the code is the API

**ANSWER: retrieval@5 over a 50-skill library is 20/20 for BM25 across all
three fields and 0/20 for the shipped search.** Each query names an
identifier from a skill's `code` and never from its prose, so the shipped
scorer finds **0** candidates and returns an empty list — not a wrong answer,
no answer.

**FINDING: the scorer is not the reason.** BM25 over descriptions only also
scores **0/20**; plain token overlap over all three fields scores **20/20**.
The field set moves the number by **20** and the scorer by **0**. That is
worth holding onto before reaching for sentence-transformers: an embedding of
the wrong field is still an embedding of the wrong field.

**FINDING: where the scorer does matter is length.** Two skills that both
contain every query term score **0.667** and **0.1** under the shipped
Jaccard — a **6.7x** penalty purely for having a longer description — against
**2.0x** under BM25. Jaccard divides by the union, which grows with the
document; BM25's `b` normalises against the corpus mean. In a skill library
the longer descriptions are the well-documented skills.

**FINDING: `tag_filter` fails silently.** It is an exact `in skill.tags`
test, so `tag_filter="crafting"` against skills tagged `craft` returns **0**
results while `craft` returns **5**. The lesson recommends supplementing
retrieval with tag filters as the library grows; a filter whose typo mode is
"empty result set" makes that advice load-bearing and unverifiable at once.

### 4 — the only machine-readable gap is a dependency that does not exist

**ANSWER: five proposals, the library's own evidence first.** Over a
**6**-skill library and an **8**-capability domain, the agent proposes the
**2** dangling dependencies other skills already name, then the **3**
highest-priority uncovered capabilities. All five are names no skill has.

**FINDING: a dangling dependency is the only self-reported gap.** Executing
every skill and reading the logs finds the same **2** names that reading
`depends_on` finds directly — which means today they are discoverable only by
running the library and parsing prose. A curriculum agent that reads the
declarations turns exercise 1's run-time failure into a weekly report, at no
cost.

**FINDING: the library cannot say what failed.** `Skill` has **8** fields and
**0** record an attempt; `execute` reports `failed` and `log` into a context
dictionary the caller discards. The Voyager loop the lesson describes —
success, error, self-verification failure, rewrite — has three signals, and
this library persists none of them. So a weekly agent proposes the same skill
every week until someone writes it, and cannot distinguish "not attempted"
from "attempted and abandoned".

**FINDING: duplicate detection is by name, because retrieval reads prose.**
Proposing `extract_minerals` against a library containing `mine_ore` passes
the name check and returns **0** search candidates — the two share no token.
This is the lesson's "skill library rot" pitfall arriving through the front
door: dedup-on-write cannot catch what retrieval cannot match, so the
curriculum agent is a rot generator unless the matching improves first
(exercise 3).

### 5 — the SDK moves discovery from a scorer to a contract

*Cites "Skill retrieval".*

**The port replaces a ranking problem with a declaration problem, and that
is the whole change to discoverability.**

The lesson's retrieval story is four steps: embed the task, query the library
for top-k similar skills, retrieve the primitives, compose. Every one of
those steps is a *guess with a score*, and exercise 3 measures what the guess
is worth here — **0/20** retrieval@5 when the caller knows the function name
rather than the prose, because `search` indexes `description` and nothing
else. The Claude Agent SDK's skill surface inverts this: a skill is a
directory with a frontmatter `name` and `description` that the model reads
directly, plus a body and optional bundled files that are loaded only when
the skill is selected. Discovery is a short, authored list in context, not a
nearest-neighbour search over an index.

**Four things change, in order of how much they move this library's
measurements.**

1. **The description becomes a contract with a reader, not a document to be
   scored.** In the toy, `description` is the only indexed field and its
   *length* is charged against it — exercise 3's **6.7x** Jaccard penalty
   means a well-documented skill ranks below a terse one containing fewer of
   the query's terms. Under the SDK the description's job is to tell a model
   *when to use this*, and length is a cost in tokens rather than a penalty
   in rank. That makes the failure mode different and inspectable: a skill
   that is never selected has a description you can read, not a score you
   have to debug.
2. **Progressive disclosure replaces top-k.** The toy retrieves `top_k=3` and
   composes from whatever comes back; anything ranked fourth is invisible and
   nothing says so. The SDK's shape is name-and-description always in
   context, body loaded on selection, bundled files loaded on demand — so the
   *set* of candidates is complete and bounded, and the model's choice is
   visible. Exercise 3's failure — an empty result set that looks identical
   to "no such skill" — cannot occur, because there is no result set.
3. **Dependencies stop being a namespace.** `depends_on: tuple[str, ...]`
   resolves names against a mutable dict, which is how exercise 1 gets a
   cycle with two valid orders and exercise 2 gets a parent upgraded without
   consenting. A skill packaged as a directory of files has its
   dependencies as files, resolved by the filesystem, versioned by whatever
   versions the directory. That does not make composition correct — it makes
   composition *someone else's* correctness problem, which is the right trade
   for a library that had none.
4. **What gets harder: the curriculum loop.** Exercise 4's gap signal is
   `depends_on` names that resolve to nothing — a machine-readable statement
   the library makes about itself. A prose-and-files skill has no equivalent
   field, so "what are we not covering" goes back to being a human review,
   exactly as the lesson says teams do it. The SDK buys discoverability for
   the *model* and gives up the one structured signal a weekly agent could
   have read.

**The honest summary:** discoverability improves because selection moves from
an unobservable score to an authored list the model sees, and because the
fields that carry meaning are the fields that are read. What does not improve
is anything exercises 1, 2 and 4 measured — cycles, pinning and gap analysis
are all properties of a dependency graph, and the SDK's answer is to not have
one.
