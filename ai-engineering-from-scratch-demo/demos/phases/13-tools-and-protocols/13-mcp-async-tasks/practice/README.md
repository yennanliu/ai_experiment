<!-- generated:start -->
# 13-tools-and-protocols / 13-mcp-async-tasks

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/13-mcp-async-tasks/) · upstream spec
`phases/13-tools-and-protocols/13-mcp-async-tasks/docs/en.md`

```bash
uv run demo practice run 13-mcp-async-tasks --ex 1
uv run demo explain 13-mcp-async-tasks --ex 1
uv run pytest demos/phases/13-tools-and-protocols/13-mcp-async-tasks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a second outstanding input key. Send a partial `tasks/update` and prove the task remains… | code | T0 | `ex01_the_update_reads_one_key_by_name_so_partial_is_not_expressible.py` |
| 2 | Add tenant ownership to the store and reject a valid task id presented by the wrong authentic… | code | T0 | `ex02_isolation_is_a_filter_over_a_shared_namespace_and_one_path_skips_it.py` |
| 3 | Add a worker lease with expiry. Demonstrate that two service instances cannot complete the sa… | code | T0 | `ex03_the_lease_has_to_be_rechecked_at_the_commit_not_only_at_the_claim.py` |
| 4 | Implement a POST-response SSE adapter for `subscriptions/listen`. Do not add GET, `Last-Event… | code | T0 | `ex04_the_ack_drops_silently_where_the_notification_raises.py` |
| 5 | Add expiry cleanup. Distinguish an expired task from a malformed task id without leaking cros… | code | T0 | `ex05_expired_may_only_be_said_to_the_owner.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code.

The tasks extension moves work off the request, and every one of these
exercises finds the same consequence: **a guarantee that held inside one
request, in one process, over one in-memory dict has to be rebuilt once the
work outlives all three.** Two of the five find the shipped code already
losing that bet.

### 1 — the update reads one key by name, so partial is not expressible

**ANSWER: two keys, and the task stays `input_required` until both are
answered.**

**FINDING: the shipped update transitions on one key of two, and discards the
other.** `TaskService.tasks_update` matches `responses.get("approve_outline")`
and then clears `input_requests` wholesale — so the unanswered `choose_format`
disappears with **0** answers. Iterating the outstanding set is the difference
between the requirement holding and not.

**FINDING: `tasks/update` says nothing about what it applied.** An unknown
key, a malformed answer and a correct one return **1** distinct bare
`complete()` while leaving the task in **2** different states. The outcome is
learned only by polling `tasks/get`.

**FINDING: the key-reuse guard is unreachable.** `advance_worker` raises on a
reused key, but only at `stage == 0`, and it sets `stage = 1` immediately. The
stage gate, not the guard, prevents reuse.

### 2 — isolation is a filter over a shared namespace, and one path skips it

**ANSWER: a foreign principal gets `task not found`, byte-identical to a
nonexistent id** — while a same-tenant co-worker reads the task normally.

**FINDING: the ownership check lives in the service, and `advance_worker`
skips it.** It takes `(self, task_id)` — no principal at all — and calls
`store.get` directly, so it will drive any tenant's task through its state
machine.

**FINDING: one directory holds every tenant and `reload()` loads them all.**
**3** files from **3** principals into one dict keyed by id. A missed
predicate is a full cross-tenant read, not a narrow bug.

**FINDING: what protects tenants today is entropy, not authorization.**
`uuid4().hex[:12]` is **48** bits. Unguessable is a useful property and not
the same one — a tenant-keyed store refuses a correctly guessed id.

### 3 — the lease has to be re-checked at the commit, not only at the claim

**ANSWER: without a lease two instances both complete; with one, exactly one
does.** Two services over one directory both transition the task, because each
holds its own `store.tasks` copy.

**FINDING: claiming is not enough, because the lease can move while you
work.** A claims → A's lease expires → B claims and completes → A's commit is
refused. The expiry stops a crashed holder blocking forever, and this window
is its price.

**FINDING: the reference is safe in one process and only there.** A second
`advance_worker` on the same instance transitions nothing, because the guard
reads a copy rather than the record.

**FINDING: the lease is state the extension has nowhere to put.** A `Task` has
**15** fields and its wire form **7**, none naming a worker — so adding the
lease changes no protocol message.

### 4 — the ack drops silently where the notification raises

**ANSWER: one POST, one SSE body, and none of the three prohibited things.**
**4** frames for **2** tasks, each tagged with the subscription id (the
terminal one by merging into the `_meta` that `complete()` writes last), **0**
SSE `id:` fields, no session header, one verb.

**FINDING: the two helpers disagree about an unowned task id.**

| helper | foreign id |
|---|---|
| `subscription_acknowledgement` | dropped from `taskIds`, no error |
| `task_notification` | raises `-32602 task not found` |

So the adapter must follow the ack, and the ack is the authorization boundary.

**FINDING: the acknowledgement is the only place the narrowing is visible.**
2 requested, 1 accepted, 3 frames — and nothing says the other was rejected.

**FINDING: the prohibitions cost nothing, because the tasks are the durable
thing.** Discarding the stream and polling `tasks/get` recovers the identical
state. There is nothing to resume, only something to re-read.

### 5 — "expired" may only be said to the owner

**ANSWER: three outcomes, and a stranger never sees the third.**

| input | answer |
|---|---|
| malformed id | `malformed task id` |
| owner's expired task | `task expired` |
| unknown / foreign / **foreign expired** | `task not found` |

That last row is the one that matters — saying "expired" there would confirm
the id exists.

**FINDING: the malformed check costs 0 store reads, which is why it leaks
nothing.** Any distinction drawn *after* a lookup is about a record somebody
owns, and can only be reported to them. That fixes the order: shape,
existence, ownership, expiry.

**FINDING: the lesson collapses all three into one message.** Safe, and unable
to tell a client its own id is the wrong shape — the one fault it could fix
unaided.

**FINDING: cleanup destroys the evidence "expired" depends on.** After the
sweep even the owner gets `task not found`. A durable expired answer needs a
tombstone keeping the owner and nothing else.
