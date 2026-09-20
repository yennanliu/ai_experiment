<!-- generated:start -->
# 13-tools-and-protocols / 12-mcp-roots-and-elicitation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/) · upstream spec
`phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/docs/en.md`

```bash
uv run demo practice run 12-mcp-roots-and-elicitation --ex 1
uv run demo explain 12-mcp-roots-and-elicitation --ex 1
uv run pytest demos/phases/13-tools-and-protocols/12-mcp-roots-and-elicitation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace the in-memory replay store with SQLite. Use one transaction to claim the nonce and de… | code | T0 | `ex01_the_lock_was_per_process_so_two_processes_had_no_lock.py` |
| 2 | Add `url` capability negotiation and an out-of-band setup flow. Keep third-party credentials… | code | T0 | `ex02_the_schema_declares_the_fields_and_refuses_none_of_the_others.py` |
| 3 | Replace the in-memory note map with a temporary SQLite database. Re-check authorization and c… | code | T0 | `ex03_the_token_binds_the_arguments_and_cannot_bind_the_world.py` |
| 4 | Add a symbolic-link policy for a real filesystem implementation. Explain why URI lexical cont… | code | T0 | `ex04_a_string_check_cannot_see_a_link_because_the_link_is_a_fact.py` |
| 5 | Design a 2025-11-25 adapter that maps modern MRTR handler output to legacy server-initiated e… | code | T0 | `ex05_the_eras_disagree_about_who_owns_the_request_id.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code. The scaffold classified
exercise 4 as prose on the word "Explain", but it has a code deliverable — a
symlink policy for a real filesystem — so it is filed as code, with the
explanation demonstrated in a temporary directory rather than argued.

Three of the five replace an in-memory structure with SQLite, and the pattern
underneath them is the same: **the lesson's guarantees hold inside one
process, at one instant, over strings.** Each exercise moves one of those
three assumptions and measures what breaks.

### 1 — the lock was per-process, so two processes had no lock

**ANSWER: one shared database, one transaction, exactly one commit.** Two
servers replaying one token delete **1** note; the loser gets
`requestState was already consumed`.

**FINDING: the in-memory store does not fail this test — it does not
participate.** Two `ReplayStore` instances both commit the same nonce, **2**
deletions for **1** token, because `_consumed` and `_lock` are per-instance.
Within one instance the same replay is correctly refused, so the defect is
invisible to any single-process test.

**FINDING: claiming and mutating must share the transaction.** An operation
that raises inside the block leaves **0** rows and the nonce is claimable
again. Committing the claim separately would consume a nonce for an operation
that never happened — unretryable by construction.

**FINDING: the capacity limit has to be re-implemented.** `max_entries=1`
reproduces `-32023`, and expiry eviction becomes a `DELETE ... WHERE
expires_at <= ?` in the same transaction. Both were invariants of a dict that
the design never stated.

### 2 — the schema declares the fields and refuses none of the others

**ANSWER: `url` is negotiated separately, and the reply carries a signal not a
secret.** Form-only is refused **-32021**; url gets a setup URL; the retry
carries **0** credential fields while the server's vault holds **1**.

**FINDING: the lesson's negotiation reads absence of detail as support.**
`supports_form_elicitation({"elicitation": {}})` is **True** — an empty dict
short-circuits to yes. The new mode has to be checked by presence of its own
key, which is what makes the `-32021` reachable at all.

**FINDING: the URL carries a handle, not a secret.** A `token_urlsafe` handle,
bound into the sealed state, and **0** characters of the credential. A URL
recovered from browser history is worth nothing without the token, which the
browser never saw.

**FINDING: nothing refuses an undeclared field.** A client that also sends
`api_key` has it delivered intact — **2** keys where **1** was declared.
Keeping credentials out needs the server to *reject* keys it did not ask for,
not merely to avoid reading them.

### 3 — the token binds the arguments and cannot bind the world

**ANSWER: notes in a temporary database, both re-checks inside one
transaction.** 5 rows → 4.

**FINDING: containment can lapse between rounds, and `candidateIds` does not
notice.** Moving a confirmed candidate outside the workspace after round one
is refused at delete time, with **5** rows still present. The token records
which notes *were* eligible — a fact about the past.

**FINDING: authorization can lapse too, and the token has no field for it.**
`requestState` binds principal, method and an arguments digest — all
properties of the request. Grants are a property of the server, so no sealed
token could have carried this.

**FINDING: the transaction is what makes a refused check harmless.** Checking
after a committed delete would be a check on a row that no longer exists.

### 4 — a string check cannot see a link, because the link is a fact

**WHY LEXICAL CONTAINMENT CANNOT WORK:** `uri_within_workspace` and
`_normalized_uri_parts` make **0** filesystem calls — they are `normpath`,
`unquote` and `commonpath`. They decide what a path *spells*; a symlink is the
filesystem's decision about what a name *means*. `normpath` even cancels `..`
textually, which is right for a pure path and wrong the moment a component is
a link.

**ANSWER: the escaping link answers `True` lexically and `False` resolved.**
The policy is `realpath` both sides, then ask the same question.

**FINDING: the policy has to resolve the root as well.** With the workspace
reached through a link, a legitimate inside file answers `False` against the
unresolved root. Resolving one side makes the question ill-posed in the other
direction.

**FINDING: resolve-then-check is still time-of-check to time-of-use.**
Repointing the link after the check leaves the verdict stale. The durable fix
is opening the file and checking the handle — `O_NOFOLLOW`, or
`st_dev`/`st_ino` after the fact — not a better predicate.

### 5 — the eras disagree about who owns the request id

**ANSWER: the adapter round-trips and the note is deleted.** **3** messages on
the legacy wire, **2** on the modern one.

**FINDING: the eras differ by who owns the id.**

| era | allocates the id | correlates by |
|---|---|---|
| modern | the client (retry) | `requestState` |
| legacy | the server (elicitation) | the id itself |

The adapter mints an id the modern server never sees, so the two numbering
spaces never have to agree.

**FINDING: isolation costs the adapter the state the modern design removed.**
The server's fields are `authorized_workspaces`, `notes`, `replay_store` —
none per in-flight call. The adapter must hold the `requestState` between the
legacy request and its response, because the legacy exchange has nowhere to
carry it. Re-adding the era re-adds per-connection state, in the one component
allowed to have it.

**FINDING: the isolation is checkable.** **0** `elicitation/create` messages
reach the modern transcript, and the lesson's own `run_mrtr` still completes
without the adapter present.
