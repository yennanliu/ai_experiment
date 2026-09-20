<!-- generated:start -->
# 13-tools-and-protocols / 15-mcp-security-tool-poisoning

Solutions to all 6 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/15-mcp-security-tool-poisoning/) · upstream spec
`phases/13-tools-and-protocols/15-mcp-security-tool-poisoning/docs/en.md`

```bash
uv run demo practice run 15-mcp-security-tool-poisoning --ex 1
uv run demo explain 15-mcp-security-tool-poisoning --ex 1
uv run pytest demos/phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Bind the authenticated principal and current authorization decision to the sealed MRTR state,… | code | T0 | `ex01_the_token_is_bearer_because_the_gateway_has_no_caller.py` |
| 2 | Replace the in-memory replay store with a persistent conditional insert and prove two process… | code | T0 | `ex02_the_claim_is_the_rowcount_and_a_select_first_is_not_one.py` |
| 3 | Inject a failure after replay claim but before a simulated export. Define and test the transa… | code | T0 | `ex03_the_two_orders_lose_different_things_and_idempotence_loses_neither.py` |
| 4 | Change a tool's `inputSchema` without changing its description. Confirm whole-descriptor pinn… | code | T0 | `ex04_the_scanner_reads_prose_and_the_digest_reads_the_whole_thing.py` |
| 5 | Add a policy that refuses public caching when `tools/list` differs by principal. | code | T0 | `ex05_the_listing_is_already_private_and_does_not_yet_need_to_be.py` |
| 6 | Model an older server behind the gateway. Put all handshake and session behavior behind an ex… | code | T0 | `ex06_the_gateway_keeps_the_session_so_the_server_does_not_have_to.py` |
<!-- generated:end -->

## Answers

All six are T0 and stdlib, and all six ship code.

The gateway's job is to distrust what it forwards, and the six exercises keep
finding the same shape of gap: **the check exists, and the thing it checks is
not the thing that varies.** A seal that binds the request but not the
requester; a pin over prose rather than the descriptor; a cache scope that is
right for the wrong reason.

### 1 — the token is bearer, because the gateway has no caller

**ANSWER: `principal` and `decision` in the sealed state.** Mallory replaying
alice's token is refused; **1** export from **2** retries of one token.

**FINDING: the shipped state binds the request and not the requester.** Its
**6** fields are `arguments`, `expiresAt`, `issuedAt`, `nonce`, `purpose`,
`tool` — and `handle` takes `(self, body, headers)`. The token is a bearer
credential.

**FINDING: binding the decision records it and does not re-check it.** A token
sealed while alice was authorized still says `allow` after revocation. The
seal is a snapshot of what was decided, not a substitute for deciding.

**FINDING: the three bindings fail independently.** Changed arguments → the
lesson's own digest check; changed caller → the principal; revoked grant → the
re-check. No one of them subsumes another.

### 2 — the claim is the rowcount, and a SELECT first is not one

**ANSWER: `INSERT ... ON CONFLICT DO NOTHING`, one row, one export.**

**FINDING: the rowcount is the claim.** Through the same interleave — both
check, then both write:

| strategy | winners |
|---|---:|
| conditional insert | 1 |
| check then insert | 2 |

The constraint alone is not what makes it safe; it is that the database
decides and reports in one statement.

**FINDING: the in-memory store never enters the race.** Two `ReplayStore`
instances, two dicts, two locks, **2** exports.

**FINDING: the gateway builds its own store unless one is passed.** The
default is the unsafe configuration.

### 3 — the two orders lose different things, and idempotence loses neither

**THE RULE: the nonce is the idempotency key, written with the export.**

**ANSWER:** each order crashed in its own window, then retried:

| order | crash point | exports |
|---|---|---:|
| claim-first | after the claim | 0 |
| operate-first | after the export | 2 |
| operate-first + nonce key | after the export | 1 |

**FINDING: the lesson's own order is operate-first, so it is at-least-once.**
`claim_and_consume` calls `operation()` and records the nonce afterwards. The
right trade only if the export is idempotent, which nothing says it is.

**FINDING: the key must be the nonce.** Keyed on the arguments, two
legitimately confirmed exports collapse into **1**. The nonce is
per-confirmation, which is the granularity a user approved.

### 4 — the scanner reads prose and the digest reads the whole thing

**ANSWER: `scan_description` finds **0**; the pin reports `rug_pull`.** The
tool then drops out of `tools/list`, 3 visible → 2.

**FINDING: pinning the description alone would have missed it.** All **5**
injection patterns match **0** times and the description's digest is
unchanged. A pin over the field a human reads cannot see a change to the field
a model is constrained by.

**FINDING: the digest is canonical.** `sort_keys=True`, so reordering is not a
change — the pin distinguishes content from spelling.

**FINDING: one finding can never block anything.** `_visible_tools` blocks
keys containing a dot; `shadowing` is reported under the bare name `search`.
Emitted on every scan, filters nothing.

### 5 — the listing is already private and does not yet need to be

**ANSWER: the policy permits `public` when the listings match and refuses when
they differ.** Shipped gateway → permit; per-principal grants → refuse.

**FINDING: today's `private` is correct and unearned.** `_visible_tools` takes
`(self)` alone, so both principals get identical listings — and the result
already declares `private`.

**FINDING: `server/discover` stays public under the same policy.** Scope is
decided per response, not per server — one gateway, two verdicts.

**FINDING: a catalog-wide block is not principal-dependence.** A rug pull
hides a tool from everyone, the listings still match, and `public` is still
permitted. What forbids public caching is variation across *callers*.

### 6 — the gateway keeps the session, so the server does not have to

**ANSWER: a `2025-11-25` adapter that handshakes once and holds the session.**
Cold `tools/list` → `-32002`; after `initialize`, three tools.

**FINDING: the session is one field, and it is on the adapter.** The gateway's
**4** fields are `approved`, `catalog`, `replay_store`, `secret` — none
per-connection. Re-adding the era re-adds exactly the state the modern design
removed, in the one component allowed to have it.

**FINDING: the handshake costs a message the modern path does not have.**
**2** legacy requests for one answer against **1**. The extra round trip *is*
the session.

**FINDING: the branch is chosen by declaration, not by failure.** **0** modern
requests reach the adapter. Routing on "the modern call failed, try legacy"
would start a legacy handshake after any transient error.
