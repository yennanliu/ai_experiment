<!-- generated:start -->
# 13-tools-and-protocols / 06-mcp-fundamentals

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/06-mcp-fundamentals/) · upstream spec
`phases/13-tools-and-protocols/06-mcp-fundamentals/docs/en.md`

```bash
uv run demo practice run 06-mcp-fundamentals --ex 1
uv run demo explain 06-mcp-fundamentals --ex 1
uv run pytest demos/phases/13-tools-and-protocols/06-mcp-fundamentals
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Change one request's protocol version to `2027-01-01`. Confirm the error code is `-32022` and… | code | T0 | `ex01_the_version_gate_is_per_request_and_a_notification_skips_it.py` |
| 2 | Remove `io.modelcontextprotocol/clientCapabilities` from the second request. Confirm the serv… | code | T0 | `ex02_there_is_no_state_to_reuse_so_the_question_cannot_be_answered_wrong.py` |
| 3 | Reverse the in-memory tool registry. Confirm `tools/list` still returns the same deterministi… | code | T0 | `ex03_the_order_is_deterministic_because_the_names_are_unique.py` |
| 4 | Change `cacheScope` from `public` to `private`. Explain which authorization contexts may reus… | code | T0 | `ex04_private_is_the_default_that_nothing_in_the_lesson_uses.py` |
| 5 | Add an optional `clientInfo` omission test. The request should remain valid because client id… | code | T0 | `ex05_the_optional_field_is_the_only_one_validated_for_shape.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code.

This is the first lesson in the phase whose exercises all predict correctly —
`-32022` fires, capabilities are not reused, the order holds, `clientInfo` is
optional. So the findings sit one layer under each: **every guarantee the
exercises confirm turns out to be conditional on something the shipped fixture
happens to satisfy.**

### 1 — the version gate is per-request, and a notification skips it

**ANSWER: `-32022`, with `data` naming both sides** —
`{"requested": "2027-01-01", "supported": ["2026-07-28"]}`. The client learns
what it asked for and what it may ask next in one round trip, with no handshake
to have failed.

**FINDING: the same change on a notification is accepted silently.** Strip the
`id` and `dispatch` returns **`None`** before `validate_request` is ever called.
The version gate is a property of messages that expect a reply, not of the
protocol.

**FINDING: the gate is per-request because nothing survives a request.** All
three id-bearing methods reject the future version identically and all three
accept the supported one. Accepting one request does not make the next cheaper
to check — the per-message cost of having no session.

**FINDING: `-32022` is a private code inside JSON-RPC's reserved block**
(`-32768..-32000`) and is none of the **5** codes JSON-RPC defines. The lesson's
other errors are standard; version negotiation had no standard code to take.

### 2 — there is no state to reuse, so the question cannot be answered wrong

**ANSWER: `-32602`, and the first request's success does not help.**

**FINDING: there is nowhere for the capabilities to have been kept.** The module
holds **0** mutable containers that outlive a call, and `validate_request` reads
only its argument. The server does not *decline* to reuse state — it has none.
That is what "no protocol session" means as an implementation property rather
than a promise.

**FINDING: the same is true of the version and the client identity.** All **3**
`_meta` keys are re-established per message; a client cannot amortise any of it.

**FINDING: the empty capability object is accepted**, so the check is presence,
not content. Nothing reads a capability afterwards — a client advertising
nothing is indistinguishable from one advertising everything.

### 3 — the order is deterministic because the names happen to be unique

**ANSWER: `['notes_list', 'notes_search']` both ways**, because `dispatch`
returns `sorted(TOOLS, key=name)` and the registry's own order never reaches the
wire.

**FINDING: the determinism comes from the names being unique, not from the
sort.** `sorted` is stable, so equal keys keep input order. Add a second
`notes_list`:

| | names | descriptions |
|---|---|---|
| forward | `notes_list, notes_list, notes_search` | `List note titles.`, `A second tool…` |
| reversed | *identical* | **`A second tool…`, `List note titles.`** |

A client diffing `tools/list` by name sees no change and gets a different tool.

**FINDING: nothing in the lesson would catch that.** No uniqueness check, no
deduplication, and `tools/call` dispatches on **2** hard-coded name comparisons
rather than a lookup — so a duplicate is unreachable by call and invisible by
list. Lesson 13.05's linter has the rule; this lesson does not import it.

**FINDING: only one of the three methods sorts anything.** `dispatch` calls
`sorted` once. The guarantee covers the one method where it matters.

### 4 — `private` is the default that nothing in the lesson uses

**ANSWER: `public` means any authorization context; `private` means exactly
one.** A `public` response is a function of the server alone, so a shared cache
may serve it to every caller. A `private` one is a function of the caller's
authorization, so it may be reused only for requests carrying that same
authorization — a per-principal cache, never a shared one. Backwards, it leaks.

| method | `ttlMs` | `cacheScope` |
|---|---:|---|
| `server/discover` | 3,600,000 | `public` |
| `tools/list` | 30,000 | `public` |
| `tools/call` | — | — |

**FINDING: the split is not arbitrary.** The two cacheable methods are pure
functions of the server; the third is a function of the request.

**FINDING: `private` is `complete_result`'s default and no call site reaches
it** — both cacheable sites pass `public` explicitly. The default protects a case
the lesson never constructs.

**FINDING: and nothing in the response says what the caller was.** With no
session and no principal in `_meta`, a `private` response would be cacheable
against *what*? `clientInfo` is a name and a version, not an authorization
context. Making `tools/call` cacheable would need a key the protocol does not
carry — which is why it carries no scope instead.

### 5 — the optional field is the only one validated for shape

**ANSWER: omitting `clientInfo` leaves the request valid** on all three methods.

**FINDING: the optional field is the only one whose shape is validated.**
`protocolVersion` must be a `str` and `clientCapabilities` a `dict` — presence
checks. `clientInfo` is checked for being a dict *and* for `name` and `version`
both being strings: **5 of 5** malformed values give `-32602`, while
`clientCapabilities: {}` sails through. The field that may be absent is the
field inspected hardest.

**FINDING: that asymmetry is an accident of what is consulted.** Nothing reads a
capability, so nothing can be broken by a wrong one. `clientInfo` is the only
`_meta` key whose value is destined for anything but a presence check — a log
line, a rate-limit bucket, an audit trail — so its shape is the only one that can
corrupt something downstream.

**FINDING: and "recommended" is not enforced anywhere.** Its absence is accepted
on every method, while the server's reply always carries `serverInfo`. The
recommendation is one-directional in the code as well as the spec.
