<!-- generated:start -->
# 13-tools-and-protocols / 09-mcp-transports

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/09-mcp-transports/) · upstream spec
`phases/13-tools-and-protocols/09-mcp-transports/docs/en.md`

```bash
uv run demo practice run 09-mcp-transports --ex 1
uv run demo explain 09-mcp-transports --ex 1
uv run pytest demos/phases/13-tools-and-protocols/09-mcp-transports
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Remove `Mcp-Method` from a POST. Confirm HTTP `400` and error `-32020`. | code | T0 | `ex01_a_missing_header_and_a_wrong_one_are_the_same_answer.py` |
| 2 | Send matching header and body version `2027-01-01`. Confirm HTTP `400`, error `-32022`, and e… | code | T0 | `ex02_matching_is_the_word_that_selects_the_only_path_to_32022.py` |
| 3 | Send a Base64 sentinel `Mcp-Name` for a non-ASCII resource URI. Confirm the decoded value is… | code | T0 | `ex03_the_comparison_is_proved_by_which_failure_you_get.py` |
| 4 | Break the finite listen stream before its final response. Reissue it with a new JSON-RPC id a… | code | T0 | `ex04_a_reissue_is_not_a_resume_because_the_old_id_names_nothing.py` |
| 5 | Add an explicit workflow handle to the ping tool. Bind it to an authorization subject without… | code | T0 | `ex05_affinity_is_neither_necessary_nor_sufficient_for_the_handle.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code. Four of the five drive a
real `ThreadingHTTPServer` on a loopback port; the handler logs every request
to stderr, so each one runs its exchange under a redirect.

Lesson 13.08's client talked to fakes. This one is the wire itself, and the
recurring shape is that **the transport's checks run in an order the lesson
does not state, and the order is what you actually observe.** Three of the five
answers are only visible as the *difference* between two failures.

### 1 — a missing header and a wrong one are the same answer

**ANSWER: HTTP `400`, `-32020`, `Header mismatch: Mcp-Method`,** with the
JSON-RPC id echoed. The same request with the header answers `200`.

**FINDING: absent and wrong are indistinguishable.** Dropping the header,
setting it to `tools/call`, and setting it to `TOOLS/LIST` all produce the same
status, code and message — `validate_http_headers` raises one fault for
`is None or != method`. A client cannot tell "you forgot it" from "you sent the
wrong one".

**FINDING: the name is case-insensitive and the value is not.** `mcp-method`
answers **200** because `http.client` folds header names; `TOOLS/LIST` as a
value answers **400**. Both casings, one request, opposite sides.

**FINDING: the header contract is checked before the version.**

| sent | answer |
|---|---|
| `2027-01-01`, header present | `-32022` |
| `2027-01-01`, header absent | `-32020` |

A client fixing two faults at once only learns about the second after fixing
the first.

**FINDING: and a notification is not exempt.** No `id` and no `Mcp-Method`
answers `400` with `"id": null`, where a well-formed one answers `202`. "No id"
does not mean "no response".

### 2 — "matching" is the word that selects the only path to -32022

**ANSWER: `400`, `-32022`, and
`{"supported":["2026-07-28"],"requested":"2027-01-01"}` byte for byte** —
checked on the serialized bytes, because key order is the part of "exact" a
dict comparison would not catch.

**FINDING: disagreeing either way is `-32020`, never `-32022`.** Header current
over body future, and header future over body current, both answer `Header
mismatch: MCP-Protocol-Version` with no data. The two strings are compared
before `validate_supported_version` asks whether the version exists — so
"matching" is what selects the path the exercise wants.

**FINDING: `-32022` does not mean "too new".** The legacy `2025-11-25` answers
the same code and shape. The number reports an empty intersection, not a
direction; only `data.supported` says what to do.

**FINDING: the rejection carries discovery's answer, and the status is a
separate channel.** `data.supported` equals `server/discover`'s
`supportedVersions`, so renegotiation needs no second round trip. Meanwhile
`-32022` rides on `400` and `-32601` on `404`, picked by one branch in
`do_POST`.

### 3 — the comparison is proved by which failure you get

`resources/read` has **no handler**, so a correct sentinel has no success to
reach. The comparison has to be shown by moving one input.

**ANSWER: correct sentinel → `404 / -32601` (past the headers, dead at the
dispatcher). Sentinel encoding a different URI → `400 / -32020 Header mismatch:
Mcp-Name`.** Only a comparison against `params.uri` separates those two.

**FINDING: the contract is enforced for a method the server does not
implement.** An ASCII URI answers the same `-32601` with the value sent
unencoded. `body_name` knows `resources/read` reads `params.uri`; `dispatch`
has never heard of the method.

**FINDING: the sentinel is what makes the channel usable, not a convention on
top of one.** The raw UTF-8 value raises **`UnicodeEncodeError`** in
`http.client`, which encodes headers as latin-1. There is no server behaviour
to observe because there is no request.

**FINDING: one code, two faults — and the encoder is round-trip safe.**
`=?base64?!!!notb64?=` answers `Malformed Base64 MCP header value`, a different
message under `-32020`. And `encode_header_value("=?base64?zzz?=")` encodes
that literal rather than passing it through, so a value shaped like an encoding
decodes back to itself.

### 4 — a reissue is not a resume, because the old id names nothing

**ANSWER: the break costs the final response; the reissue replaces the whole
stream.** Closing after the acknowledgement takes **1** of **3** frames.
`listen-2` delivers `acknowledged`, `tools/list_changed` and the result, each
tagged with its own `subscriptionId`, and the refetch answers `['ping']`.

**FINDING: there is nothing to resume from.** The stream writes **0** `id:` SSE
fields, so `Last-Event-ID` has no anchor — as the lesson says it should not.

**FINDING: a reissue is not a resume, because the server keeps no
subscription.** `subscription_messages` is a pure function of the request, and
the module's only subscription-shaped names are the key constant and that
function. Nothing is keyed by a listen id, so a new one is the only thing a
client *can* send.

**FINDING: the refetch is load-bearing, because the notification carries no
tools.** The `list_changed` frame's params are `['_meta']` and nothing else.
The acknowledgement echoes the *accepted* filter — `{"toolsListChanged": true,
"bogus": true}` comes back without `bogus` — so a client is told what it
subscribed to and still has to ask what changed.

### 5 — affinity is neither necessary nor sufficient for the handle

**ANSWER: `ping` mints a 32-character `token_urlsafe` handle, and any replica
continues it.** Replica `a` mints at `turns=1`; replica `b`, with no memory of
that call, reaches `turns=2`.

**FINDING: per-replica storage reproduces the lesson's five-step failure.** The
same `Replica` code over two separate dicts answers **-32004** on `b` — step 4,
reached by changing one argument. The handle is not what breaks; the place it
is kept is.

**FINDING: the binding is the subject.**

| who | where | answer |
|---|---|---|
| intruder | minting replica | `-32003` |
| owner | other replica | `turns=3` |

Staying on the connection buys nothing and leaving it costs nothing, so
affinity is neither necessary nor sufficient. Authorization is checked on every
use rather than established once.

**FINDING: the handle must be an argument, because the envelope has no slot for
it.** A real POST answers with `Mcp-Session-Id: None` and `http_headers_for`
sends none. Expiry therefore lives on the record — an aged handle is
**-32005** — and the connection it was minted on is irrelevant either way.
