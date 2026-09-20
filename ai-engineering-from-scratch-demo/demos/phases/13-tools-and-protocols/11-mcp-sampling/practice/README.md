<!-- generated:start -->
# 13-tools-and-protocols / 11-mcp-sampling

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/11-mcp-sampling/) · upstream spec
`phases/13-tools-and-protocols/11-mcp-sampling/docs/en.md`

```bash
uv run demo practice run 11-mcp-sampling --ex 1
uv run demo explain 11-mcp-sampling --ex 1
uv run pytest demos/phases/13-tools-and-protocols/11-mcp-sampling
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Change the file-selection response to invalid JSON. Confirm the server returns `-32602` inste… | code | T0 | `ex01_the_server_distrusts_the_shape_and_takes_the_contents_on_faith.py` |
| 2 | Change `audience` between the first call and retry. Explain why the sealed state blocks cross… | code | T0 | `ex02_the_seal_binds_the_arguments_and_publishes_everything_else.py` |
| 3 | Add a third round that asks the host to critique the summary. Carry the earlier summary insid… | code | T0 | `ex03_a_signed_counter_is_not_a_rate_limit_while_replay_is_free.py` |
| 4 | Remove Sampling by replacing the fake host callback with a server-owned model adapter. List w… | code | T0 | `ex04_the_round_trip_was_the_approval_and_removing_it_removes_that.py` |
| 5 | Add an expiry test using a state value that is one second past its deadline. | code | T0 | `ex05_the_deadline_itself_passes_and_the_clock_has_no_caller.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code. The scaffold classified
exercise 2 as prose on the word "Explain", but it has a code deliverable — a
changed argument and the rejection it earns — so it is filed as code with the
explanation argued from the measurement.

The lesson replaces a session with a signed token, and the five exercises walk
the seam that creates: **what the server verifies, what it merely receives,
and what the token publishes to whoever is carrying it.** Two of the five find
a control that is sound within one request chain and worth nothing across
chains.

### 1 — the server distrusts the shape and takes the contents on faith

**ANSWER: `-32602`, `pick_files must return JSON`.** The text is parsed before
it can reach `FAKE_REPO`.

**FINDING: three distinct rejections, testing three different things.**

| response | rejected by |
|---|---|
| `not json` | parsing |
| `{"a":1}`, `[1,2]` | the shape check |
| `["nope.md"]` | membership |

**FINDING: the membership gate is "at least one", not "all".**
`["README.md","ghost.md","server.py"]` **succeeds**, yielding two files — the
invented name filtered out by `if name in FAKE_REPO` with nothing said. A
model that hallucinated a third of its answer is indistinguishable downstream
from one that returned two files.

**FINDING: over-delivery is truncated as quietly.** Five valid files yield
three, because the comprehension ends in `[:3]`. And every outcome here is
`-32602` — *invalid params* — so a model failure is attributed to the client
that relayed it. There is no code for "the text you asked for came back
wrong".

### 2 — the seal binds the arguments and publishes everything else

**WHY IT BLOCKS CROSS-REQUEST REUSE:** the first response wrote
`argumentsDigest` into an HMAC-signed body; the retry recomputes the digest
from the arguments actually sent and compares. Change `audience` and the
digest changes; the token cannot be re-signed without the server secret. So a
token is valid for one argument set, one principal and one method, and travels
only with them.

**ANSWER: `-32602`, `requestState arguments mismatch`.**

**FINDING: the binding is to meaning, not spelling.** The digest is over
`json.dumps(sort_keys=True)`, so re-ordering keys is accepted and adding an
unused one is not. A client may reformat and may not add.

**FINDING: four things are bound, and the earlier check wins.** Integrity →
principal → method → arguments → expiry. A token replayed by another principal
*with* changed arguments reports the principal, so fixing a bad token is
iterative.

**FINDING: sealed means signed, not hidden.** The body is
`urlsafe_b64encode(json)`. Without the secret a client reads `principal`,
`phase`, `expiresAt`, the digest — and on round two the `picked` file list.
Integrity is not confidentiality: whatever the server puts in this token is
published to whoever holds it.

### 3 — a signed counter is not a rate limit while replay is free

**ANSWER: three rounds, and the summary reaches the end without being
resent.** `pick` → `summarize` → `critique`; round three's `inputResponses`
carry only the critique, yet the final `structuredContent` has all three
fields. The server trusts the summary because it signed it.

**FINDING: the cap lives in the token because there is nowhere else, so
replay resets it.** Replaying the round-1 token **5** times yields **5** fresh
tokens, all at `round: 2`. A signed counter bounds a *chain*, not a flow.
Bounding the flow needs single-use tokens, which needs the server state the
lesson removed.

**FINDING: each round refreshes the deadline.** `expiresAt` is re-set to
`now + 300` in every re-seal, so a three-round flow has no overall time limit
— it has three consecutive ones.

**FINDING: the cap is checked after the whole verification runs.** A replayed
token costs the server the HMAC, the principal, the method, the digest and the
expiry before being refused — which is the work an attacker wanted done.

### 4 — the round trip was the approval, and removing it removes that

The adapter calls the lesson's own `fake_host_model`, so the model is held
constant and only its *owner* changes. Every number below is that one
substitution.

**ANSWER: one dispatch instead of three, zero tokens instead of two, identical
`structuredContent`.** Seal, digest, principal and expiry all go — there is no
second request to bind.

| responsibility | host, before | server, after |
|---|---:|---:|
| `input_required` results to approve | 2 | 0 |
| `modelPreferences` objects received | 2 | 0 |
| prompt texts seen | 2 | 0 |

**APPROVAL moves** because the approval point was the round trip, not the
prompt text — the trip is the only moment a host can decline, and nothing in
the remaining protocol offers another.

**BILLING moves** because `costPriority`/`intelligencePriority` were the
host's per-round spending lever. The `sampling` capability goes the same way:
without it MRTR answers **-32021** and the adapter completes regardless, so
the client can no longer refuse by declining to advertise.

**OBSERVABILITY moves** rather than disappearing — the server sees both
prompts now. The client's remaining evidence is the final content, and the
repository text that reached a model is no longer anything it saw.

### 5 — the deadline itself passes, and the clock has no caller

**ANSWER: `expiresAt + 1` is `requestState expired`; `expiresAt` itself is
not.** The comparison is `expiresAt < now`, so the deadline second is inside
the window. A test written *at* the deadline would have passed and proved
nothing — which is why the exercise says "one second past".

**FINDING: the injectable clock has no caller.** `verify_request_state` takes
`now=None`, and `tools_call` never passes it. The live path always reads
`time.time()`, so the parameter exists for a test the lesson does not ship —
and this exercise has to reach past `dispatch` to use it.

**FINDING: expiry is checked last, so an expired token reports something
else.** Expired *and* carrying changed arguments answers `arguments mismatch`;
fix the arguments and the same token answers `expired`. Two round trips for
two faults, and the fixable one is reported second.

**FINDING: the 300-second window is re-granted, not consumed.** It measures
idle time between rounds rather than the age of the flow, so an expired token
is only ever evidence that *this* round was slow.
