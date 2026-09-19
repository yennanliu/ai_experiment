<!-- generated:start -->
# 13-tools-and-protocols / 10-mcp-resources-and-prompts

Solutions to all 6 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/10-mcp-resources-and-prompts/) · upstream spec
`phases/13-tools-and-protocols/10-mcp-resources-and-prompts/docs/en.md`

```bash
uv run demo practice run 10-mcp-resources-and-prompts --ex 1
uv run demo explain 10-mcp-resources-and-prompts --ex 1
uv run pytest demos/phases/13-tools-and-protocols/10-mcp-resources-and-prompts
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `notes://projects/{project}/notes/{id}` resource template and validate both variables. | code | T0 | `ex01_validating_a_variable_is_a_character_class_not_a_presence_check.py` |
| 2 | Add pagination to `resources/list` while preserving deterministic order. | code | T0 | `ex02_the_cursor_has_to_be_the_sort_key_or_a_page_repeats_itself.py` |
| 3 | Change one resource to `cacheScope: "private"` with `ttlMs: 0`, add a host-level no-store pol… | code | T0 | `ex03_zero_is_a_ttl_and_a_truthiness_test_turns_it_into_a_default.py` |
| 4 | Add a prompt-list change subscription and prove no event is sent when the filter omits `promp… | code | T0 | `ex04_no_event_is_sent_is_a_claim_about_the_caller_not_the_stream.py` |
| 5 | Create two simultaneous subscriptions and prove each event carries the correct request ID. | code | T0 | `ex05_simultaneous_is_the_clients_bookkeeping_and_the_server_has_none.py` |
| 6 | Add an authorization subject to the read handler and prove a cache entry cannot cross subjects. | code | T0 | `ex06_the_subject_cannot_come_from_the_request_that_is_using_it.py` |
<!-- generated:end -->

## Answers

All six are T0 and stdlib, and all six ship code. The scaffold classified
exercise 3 as prose on the word "explain", but it has a code deliverable —
a changed hint and a host policy — so it is filed as code with the threat
argued in its docstring and below.

Lesson 13.09 was the wire. This one is what the wire carries, and the theme is
narrower and sharper: **every one of these six exercises asks for a control
the server states and does not implement.** `cacheScope` with no cache,
`private` with no subject, a template with no handler, a filter whose
enforcement lives in the caller's list comprehension. The work is building the
missing half and then measuring what the stated half was worth.

### 1 — validating a variable is a character class, not a presence check

**ANSWER: `[^/]+` for both variables, matched after decoding** — one
acceptance against four rejections, and the accepted pair round-trips.

**FINDING: a presence check does not merely over-accept, it is ambiguous.**

| reading of `notes://projects/x/notes/y/notes/z` | project |
|---|---|
| greedy `(.+)` | `x/notes/y` |
| lazy `(.+?)` | `x` |
| `[^/]+` | no match |

Both loose readings pass a non-empty test, so the template no longer
determines *which* values it accepted.

**FINDING: percent-encoding puts the ambiguity back if you decode too late.**
`a%2Fb` satisfies `[^/]+` — there is no slash in `%2F` — and unquotes to
`a/b`. The two steps are ordered, and "both variables" means both decoded.

**FINDING: the grammar is well-formed and unreachable.** An expanded URI
answers `-32602` from `resources_read`'s exact `NOTES` lookup, `HANDLERS` has
no templates method, and the advertised resources capability is
`['listChanged', 'subscribe']` — no key to announce a template under.

### 2 — the cursor has to be the sort key, or a page repeats itself

Two notes cannot falsify any paging scheme, so the store is seeded to six.

**ANSWER: a cursor that is the last URI served.** Three full pages whose
concatenation equals the unpaginated list. Six divides evenly by two, so no
page can say it is the last — a **4th** request returns **0** rows. Exhaustion
is a round trip, not a flag.

**FINDING: an offset cursor is wrong under insertion, and wrong by
repeating.** Insert `note-1a` between page 1 and page 2:

| cursor | page 2 |
|---|---|
| key (`note-2`) | `note-3, note-4` |
| offset (`2`) | `note-2, note-3` |

`note-2` served twice. Nothing about the offset became invalid; the list
underneath it moved.

**FINDING: the order was already deterministic, and paging is what makes it
matter.** Reversing `NOTES` changes no page, because `resources_list` sorts.
With one page that is untestable.

**FINDING: the cache hint does not survive being split.** Every page inherits
`ttlMs: 300000, cacheScope: "public"` and nothing names its generation — page
1 from before an insert and page 2 from after are individually fresh and
jointly a list that never existed.

### 3 — zero is a ttl, and a truthiness test turns it into a default

**THREAT: a shared host cache serving a private resource.** `private` stops
the stored copy reaching a *second subject*; `ttlMs: 0` stops the *same*
subject being served it after the note changed or access was revoked. Neither
implies the other — which is the argument for both.

**ANSWER: the sensitive read is `private` with `ttlMs: 0`, and the strict host
stores nothing.** One entry after reading both notes. The no-store path is the
ttl, not the scope.

**FINDING: `ttlMs: 0` is not `ttlMs` absent, and `or` cannot tell them
apart.** `result.get("ttlMs") or 60000` assigns **60000** and stores;
`result.get("ttlMs", 60000)` gives **0**. The strictest instruction in the
protocol becomes the default, silently, because `0` is falsy.

**FINDING: the two ablations leak in different directions.** Keyed on the URI
alone, alice's read is served to mallory. Honouring the scope but not the
zero, alice is served her own stale copy after the text changed.

**FINDING: neither control is enforced by the server, or could be.** The
module has **0** cache-shaped names. The server states a policy it has no
machinery to apply — which is why the exercise asks for the host half.

### 4 — "no event is sent" is a claim about the caller, not the stream

**ANSWER: subscribed carries the change; omitted carries only the
acknowledgement.** One change event against zero.

**FINDING: the method cannot enforce it — the caller can.**
`prompts_list_changed()` returns `None` rather than declining. An unfiltered
transmitter puts the JSON literal `null` on the wire. The lesson's own
`demo()` ends with `if item is not None`, which is where "no event is sent" is
actually implemented.

**FINDING: only the literal `True` subscribes.** `value is True`, so of
`True`, `1` and `"yes"` exactly one is accepted — even though `1 == True`.

**FINDING: a typo unsubscribes you silently.** `promptListChanged` is not in
`SUPPORTED_NOTIFICATION_FIELDS`, so the agreed filter is `{}` with no error.
The acknowledgement is the only place it shows — and it is rebuilt
alphabetically rather than echoed, so it is a normalized restatement and
comparing it against what you sent is the check.

### 5 — "simultaneous" is the client's bookkeeping, and the server has none

Both subscriptions watch the same resource, because a mis-tag is only
interesting when it would have produced a plausible frame.

**ANSWER: all six frames carry the id of the stream that produced them, none
cross-tagged.**

**FINDING: the fan-out is the caller's loop, not a broadcast.**
`resource_updated` consults only its own filter, and the module's
subscription-shaped names are a class and a constructor — nothing that holds
open streams to iterate.

**FINDING: two streams may share an id, and the server cannot tell.** The id
is an argument, not an allocation; a frozen dataclass of `(notifications,
subscription_id)` has nothing else to differ by, so two objects built with
`sub-1` produce **equal** acknowledgements. Only a null id is refused.
"The correct request ID" is a property the client maintains.

**FINDING: the terminal frame's `_meta` has a shape the notifications do
not.** `close()` merges in `response_meta()`, so the final result carries
**2** keys where every notification carries **1**.

### 6 — the subject cannot come from the request that is using it

**ANSWER: `read(params, subject)` against a grant table, cached on
`(subject, uri)`.** Three entries, and mallory's lookup of alice's note
misses. Keyed on the URI alone: two entries, and mallory is served it.

**FINDING: `clientInfo` is an identity claim, not an authorization subject.**
`validate_request_meta` accepts any `clientInfo.name` — a caller naming itself
`user:alice` is believed exactly as far as it asked to be. The subject has to
arrive beside the message, from whatever authenticated the transport.

**FINDING: the hint was always there and the parameter never was.**
`resources_read`'s signature is `('params',)` and it returns
`cacheScope: "private"` for every note.

**FINDING: a distinct denial code is an enumeration oracle.** `-32003` for
"exists but not granted" against `-32602` for "does not exist" lets mallory
tell them apart in one request. Answering `-32602` for both makes the
responses identical, and costs an unauthorized caller the difference between
"not yours" and "not there".
