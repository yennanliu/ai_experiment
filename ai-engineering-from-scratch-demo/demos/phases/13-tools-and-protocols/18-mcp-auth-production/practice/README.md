<!-- generated:start -->
# 13-tools-and-protocols / 18-mcp-auth-production

Solutions to all 7 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/18-mcp-auth-production/) · upstream spec
`phases/13-tools-and-protocols/18-mcp-auth-production/docs/en.md`

```bash
uv run demo practice run 18-mcp-auth-production --ex 1
uv run demo explain 18-mcp-auth-production --ex 1
uv run pytest demos/phases/13-tools-and-protocols/18-mcp-auth-production
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Trace the flow. Note how the IdP rotates a key in step 6, the scheduled `… | code | T0 | `ex01_the_overlap_is_one_rotation_and_only_for_the_current_key.py` |
| 2 | Add a new IdP to the protected-resource metadata's `authorization_servers` list. Issue a toke… | code | T0 | `ex02_the_allowlist_admits_an_issuer_the_fetch_can_never_serve.py` |
| 3 | Add a rate-limit check to `register_client` that runs before the registrar accepts a request.… | code | T0 | `ex03_the_bucket_has_to_precede_the_work_it_is_protecting.py` |
| 4 | Read RFC 7591 and identify two fields the lesson's `/register` handler does not validate. Add… | code | T0 | `ex04_the_hint_names_one_field_that_is_already_validated.py` |
| 5 | Add a second authorization server. Confirm the client stores a separate issuer-keyed enrollme… | code | T0 | `ex05_the_client_is_keyed_by_issuer_and_the_server_it_points_at_is_not.py` |
| 6 | Prove the DoS fix. Send the validator a token with a random `kid` and confirm `refresh_jwks`… | code | T0 | `ex06_a_pure_fetch_is_idempotent_and_a_mint_is_an_amplifier.py` |
| 7 | Exercise deprecated DCR with both `native` and `web` clients. Confirm a web client with an HT… | code | T0 | `ex07_the_two_client_types_forbid_each_others_redirect.py` |
<!-- generated:end -->

## Answers

All seven are T0 and stdlib, and all seven ship code — including exercise 1,
which reads as an observation task but is answerable only by running the flow
twice and comparing.

Production auth is a set of caches and lists that have to agree with each
other, and the recurring finding is **a pair that does not**: an allowlist
with no matching fetch, a key cache with no relation to token lifetime, a
client whose per-issuer memory outlives its single-issuer state.

### 1 — the overlap is one rotation, and only for the current key

**ANSWER: both tokens validate after a rotation and a refresh, no restart.**

**FINDING: the window is one rotation for the newest key and zero for the
other.** `rotate_key` retires by position, so a token minted from `keys[0]` is
already dead after one rotation. The grace period is decided by *where in the
cycle* a token was issued.

**FINDING: the resource server cannot rotate, only re-pull.**

**FINDING: retirement is by count.** The only comparison in `rotate_key` is on
`len(self.keys)` — never a clock, never the 300-second TTL. Whether a live
token outlives its key is a scheduling property enforced nowhere.

### 2 — the allowlist admits an issuer the fetch can never serve

**ANSWER: the listed IdP is accepted; the unlisted one gets the exact
challenge.**

**FINDING: the allowlist is checked before any JWKS work, on purpose.**
Unlisted issuer → **0** refreshes; listed issuer with unknown kid → **1**.

**FINDING: `allowed_issuers` and `jwks_cache` must agree, and only one has an
updater.** `refresh_jwks` hard-codes `self.auth_server.issuer` while the cache
is keyed by issuer. Allowlisting is necessary and not sufficient, and the
shipped code has no path that makes it sufficient.

**FINDING: two rejections that mean opposite things are one status.** One is
never worth retrying and the other is.

### 3 — the bucket has to precede the work it is protecting

**ANSWER: a token bucket per IP, in front, refusing the sixth.**

**FINDING: placement is measurable as work, not verdicts.**

| placement | calls into `register_client` | clients minted |
|---|---:|---:|
| in front | 5 | 5 |
| behind | 6 | 6 |

Both return the same statuses, so a status-only test cannot tell them apart —
and the late refusal mints a registration the caller is told it did not get.

**FINDING: the bucket refills, so the limit is a rate and not a quota.**

### 4 — the hint names one field that is already validated

**ANSWER: `software_statement` and `grant_types`.** The first is ignored
(RFC 7591 §2.3 makes it a signed JWT whose claims take precedence); the second
is echoed and persisted verbatim on a server that implements only
`authorization_code`.

**FINDING: the hint's second field is already validated, in three layers.**
The redirect scheme is the most-checked thing in the handler.

**FINDING: validating them is a rejection each, and they fail differently.**

**FINDING: the unvalidated grant is quoted back.** A registration response is
a statement about the server, and this one repeats the client's claim as one.

### 5 — the client is keyed by issuer, and the server it points at is not

**ANSWER: two enrollments, two tokens, neither reused.** What is issuer-keyed
is the *entry*, not the identifier — both carry the same `client_id`, because
CIMD makes it the document URL and that URL is portable by design.

**FINDING: `expected_issuer` is a single slot.** A response from the first
issuer arriving after the switch is checked against the second's name.

**FINDING: `auth_server` is one attribute.** The dicts are a memory of past
enrollments, not a router.

**FINDING: refusing to reuse is the resource server's doing.**

### 6 — a pure fetch is idempotent and a mint is an amplifier

**ANSWER: one refresh per bogus token, key count unmoved.** 20 tokens, 20
refreshes, 2 keys throughout.

**FINDING: re-wired, the count does *not* climb — and the damage is worse.**
`keys[-2:]` caps the list, so the exercise's "watch the key count climb" does
not happen. What climbs is churn: 20 forced rotations leave a legitimate token
that verified before the storm answering `unknown kid` after it. An attacker
cannot grow the list and can revoke everyone else's tokens.

**FINDING: the fall-back is reachable twice and fires once.** Cold cache costs
**2** fetches, warm cache **1** — the bound is per request, not per token.

**FINDING: the allowlist is what makes the bound cheap.**

### 7 — the two client types forbid each other's redirect

**ANSWER: both rejections land with `invalid_redirect_uri`.**

**FINDING: the two rules are disjoint on the interesting cases.**

| URI | legal for |
|---|---|
| `https://app.example/cb` | web, native |
| `http://app.example/cb` | neither |
| `http://127.0.0.1:1234/cb` | native |
| `com.example.app:/cb` | native |

No URI is web-only, which makes web the narrower rule rather than a different
one.

**FINDING: the loopback allowance is by hostname, not by port** — the correct
reading of RFC 8252, since a native app cannot reserve a port in advance.

**FINDING: `application_type` is required, so there is no third behaviour.**
