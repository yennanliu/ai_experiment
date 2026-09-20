<!-- generated:start -->
# 13-tools-and-protocols / 16-mcp-security-oauth-2-1

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/16-mcp-security-oauth-2-1/) · upstream spec
`phases/13-tools-and-protocols/16-mcp-security-oauth-2-1/docs/en.md`

```bash
uv run demo practice run 16-mcp-security-oauth-2-1 --ex 1
uv run demo explain 16-mcp-security-oauth-2-1 --ex 1
uv run pytest demos/phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add refresh-token rotation and reject reuse of the previous refresh token. | code | T0 | `ex01_refusing_the_old_token_is_not_the_point_revoking_the_family_is.py` |
| 2 | Add an issuer allowlist. On issuer change, reuse only a portable CIMD URL; refuse all prior i… | code | T0 | `ex02_the_url_is_portable_because_it_names_the_client_not_the_registration.py` |
| 3 | Add an expiry to authorization codes and confirm a late exchange fails. | code | T0 | `ex03_the_expiry_is_already_there_and_the_clock_is_not_reachable.py` |
| 4 | Build a web client variant with a remote HTTPS redirect and compare its DCR metadata to the n… | code | T0 | `ex04_dcr_demands_the_field_that_decides_the_rule_and_cimd_does_not.py` |
| 5 | Add a second resource under the same issuer. Confirm its access token cannot be used at the f… | code | T0 | `ex05_the_audience_holds_and_the_challenge_points_at_the_wrong_resource.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code.

OAuth 2.1 is a set of bindings — a token to an audience, a client to an
identifier, a code to a moment — and the five exercises each pull on one of
them. Twice the binding is already in the lesson and the interesting question
is what it *cannot* say; twice it is missing and building it exposes a second
decision nobody named.

### 1 — refusing the old token is not the point; revoking the family is

**ANSWER: rotation, and a reused refresh token is refused.** Three refreshes
hand out **4** tokens and retire **3**.

**FINDING: refusing the replay leaves the thief holding the live token.**

| policy | tokens live after the replay |
|---|---:|
| rotation only | 1 (the thief's) |
| rotation + family revocation | 0 |

The reuse is the signal that two parties hold one credential, and ending the
session is the only response that helps.

**FINDING: the lesson has no refresh token to rotate.** `Token` has **7**
fields, none of them a refresh token, and no renewal path short of a fresh
PKCE authorization.

**FINDING: the family must be named at issue time.** Deriving it from the
access tokens would group a legitimate second session too — they share
`client_id` and `subject`.

### 2 — the URL is portable because it names the client, not the registration

**ANSWER: the CIMD id crosses unchanged and the DCR id does not.** One
identifier reused, one reissued.

**FINDING: the allowlist is what makes the move deliberate.** Without it the
client follows whichever issuer the resource named — and that metadata is what
an attacker who controls the resource gets to write.

**FINDING: the client's stores are keyed by issuer, so dropping is a
deletion.** What the keying does *not* do is expire anything: the dropped
token object is still valid until something drops it.

**FINDING: the resource server refuses a foreign issuer and cannot say why.**
Same code, message and data as an expired token — the issuer and audience
checks share one branch.

### 3 — the expiry is already there, and the clock is not reachable

**ANSWER: a late exchange fails; the window is 300 seconds.**

**FINDING: the expiry already existed, and the test cannot reach it
honestly.** `exchange` takes **5** parameters and no clock. Confirming the
expiry means rewriting `expires_at` inside `pending_codes` — testing the
branch rather than the behaviour.

**FINDING: expired and never-issued are the same answer, deliberately.**
Distinguishing them would tell an attacker which guesses were once real.

**FINDING: single use is enforced separately, and is the stronger control.**
Expiry bounds what a *leaked, unused* code is worth; the pop is what stops the
replay of one that worked.

### 4 — DCR demands the field that decides the rule, and CIMD does not

**ANSWER: the documents differ in exactly two fields** — `application_type`
and `redirect_uris` — and share the other **4**.

**FINDING: the two are not independent.** `_validate_application` applies the
HTTPS-and-not-loopback rule *only* under `application_type == "web"`, so the
declaration selects the rule that judges the URI.

**FINDING: DCR requires the deciding field and CIMD does not.**
`require_application_type=True` against `False`. A CIMD client can register a
remote HTTPS redirect with no type at all, and the web rule is never applied.

**FINDING: the loopback redirect is what the native client cannot share.**
Portability of the identifier is not portability of the registration.

### 5 — the audience holds, and the challenge points at the wrong resource

**ANSWER: each token is refused at the other resource**, both directions,
`401 invalid_token`.

**FINDING: the `WWW-Authenticate` challenge names a module constant, not the
server.** Both servers answer with `RESOURCE_METADATA_URI`, so a client
following the second resource's challenge reads the *first* resource's
metadata and comes back with a token for the wrong audience — in a loop.

**FINDING: the client will not make that mistake.** Its cache is keyed
`(issuer, resource)`: **2** entries, **2** audiences, **1** issuer. Only the
server's side of the isolation is the protocol's.

**FINDING: wrong-resource and wrong-issuer are one branch.** One needs a new
token from the same authorization server and the other a different
authorization server, and the response separates them by nothing.
