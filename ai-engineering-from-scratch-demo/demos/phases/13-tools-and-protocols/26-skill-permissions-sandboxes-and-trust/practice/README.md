<!-- generated:start -->
# 13-tools-and-protocols / 26-skill-permissions-sandboxes-and-trust

Solutions to all 6 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/26-skill-permissions-sandboxes-and-trust/) · upstream spec
`phases/13-tools-and-protocols/26-skill-permissions-sandboxes-and-trust/docs/en.md`

```bash
uv run demo practice run 26-skill-permissions-sandboxes-and-trust --ex 1
uv run demo explain 26-skill-permissions-sandboxes-and-trust --ex 1
uv run pytest demos/phases/13-tools-and-protocols/26-skill-permissions-sandboxes-and-trust
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add separate read, create, overwrite, and delete path permissions. Test the same path under e… | code | T0 | `ex01_deleting_the_workspace_root_is_inside_the_workspace_jail.py` |
| 2 | Add an origin policy that permits `https://registry.example.test` on port 443, separately per… | code | T0 | `ex02_one_malformed_allowlist_entry_denies_every_origin.py` |
| 3 | Model a package-manager command whose lifecycle hooks execute repository code. Decide whether… | code | T0 | `ex03_a_prefix_match_is_a_prefix_not_a_command.py` |
| 4 | Extend `ActionRequest` with an idempotency key and require one for external writes. | code | T0 | `ex04_the_reviewer_stops_being_a_function_the_moment_it_dedupes.py` |
| 5 | Write an approval message for a staging publish, then for a production publish. Make the targ… | code | T0 | `ex05_the_shipped_prompt_is_the_same_sentence_for_both_publishes.py` |
| 6 | Threat-model a skill that reads web pages and writes pull-request comments. Mark every trust… | code | T0 | `ex06_the_untrusted_flag_is_set_by_the_code_the_page_is_attacking.py` |
<!-- generated:end -->

## Answers

All six are T0 and stdlib, and all six ship code.

`review_action` is a pure function from a request to a verdict, and five of
these six exercises end up at the same place: **the facts the verdict needs
are not in the request**. Whether the file already exists. What the
lifecycle hooks will run. Whether this is a read or a write. Whether the
publish is reversible. Whether the content came from a web page. The review
is well built and it is deciding on a description of an action rather than
on the action.

Where the module does read content — `contains_secret` on the payload — it
catches the one attack that crosses from what was read to what is written.
And the one boundary it enforces unconditionally, `policy-change`, is denied
before the kind allowlist is even consulted. That is the lesson's own claim
in code: a skill is context, and the host is the boundary.

### 1 — deleting the workspace root is inside the workspace jail

**ANSWER: four permissions, one path, and the answer moves with the file
too.** With the file on disk: `allow, require-approval, require-approval,
deny`. Without it: `allow, allow, allow, deny`. The shipped policy gives
**2** distinct answers to **4** questions, and the same **2** either way.

**FINDING: `write` cannot distinguish create from overwrite, because the
review never looks.** `strict=False` resolution means the request is
identical whether the file exists or not. Splitting them needs a `stat`,
which makes the verdict a claim about a file — and therefore stale the
moment it is returned.

**FINDING: deleting the workspace root is inside the workspace jail.** The
containment test is `resolved != root and root not in resolved.parents`; an
equal path passes both halves. The shipped review answers
`require-approval` for deleting the whole workspace.

**FINDING: the four operations need four path sets, not one root.**
`SandboxPolicy` has **1** path field and **1** kind allowlist.

### 2 — one malformed allowlist entry denies every origin

**ANSWER: two entries, and every hop reviewed as if it were the first.**
`:443` and `:8443` are two entries, because the port is part of the origin.
A redirect chain is the caller's loop over the shipped review.

**FINDING: one malformed allowlist entry denies every origin.** A trailing
path fails `origin_only` validation *inside* `review_action`, which returns
`network-policy-shape` — a denial of the request. Failing closed is right;
reporting it as a policy-shape error rather than an allowlist miss is what
makes it debuggable.

**FINDING: normalization does most of the work before the allowlist.**
Uppercase, a trailing dot, an explicit `:443`, a query and a fragment all
collapse to the same origin; the userinfo form is rejected as
`network-shape` rather than prefix-matched.

**FINDING: the request has one URL, so the chain lives outside the model.**
No redirect history on `ActionRequest`, no hop count on `ReviewDecision`.

### 3 — a prefix match is a prefix, not a command

**ANSWER: deny at the review layer, isolate at the executor, never ask.**
Two repositories with byte-identical requests run **2** different programs,
so an approval prompt would be one prompt for two outcomes.

**FINDING: a prefix match is a prefix, not a command.** Allowlisting
`("npm", "install", "--ignore-scripts")` also accepts a trailing
`--foreground-scripts`. The mitigation can be added and then cancelled by a
flag the allowlist never sees.

**FINDING: the review is a pure function of the request, and the hooks are
not in it.** Deciding on argv alone is sound only when the verdict is deny.

**FINDING: isolation is not expressible in `SandboxPolicy`.** **0** of its
**6** fields mention a namespace, isolation or credentials.

### 4 — the reviewer stops being a function the moment it dedupes

**ANSWER: a key is required for external writes, and a replayed key is a
duplicate rather than a second write.** A *different* key with the same
payload allows, because two deliberate publishes are two publishes.

**FINDING: the reviewer stops being a function the moment it dedupes.**
A key store makes the second identical call answer differently. The
component becomes a service, with durability, ordering and eviction
questions of its own.

**FINDING: "external write" is not a thing the request can say.** No
method, verb or direction field, so the rule falls back to `payload != ""`
and misclassifies **2** of **4** shapes.

**FINDING: the key is the caller's word, exactly like `approved`.** It
protects against a dropped response, not against a caller that wants two.

### 5 — the shipped prompt is the same sentence for both publishes

**ANSWER: one renderer, four required facts, and two messages that differ
only where the deployment differs.** The renderer refuses **4** of **4**
single-fact omissions, which is what makes "explicit" testable.

**FINDING: the shipped prompt is the same sentence for both publishes.**
`host policy gates 'network' behind approval` names the policy that fired.
An approver needs the consequence.

**FINDING: `ReviewDecision` has no field for the prompt.** The approval text
is generated elsewhere, so nothing guarantees the approver and the reviewer
describe the same action.

**FINDING: reversibility is the axis and it is not in the model.**
`approval_kinds` is keyed by kind; staging and production are both
`network`.

### 6 — the untrusted flag is set by the code the page is attacking

**ANSWER: six boundaries, three the module enforces and three left to the
caller.** Enforced: authority, origin, secret. Not enforced: provenance,
direction, and the fact that read access and exfiltration are one
permission.

**FINDING: the untrusted flag is set by the code the page is attacking.**
With `influenced_by_untrusted_content` the write needs approval; without it
the identical write is allowed.

**FINDING: reading pages and posting comments are the same permission.** A
bare fetch and a fetch carrying repository contents both answer `allow`.

**FINDING: the secret scan is the one control that reads the content.** It
catches `api_key=` copied from a page into a comment — and its **3**
patterns are its whole coverage; a bare `ghp_` token matches none of them.

**FINDING: the authority boundary is the one thing that cannot be
negotiated.** `policy-change` is denied before the kind allowlist is
consulted, whatever the approval or the claimed permissions say.
