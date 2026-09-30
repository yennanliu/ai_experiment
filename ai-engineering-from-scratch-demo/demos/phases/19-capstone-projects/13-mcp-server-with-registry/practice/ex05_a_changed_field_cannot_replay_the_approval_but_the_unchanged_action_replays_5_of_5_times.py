"""Exercise 5 -- a changed field cannot replay the approval, but the unchanged action replays 5 of 5 times.

    Bind an approval to one normalized argument digest. Change one field and prove the approval cannot be replayed.

Reading of the exercise: the approval is the lesson's
`ApprovalRecord.for_action`, bound to actor, tool, target and
`arguments_digest` (sha256 of `json.dumps(sort_keys=True,
separators=(",", ":"))`). It is minted for `jira.create` with
`{"title": "new bug"}` on the destructive server; the call is then made with
one field changed, and "cannot be replayed" is measured as a denial with the
handler never invoked. "Normalized" is probed on the digest itself.

**ANSWER: with `title` changed to `"new bug!"` the call is denied with
"approval arguments do not match requested action" and the handler runs 0
times.** A record bound to another target, or presented by another actor, fails the
same way with its own reason. The digest is normalized for key order
(`{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` share one digest) but not for Unicode
form or number spelling: NFC and NFD `"cafe"`-with-accent, and `1` vs `1.0`,
give different digests. Those fail closed.

**FINDING: the approval is not one-time.** The record has no use counter, so
the unchanged action is allowed 5 of 5 times, the handler runs 5 times, and
every call returns the same hard-coded `PROJ-99`. The doc lists "one-time or
repeat-use policy" among the bindings; the model implements only repeat-use.

**FINDING: the approval is the only gate on out-of-schema arguments.**
`dispatch` never checks `inputSchema`. An approval minted for `{"title": "x",
"assignee": "root"}` is honoured and the handler receives `assignee`, though
the schema says `additionalProperties: false`. The lesson's TypeScript
`tools/call` validates the schema first, per the doc.
"""

from __future__ import annotations

import time
import unicodedata

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "13-mcp-server-with-registry"
ARGS = {"title": "new bug"}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rw = ref.build_destructive_server()
    seen = []
    real = rw.handlers["jira.create"]
    rw.handlers["jira.create"] = lambda args: seen.append(dict(args)) or real(args)
    later = time.time() + 3600
    token = ref.Token("u42", rw.trusted_issuer, rw.url, frozenset({"jira:write"}), later)
    approval = ref.ApprovalRecord.for_action("u42", "jira.create", ARGS, rw.url, later)

    def call(args, record=approval, tok=token, tool="jira.create"):
        out = ref.dispatch(rw, tok, tool, args, ref.request_meta(), [], record)
        return out.get("error", {}).get("message") or out["structuredContent"]["id"]

    changed = call({"title": "new bug!"})
    handler_after_change = len(seen)
    other = ref.ApprovalRecord.for_action("u42", "jira.create", ARGS, "https://elsewhere.example.com", later)
    mallory = ref.Token("u99", rw.trusted_issuer, rw.url, frozenset({"jira:write"}), later)
    bindings = {"target": call(ARGS, other), "actor": call(ARGS, tok=mallory)}
    handler_after_bindings = len(seen)
    replays = [call(ARGS) for _ in range(5)]
    handler_after_replays = len(seen)
    extra = {"title": "x", "assignee": "root"}
    smuggled = call(extra, ref.ApprovalRecord.for_action("u42", "jira.create", extra, rw.url, later))
    d = ref.arguments_digest
    cafe = "café"
    return {
        "changed": changed, "handler_after_change": handler_after_change, "bindings": bindings,
        "handler_after_bindings": handler_after_bindings, "replays": replays,
        "handler_after_replays": handler_after_replays, "smuggled": (smuggled, seen[-1]),
        "key_order": d({"a": 1, "b": 2}) == d({"b": 2, "a": 1}),
        "unicode": d({"t": cafe}) == d({"t": unicodedata.normalize("NFD", cafe)}),
        "number": d({"n": 1}) == d({"n": 1.0}),
        "schema_extra": rw.tools["jira.create"].input_schema["additionalProperties"],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: one changed field denies the approval before the handler runs",
            (r["changed"], r["handler_after_change"]) == ("approval arguments do not match requested action", 0)
            and r["bindings"]
            == {
                "target": "approval target does not match this server",
                "actor": "approval actor does not match token subject",
            }
            and r["handler_after_bindings"] == 0
            and (r["key_order"], r["unicode"], r["number"]) == (True, False, False),
            f"changed title -> {r['changed']!r}, handler runs {r['handler_after_change']}; other bindings "
            f"{r['bindings']}; digest equal under key order {r['key_order']}, NFC/NFD {r['unicode']}, "
            f"1/1.0 {r['number']}",
        ),
        practice.Check(
            "FINDING: the approval is not one-time: the unchanged action is allowed 5 of 5 times",
            r["replays"] == ["PROJ-99"] * 5 and r["handler_after_replays"] == 5,
            f"5 replays -> {r['replays']}; handler runs {r['handler_after_replays']}",
        ),
        practice.Check(
            "FINDING: dispatch skips inputSchema, so an approved out-of-schema field reaches the handler",
            r["schema_extra"] is False and r["smuggled"] == ("PROJ-99", {"title": "x", "assignee": "root"}),
            f"additionalProperties={r['schema_extra']}; call -> {r['smuggled'][0]}, handler saw {r['smuggled'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
