"""Exercise 4 -- the audience check stops a read-only token before the handler, but tools/list shows it the destructive tool.

    Mint a token for the read-only server and present it to the state-changing server. Prove audience validation fails before the handler runs.

Reading of the exercise: "minted for the read-only server" is the
lesson's `Token` with `audience = readonly.url`, from the issuer both
servers trust. It is presented to `build_destructive_server()` for
`jira.create`, in three strengths: as minted (read-only scopes), with
`jira:write` added, and with `jira:write` plus a valid approval record for
the exact action. "Before the handler runs" is measured by counting calls
to the wrapped handler and by the audit entry the lesson writes.

**ANSWER: all 3 variants are denied with "token audience does not match this
server", and the handler runs 0 times.** Audience is the second check in
`policy_decide`, after the issuer (shared, so it passes) and before scope and
approval, so adding the scope or the approval changes nothing. Each denial is
audited as `denied:token audience does not match this server`. The reverse
direction (a destructive-audience token at the read-only server) is denied the
same way. A token minted for the destructive server, with the approval, runs
the handler exactly once.

**FINDING: `tools/list` is not authorized at all.** `tools_list(meta)` takes
no token, so the read-only token holder lists the state-changing
`jira.create`, with `destructiveHint: true`, from the server that just
refused it.

**FINDING: every denial is the same generic `-32000`.** Wrong audience and an
unknown tool name both come back as code `-32000`; the spec classes an
unknown tool as a protocol error, `-32602` in its example
(modelcontextprotocol.io/specification/2026-07-28/server/tools, read
2026-09-29).
"""

from __future__ import annotations

import time

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "13-mcp-server-with-registry"
ARGS = {"title": "new bug"}


def counted(server, tool):
    calls = []
    real = server.handlers[tool]
    server.handlers[tool] = lambda args: calls.append(args) or real(args)
    return calls


def present(ref, server, token, tool, args, approval=None):
    audit = []
    reply = ref.dispatch(server, token, tool, args, ref.request_meta(), audit, approval)
    return reply.get("error", {}).get("message", reply.get("resultType")), [a.outcome for a in audit]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ro, rw = ref.build_readonly_server(), ref.build_destructive_server()
    ro_calls, rw_calls = counted(ro, "postgres.readonly"), counted(rw, "jira.create")
    later = time.time() + 3600

    def mint(aud, scopes):
        return ref.Token("u42", ro.trusted_issuer, aud, frozenset(scopes), later)

    ro_scopes = {"postgres:query:readonly", "s3:list"}
    approval = ref.ApprovalRecord.for_action("u42", "jira.create", ARGS, rw.url, later)
    wrong = {
        "as minted": present(ref, rw, mint(ro.url, ro_scopes), "jira.create", ARGS),
        "plus jira:write": present(ref, rw, mint(ro.url, ro_scopes | {"jira:write"}), "jira.create", ARGS),
        "plus approval": present(
            ref, rw, mint(ro.url, ro_scopes | {"jira:write"}), "jira.create", ARGS, approval
        ),
    }
    before = len(rw_calls)
    reverse = present(ref, ro, mint(rw.url, ro_scopes), "postgres.readonly", {"sql": "SELECT 1"})
    right = present(ref, rw, mint(rw.url, {"jira:write"}), "jira.create", ARGS, approval)
    listed = rw.tools_list(ref.request_meta())["tools"]
    unknown = ref.dispatch(rw, mint(rw.url, {"jira:write"}), "jira.delete", {}, ref.request_meta(), [])
    aud = ref.dispatch(rw, mint(ro.url, ro_scopes), "jira.create", ARGS, ref.request_meta(), [])
    return {
        "wrong": wrong, "handler_before": before, "reverse": reverse, "ro_calls": len(ro_calls),
        "right": right, "handler_after": len(rw_calls),
        "listed": [(t["name"], t["annotations"]["destructiveHint"]) for t in listed],
        "codes": (aud["error"]["code"], unknown["error"]["code"], unknown["error"]["message"]),
    }


def verify(result):
    r = result
    msg = "token audience does not match this server"
    return [
        practice.Check(
            "ANSWER: a read-only-audience token is refused by audience before the jira.create handler runs",
            all(v == (msg, [f"denied:{msg}"]) for v in r["wrong"].values())
            and r["handler_before"] == 0
            and (r["reverse"], r["ro_calls"]) == ((msg, [f"denied:{msg}"]), 0)
            and (r["right"], r["handler_after"]) == (("complete", ["allowed"]), 1),
            f"wrong-audience variants {r['wrong']}; handler runs {r['handler_before']}; reverse {r['reverse']}; "
            f"correct token {r['right']} -> handler runs {r['handler_after']}",
        ),
        practice.Check(
            "FINDING: tools/list is not authorized: the refused token still lists the destructive tool",
            r["listed"] == [("jira.create", True)],
            f"destructive server lists {r['listed']} to any caller with valid _meta",
        ),
        practice.Check(
            "FINDING: wrong audience and unknown tool both return the generic -32000",
            r["codes"] == (-32000, -32000, "no such tool: jira.delete"),
            f"(audience code, unknown-tool code, message) = {r['codes']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
