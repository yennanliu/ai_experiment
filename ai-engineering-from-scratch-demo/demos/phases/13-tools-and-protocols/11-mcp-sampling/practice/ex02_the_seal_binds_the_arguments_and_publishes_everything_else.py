"""Exercise 2 — the seal binds the arguments and publishes everything else.

    Change `audience` between the first call and retry. Explain why the sealed
    state blocks cross-request reuse.

Reading of the exercise: the explanation is the deliverable, so the mechanism
is measured rather than restated -- which field does the blocking, what else
it would have blocked, and what it does not block at all. The last of those is
the one worth finding: a seal is an integrity primitive, and the exercise's
word "sealed" invites the reading that the contents are hidden. They are not.

**WHY IT BLOCKS: the first response committed a digest of the arguments, and
the second request is checked against it.** `seal_request_state` writes
`argumentsDigest` into the body and HMACs the whole body with a secret the
client does not hold. On the retry, `verify_request_state` recomputes the
digest from the arguments *actually sent* and compares. Changing `audience`
changes the digest; the token cannot be re-signed without the secret; so the
token and the arguments can only travel together. That is cross-request reuse:
a token is valid for one argument set, one principal and one method, and for
nothing else.

**ANSWER: `-32602`, `requestState arguments mismatch`.**

**FINDING: the binding is to the arguments' meaning, not their spelling.** The
digest is over `json.dumps(sort_keys=True)`, so re-ordering the keys is
accepted while adding an unused key is rejected. A client may reformat and may
not add.

**FINDING: four things are bound, and they are checked in a fixed order.**
Integrity, then principal, then method, then arguments, then expiry — so a
token that is wrong in two ways reports only the earlier one. Replaying
another principal's token fails on `principal` before the arguments are ever
digested.

**FINDING: sealed means signed, not hidden.** The body is
`urlsafe_b64encode(json)` — the client can read `principal`, `phase`,
`expiresAt` and the digest without the secret, and on the second round it can
read the `picked` file list too. Integrity is not confidentiality, and
anything the server puts in this token is published to whoever holds it.

Structure: `unseal` decodes the body without the secret, and `retry` sends one
(arguments, token, principal) triple and reports which check refused it.
"""

from __future__ import annotations

import base64
import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "11-mcp-sampling"
ORIGINAL = {"audience": "developer"}
CHANGED = {"audience": "executive"}
REORDERED = {"tone": "plain", "audience": "developer"}
EXTENDED = {"audience": "developer", "tone": "plain"}
PICKS = '["README.md","server.py"]'


def unseal(token):
    body = token.split(".", 1)[0]
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def open_flow(ref, arguments, principal="user-42"):
    params = {"name": "summarize_repo", "arguments": dict(arguments), "_meta": ref.request_meta()}
    response = ref.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": params}, principal=principal)
    return response["result"]["requestState"]


def retry(ref, token, arguments, principal="user-42"):
    params = {"name": "summarize_repo", "arguments": dict(arguments),
              "_meta": ref.request_meta(), "requestState": token,
              "inputResponses": {"pick_files": {"role": "assistant",
                                                "content": {"type": "text", "text": PICKS}}}}
    response = ref.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                             "params": params}, principal=principal)
    if "error" in response:
        return response["error"]["message"]
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    token = open_flow(ref, ORIGINAL)
    body = unseal(token)
    reordered_token = open_flow(ref, REORDERED)
    second = unseal(retry_state(ref, token))
    return {
        "unchanged": retry(ref, token, ORIGINAL),
        "changed": retry(ref, token, CHANGED),
        "extended": retry(ref, token, EXTENDED),
        "reordered": retry(ref, reordered_token, dict(reversed(list(REORDERED.items())))),
        "other_principal": retry(ref, token, CHANGED, principal="user-99"),
        "body": sorted(body), "principal": body["principal"], "phase": body["phase"],
        "digest_matches": body["argumentsDigest"] == ref._arguments_digest(ORIGINAL),
        "digest_order_free": ref._arguments_digest(REORDERED) == ref._arguments_digest(
            dict(reversed(list(REORDERED.items())))),
        "second_body": sorted(second), "picked": second["picked"],
        "secret_needed": len(ref.SERVER_SECRET) > 0,
    }


def retry_state(ref, token):
    """Round two's token, so the published body can be read at both phases."""
    params = {"name": "summarize_repo", "arguments": dict(ORIGINAL),
              "_meta": ref.request_meta(), "requestState": token,
              "inputResponses": {"pick_files": {"role": "assistant",
                                                "content": {"type": "text", "text": PICKS}}}}
    response = ref.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": params})
    return response["result"]["requestState"]


def verify(result):
    return [
        practice.Check(
            "ANSWER: changing audience answers -32602 requestState arguments mismatch",
            all([result["changed"] == "requestState arguments mismatch",
                 result["unchanged"] is None, result["digest_matches"]]),
            f"the retry with {CHANGED} answers {result['changed']!r} while the same token with "
            f"{ORIGINAL} is accepted. The first response committed "
            "argumentsDigest = sha256 of the arguments and the retry recomputes it from what "
            "was actually sent, so the token and its arguments can only travel together",
        ),
        practice.Check(
            "FINDING: the binding is to the arguments' meaning, not their spelling",
            all([result["reordered"] is None, result["digest_order_free"],
                 result["extended"] == "requestState arguments mismatch"]),
            f"the digest is over json.dumps(sort_keys=True), so re-ordering the keys is "
            f"accepted and adding an unused one answers {result['extended']!r}. A client may "
            "reformat its arguments and may not add to them",
        ),
        practice.Check(
            "FINDING: four things are bound, and the earlier check wins",
            all([result["other_principal"] == "requestState principal mismatch",
                 sorted(result["body"]) == ["argumentsDigest", "expiresAt", "method",
                                            "phase", "principal"]]),
            f"the body binds {result['body']}, and a token replayed by another principal with "
            f"changed arguments answers {result['other_principal']!r} -- principal before "
            "arguments. A token wrong in two ways reports only the earlier one, so fixing it "
            "is iterative",
        ),
        practice.Check(
            "FINDING: sealed means signed, not hidden",
            all([result["principal"] == "user-42", result["phase"] == "pick",
                 "picked" in result["second_body"],
                 result["picked"] == ["README.md", "server.py"],
                 result["secret_needed"]]),
            f"the body is urlsafe_b64encode(json), so without the secret a client reads "
            f"principal={result['principal']!r}, phase={result['phase']!r} and the digest, "
            f"and on round two reads {result['second_body']} including "
            f"picked={result['picked']}. Integrity is not confidentiality: whatever the "
            "server puts in this token is published to whoever holds it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
