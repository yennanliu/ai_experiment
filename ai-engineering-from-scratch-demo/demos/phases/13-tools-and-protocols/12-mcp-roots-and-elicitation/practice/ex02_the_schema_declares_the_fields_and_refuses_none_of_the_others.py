"""Exercise 2 — the schema declares the fields and refuses none of the others.

    Add `url` capability negotiation and an out-of-band setup flow. Keep
    third-party credentials out of `inputResponses`.

Reading of the exercise: "keep credentials out" is a claim about what cannot
arrive, so it is tested by trying to put one in. The out-of-band flow is built
so the credential has a legitimate path that avoids the client -- the browser
hands it to the server's own side channel -- and then the illegitimate path is
attempted anyway, because a design is only as closed as its narrowest check.

**ANSWER: `url` is negotiated separately, and the completed flow carries a
signal rather than a secret.** A client advertising only `{"form": {}}` is
refused **-32021** for the url flow; advertising `{"url": {}}` receives a
`mode: "url"` elicitation with a setup URL; and the retry's `inputResponses`
hold **0** credential-shaped fields while the server's vault holds **1**.

**FINDING: the lesson's negotiation reads absence of detail as support.**
`supports_form_elicitation({"elicitation": {}})` is **True** — an empty dict
short-circuits to yes. A `url` predicate written the same way would claim url
support from a client that never mentioned url, so the new mode has to be
checked by presence of its own key, which is what makes the **-32021** above
reachable at all.

**FINDING: the URL carries a handle, not a secret.** The setup URL contains a
`secrets.token_urlsafe` handle and **0** characters of the credential, and the
handle is bound into the sealed `requestState`. A URL recovered from browser
history is worth nothing without the token, which the browser never saw.

**FINDING: nothing refuses an undeclared field, so the schema is a request and
not a filter.** `requestedSchema` names `completed`; a client that also sends
`api_key` has it delivered to the handler intact — **2** keys where **1** was
declared. Keeping credentials out of `inputResponses` therefore needs the
server to reject keys it did not ask for, not merely to avoid reading them.

Structure: `Vault` is the server's side channel, `setup_flow` is the two-round
url elicitation, and `declared_only` is the check the lesson does not have.
"""

from __future__ import annotations

import base64
import json
import secrets

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "12-mcp-roots-and-elicitation"
CREDENTIAL = "sk-live-3f9c2a77"
DECLARED = {"completed": {"type": "boolean"}}


class Vault:
    """Where the third party puts the credential: the server, not the client."""

    def __init__(self):
        self.handles, self.secrets = {}, {}

    def issue(self):
        handle = secrets.token_urlsafe(16)
        self.handles[handle] = "pending"
        return handle

    def deposit(self, handle, credential):
        """The browser's redirect lands here, out of band from the MCP client."""
        self.handles[handle] = "complete"
        self.secrets[handle] = credential


def supports_url_elicitation(capabilities):
    """Presence of the mode's own key -- not the lesson's empty-dict shortcut."""
    elicitation = capabilities.get("elicitation")
    return isinstance(elicitation, dict) and isinstance(elicitation.get("url"), dict)


def setup_flow(ref, vault, capabilities):
    """Round one of the url elicitation, or the capability refusal."""
    if not supports_url_elicitation(capabilities):
        raise ref.McpError(-32021, "missing required client capability",
                           {"requiredCapabilities": {"elicitation": {"url": {}}}})
    handle = vault.issue()
    state = {"phase": "await_setup", "principal": "user-42", "method": "tools/call",
             "argumentsDigest": ref._arguments_digest({}), "setupHandle": handle,
             "nonce": secrets.token_hex(16), "expiresAt": 2 ** 31}
    return {"resultType": "input_required", "requestState": ref.seal_request_state(state),
            "inputRequests": {"connect": {"method": "elicitation/create", "params": {
                "mode": "url", "url": f"https://setup.example/start?handle={handle}",
                "message": "Authorize the notes provider in your browser.",
                "requestedSchema": {"type": "object", "properties": DECLARED,
                                    "required": ["completed"]}}}}}


def unseal(token):
    body = token.split(".", 1)[0]
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def declared_only(content, declared=DECLARED):
    """The check the lesson does not make: refuse what was not asked for."""
    return sorted(set(content) - set(declared))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    vault = Vault()
    refused = None
    try:
        setup_flow(ref, vault, {"elicitation": {"form": {}}})
    except ref.McpError as exc:
        refused = (exc.code, exc.data["requiredCapabilities"])
    pending = setup_flow(ref, vault, {"elicitation": {"url": {}}})
    request = pending["inputRequests"]["connect"]["params"]
    handle = next(iter(vault.handles))
    vault.deposit(handle, CREDENTIAL)  # the browser's redirect, not the client's reply

    honest = {"completed": True}
    smuggled = {"completed": True, "api_key": CREDENTIAL}
    return {
        "refused": refused, "mode": request["mode"], "url": request["url"],
        "declared": sorted(request["requestedSchema"]["properties"]),
        "url_has_handle": handle in request["url"],
        "url_has_secret": CREDENTIAL in request["url"],
        "sealed_handle": unseal(pending["requestState"])["setupHandle"],
        "vault_secrets": len(vault.secrets), "status": vault.handles[handle],
        "honest_extra": declared_only(honest), "smuggled_extra": declared_only(smuggled),
        "smuggled_keys": len(smuggled),
        "empty_is_form": ref.supports_form_elicitation({"elicitation": {}}),
        "empty_is_url": supports_url_elicitation({"elicitation": {}}),
        "form_is_url": supports_url_elicitation({"elicitation": {"form": {}}}),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: url is negotiated separately, and the reply carries a signal not a secret",
            all([result["refused"] == (-32021, {"elicitation": {"url": {}}}),
                 result["mode"] == "url", result["declared"] == ["completed"],
                 result["honest_extra"] == [], result["vault_secrets"] == 1,
                 result["status"] == "complete"]),
            f"a client advertising only form is refused {result['refused'][0]} naming "
            f"{result['refused'][1]}, while one advertising url receives a "
            f"{result['mode']!r} elicitation declaring {result['declared']}. The honest reply "
            f"carries {result['honest_extra']} beyond that, and the credential reaches the "
            f"server's vault out of band -- {result['vault_secrets']} secret, handle "
            f"{result['status']!r}",
        ),
        practice.Check(
            "FINDING: the lesson's negotiation reads absence of detail as support",
            all([result["empty_is_form"], not result["empty_is_url"],
                 not result["form_is_url"]]),
            f"supports_form_elicitation({{'elicitation': {{}}}}) is {result['empty_is_form']} "
            "because an empty dict short-circuits to yes. Written the same way a url "
            "predicate would claim support from a client that never mentioned url, so the "
            "new mode is checked by presence of its own key -- which is what makes the "
            "-32021 reachable",
        ),
        practice.Check(
            "FINDING: the URL carries a handle, not a secret",
            all([result["url_has_handle"], not result["url_has_secret"],
                 result["sealed_handle"] in result["url"],
                 CREDENTIAL not in str(result["sealed_handle"])]),
            f"the setup URL is {result['url']} -- a token_urlsafe handle and no part of the "
            f"credential -- and the sealed state carries setupHandle={result['sealed_handle']!r}, "
            "the same one. A URL recovered "
            "from browser history is worth nothing without the token, which the browser "
            "never saw",
        ),
        practice.Check(
            "FINDING: nothing refuses an undeclared field, so the schema is a request",
            all([result["smuggled_extra"] == ["api_key"], result["smuggled_keys"] == 2,
                 result["declared"] == ["completed"]]),
            f"a client that also sends {result['smuggled_extra']} has it delivered intact -- "
            f"{result['smuggled_keys']} keys where {len(result['declared'])} was declared. "
            "Keeping credentials out of inputResponses needs the server to reject keys it "
            "did not ask for, not merely to avoid reading them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
