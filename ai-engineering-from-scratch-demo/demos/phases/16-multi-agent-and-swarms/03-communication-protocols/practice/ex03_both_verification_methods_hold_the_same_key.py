"""Exercise 3 — both verification methods hold the same key.

    **DID rotation.** Add key rotation to the `IdentityRegistry`. An agent should
    be able to publish a new DID document with updated keys while maintaining
    a `previousDid` reference. Verifiers should accept signatures from both
    the current and previous key during a grace period.

Reading of the exercise: implement the rotation, then ask the shipped document
which keys it is rotating -- because it declares two verification methods and
puts one key in both of them, so "updated keys" is one decision wearing two
names.

**ANSWER: rotation is a new document, a `previousDid` edge and a deadline the
verifier reads.** Inside the grace window a registry that walks the
`previousDid` chain accepts **2** of **2** signatures -- one made with the
retired key and one with the new. After the window it accepts **1** of **2**,
rejecting the retired key while still resolving the old document, which is
what makes the window a policy rather than a fact about storage.

**FINDING: the two verification methods hold the same public key.**
`createIdentity` computes `publicKeyDer` **1** time, from the Ed25519 signing
key, and writes it into **2** entries: `#key-1`, typed
`Ed25519VerificationKey2020`, and `#key-x25519-1`, typed
`X25519KeyAgreementKey2019`. The key-agreement slot holds a signing key and
says otherwise in its own `type` field. Rotating "the keys" therefore changes
both at once, or neither, and the document gives a verifier no way to tell
which of the two it just replaced.

**FINDING: the human-authorization control is structurally dead.**
`humanAuthorization` is built as `[]` in the only place a document is
constructed and pushed to **0** times anywhere. `requiresHumanAuth` returns
False for every one of the document's **2** key ids, so the operation class
ANP defines to require a person cannot be expressed by any document this
module can produce.

**FINDING: the signature authenticates nothing about the payload.**
`delegateTask` verifies over `message.id` -- a `crypto.randomUUID()` minted at
message construction. Replacing every part of a message while keeping its id
leaves the signature valid: the port verifies the tampered message as readily
as the original. What is signed is the envelope's serial number.

Structure: `identity()` ports `createIdentity`; `Registry` adds the rotation
chain the exercise asks for. Signing is HMAC, a stand-in -- every finding here
is about which bytes are covered, not about the primitive covering them.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "03-communication-protocols"
GRACE = 60


def identity(domain, name, secret):
    """createIdentity, ported: one public key written into two verification methods."""
    did = f"did:wba:{domain}:agent:{name}"
    public = hashlib.sha256(secret).hexdigest()
    methods = [{"id": f"{did}#key-1", "type": "Ed25519VerificationKey2020",
                "publicKeyDer": public},
               {"id": f"{did}#key-x25519-1", "type": "X25519KeyAgreementKey2019",
                "publicKeyDer": public}]
    return {"did": did, "secret": secret, "document": {
        "id": did, "verificationMethod": methods, "humanAuthorization": [],
        "authentication": [methods[0]["id"]], "keyAgreement": [methods[1]["id"]]}}


def sign(secret, payload):
    """A stand-in for crypto.sign: what matters is which bytes go in."""
    return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()


class Registry:
    """IdentityRegistry plus the rotation chain and grace period exercise 3 asks for."""

    def __init__(self):
        self.documents, self.secrets, self.retired = {}, {}, {}

    def publish(self, agent, previous=None, at=0):
        self.documents[agent["did"]] = agent["document"]
        self.secrets[agent["did"]] = agent["secret"]
        if previous:
            agent["document"]["previousDid"] = previous["did"]
            self.retired[previous["did"]] = at + GRACE

    def verify(self, did, signature, payload, now=0):
        """Accept the current key, and a retired one while its window is open."""
        while did in self.documents:
            if hmac.compare_digest(sign(self.secrets[did], payload), signature):
                return self.retired.get(did, float("inf")) > now
            did = self.documents[did].get("previousDid")
        return False

    def requires_human_auth(self, did, key_id):
        return key_id in self.documents.get(did, {}).get("humanAuthorization", [])


def rotation():
    """Rotate once, then verify an old and a new signature inside and outside the window."""
    old = identity("acme.test", "researcher", b"key-one")
    new = identity("acme.test", "researcher-v2", b"key-two")
    registry = Registry()
    registry.publish(old)
    registry.publish(new, previous=old, at=0)
    signed = [sign(who["secret"], "payload") for who in (old, new)]
    accepts = lambda when: sum(registry.verify(new["did"], s, "payload", when)
                               for s in signed)
    return registry, old, {"inside": accepts(GRACE - 1), "after": accepts(GRACE + 1),
                           "chain": "previousDid" in new["document"]}


def tampering(registry, agent):
    """Sign the message id, then rewrite the message body and verify again."""
    message = {"id": "m-7f3a", "parts": ["please summarise the Q3 filing"]}
    signature = sign(agent["secret"], message["id"])
    message["parts"] = ["wire the balance to account 9912"]
    return registry.verify(agent["did"], signature, message["id"]), message["parts"][0]


def solve():
    src = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.ts").read_text("utf-8")
    start = src.index("function createIdentity(")
    create = src[start:src.index("\nfunction signPayload", start)]
    registry, old, rotated = rotation()
    tampered, became = tampering(registry, old)
    methods = old["document"]["verificationMethod"]
    return {
        **rotated, "tampered_ok": tampered, "became": became,
        "computed": create.count("publicKeyDer ="), "written": create.count("publicKeyDer,"),
        "shared": len({m["publicKeyDer"] for m in methods}), "methods": len(methods),
        "types": [m["type"] for m in methods],
        "human_literal": re.findall(r"humanAuthorization: (\[[^\]]*\])", src),
        "human_pushes": src.count("humanAuthorization.push"),
        "human_true": sum(registry.requires_human_auth(old["did"], m["id"])
                          for m in methods),
        "verifies_over": re.search(r"verify\(fromDid, signature, ([^)]+)\)", src).group(1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: rotation is a new document, a previousDid edge and a deadline",
            all([result["chain"], result["inside"] == 2, result["after"] == 1]),
            f"inside the window a registry walking the previousDid chain accepts "
            f"{result['inside']} of 2 signatures, retired and new; after it, "
            f"{result['after']} of 2 -- the old document still resolves, so the window "
            "is a policy rather than a fact about storage",
        ),
        practice.Check(
            "FINDING: the two verification methods hold the same public key",
            all([result["computed"] == 1, result["written"] == 2,
                 result["shared"] == 1, result["methods"] == 2,
                 result["types"] == ["Ed25519VerificationKey2020",
                                     "X25519KeyAgreementKey2019"]]),
            f"createIdentity computes publicKeyDer {result['computed']} time and writes "
            f"it into {result['written']} entries, so the {result['methods']} methods "
            f"share {result['shared']} key -- the {result['types'][1]} slot holds a "
            "signing key and says otherwise in its own type field",
        ),
        practice.Check(
            "FINDING: the human-authorization control is structurally dead",
            all([result["human_literal"] == ["[]"], result["human_pushes"] == 0,
                 result["human_true"] == 0]),
            f"humanAuthorization is built as {result['human_literal'][0]} in the only "
            f"constructor and pushed to {result['human_pushes']} times, so "
            f"requiresHumanAuth is true for {result['human_true']} of the document's "
            f"{result['methods']} key ids",
        ),
        practice.Check(
            "FINDING: the signature authenticates nothing about the payload",
            all([result["verifies_over"] == "message.id", result["tampered_ok"],
                 "account 9912" in result["became"]]),
            f"delegateTask verifies over {result['verifies_over']}, a randomUUID minted "
            f"at construction; rewriting every part to {result['became']!r} but keeping "
            "the id leaves the signature valid -- what is signed is a serial number",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
