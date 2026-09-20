"""Exercise 2 — canonical JSON is the signature, and the card has no slot for it.

    Add a signed Agent Card. Sign with HMAC over the card's canonical JSON.
    Write a verifier and confirm it fails on a mutated card.

Reading of the exercise: "canonical JSON" is the whole security property, so
the verifier is tested against a mutation that changes the bytes and against
one that changes only their order -- the second must verify and the first must
not, or the signature is measuring formatting. Where the signature *lives* is
then forced by the card's own shape, since hashing a document that contains
its own signature is circular.

**ANSWER: HMAC-SHA256 over `json.dumps(card, sort_keys=True,
separators=(",", ":"))`, detached.** The card verifies; flipping
`capabilities.streaming` to `False` fails; and re-serializing the same card
with its keys in reverse order still verifies, because canonicalization sorts
them.

**FINDING: the signature cannot live in the card it signs.** The **7**
top-level keys are all content, and adding `signature` to the document
changes the bytes the signature covers -- so verification would have to
remove it again first, by a rule nothing in the card states. Detaching it is
not a style choice; the alternative needs an out-of-band convention.

**FINDING: a mutation anywhere in the tree is caught, including inside a
skill.** Changing the nested `skills[0].description` -- a field a reader
might treat as documentation -- fails verification exactly as changing the
`url` does. Canonical JSON flattens the whole document, so there is no
"cosmetic" region.

**FINDING: the card advertises capabilities the signature can now bind, and
one of them is a lie in this repo.** `capabilities.streaming` is **True**
while the module's only transport is an in-process call with **0** streaming
functions. Signing the card makes that claim attributable rather than true --
a signature proves who said it, not that it holds.

Structure: `sign` and `verify` are the pair; `mutate` returns a copy of the
card with one path changed, so every check feeds the same verifier.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "19-a2a-protocol"
SECRET = b"lesson-19-card-signing-key"


def canonical(card):
    return json.dumps(card, sort_keys=True, separators=(",", ":")).encode()


def sign(card, secret=SECRET):
    return hmac.new(secret, canonical(card), hashlib.sha256).hexdigest()


def verify(card, signature, secret=SECRET):
    return hmac.compare_digest(sign(card, secret), signature)


def mutate(card, path, value):
    """A copy of the card with one path changed."""
    changed = copy.deepcopy(card)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return changed


def reordered(card):
    """The same document, serialized with its keys reversed."""
    return json.loads(json.dumps({k: card[k] for k in reversed(list(card))}))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    card = copy.deepcopy(ref.WRITER_AGENT_CARD)
    signature = sign(card)
    embedded = {**card, "signature": signature}
    return {
        "verifies": verify(card, signature),
        "reordered": verify(reordered(card), signature),
        "streaming_flipped": verify(
            mutate(card, ["capabilities", "streaming"], False), signature),
        "url_changed": verify(mutate(card, ["url"], "https://evil.example/a2a"), signature),
        "skill_description": verify(
            mutate(card, ["skills"], [{**card["skills"][0], "description": "Anything."}]),
            signature),
        "wrong_key": verify(card, sign(card, b"other-key")),
        "top_level": sorted(card),
        "embedded_breaks": sign(embedded) != signature,
        "streaming_claimed": card["capabilities"]["streaming"],
        "streaming_functions": [n for n in dir(ref) if "stream" in n.lower()],
        "digest_len": len(signature),
    }


def verify_checks(result):
    return [
        practice.Check(
            "ANSWER: HMAC over canonical JSON, detached, and order-insensitive",
            all([result["verifies"], result["reordered"],
                 not result["streaming_flipped"], not result["wrong_key"],
                 result["digest_len"] == 64]),
            f"the card verifies against its own {result['digest_len']}-hex signature and "
            f"still verifies when re-serialized with its keys reversed, because "
            f"sort_keys canonicalizes. Flipping capabilities.streaming fails "
            f"({result['streaming_flipped']}), and so does a signature from another key",
        ),
        practice.Check(
            "FINDING: the signature cannot live in the card it signs",
            all([result["embedded_breaks"], len(result["top_level"]) == 7]),
            f"the card's {len(result['top_level'])} top-level keys are all content, and "
            "adding a signature field changes the bytes the signature covers -- so an "
            "embedded signature would have to be stripped again by a rule the card does not "
            "state. Detaching it is not a style choice",
        ),
        practice.Check(
            "FINDING: a mutation anywhere in the tree is caught, including inside a skill",
            all([not result["skill_description"], not result["url_changed"]]),
            "changing the nested skills[0].description -- a field a reader might treat as "
            "documentation -- fails verification exactly as changing the url does. Canonical "
            "JSON flattens the whole document, so there is no cosmetic region",
        ),
        practice.Check(
            "FINDING: signing makes a claim attributable, not true",
            all([result["streaming_claimed"] is True,
                 result["streaming_functions"] == []]),
            f"capabilities.streaming is {result['streaming_claimed']} while the module has "
            f"{len(result['streaming_functions'])} streaming functions and an in-process "
            "transport. The signature binds the claim to whoever holds the key; it says "
            "nothing about whether the claim holds",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify_checks}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
