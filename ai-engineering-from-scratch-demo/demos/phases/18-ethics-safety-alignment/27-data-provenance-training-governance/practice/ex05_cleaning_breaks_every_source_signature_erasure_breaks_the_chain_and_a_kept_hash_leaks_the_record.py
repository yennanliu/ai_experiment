"""Exercise 5 — cleaning breaks every source signature, erasure breaks the chain, and a kept hash leaks the record.

    Sketch a training-data-provenance manifest that composes with the AB 2013
    fields and a C2PA-signed provenance chain for each dataset. Identify one
    technical and one legal barrier.

Reading of the exercise: the sketch is built and run. One manifest per
dataset has two parts. `ab2013` holds the reference's 12 fields under
normalised keys. `chain` is a list of C2PA-style actions (acquired,
filtered, ...), each committing to the dataset's Merkle root before and
after and signed over the previous signature. HMAC-SHA256 stands in for
C2PA's certificate signatures; the chaining is the same. The manifest
composes by derivation: item 9 (cleaning) is generated from the chain's
actions, so the two cannot drift. The toy dataset has 10 abstract records,
one of them personal data ("user [U-042] age 37").

**ANSWER: the manifest verifies end to end and renders through the
reference unchanged.** The 3-step chain verifies. Editing one record makes
verification fail at step 1 (acquired), the first step that commits to it. The 12
fields go through `render_markdown` with no "(missing)" line. 8 of the
reference's 12 field names are not usable as keys (they contain spaces,
parentheses or "§"), so the manifest keys them by the text before the
parenthesis.

**FINDING (technical barrier): cleaning breaks every source signature.**
C2PA binds a signature to exact bytes. After the whitespace strip, 1 of 10
records still hashes to the value its source signed; after the lowercase
step, 0 of 10. Only the pipeline's own re-signature vouches for the
transformation, and a hash cannot show the transformation was faithful.

**FINDING (legal barrier): erasure and an append-only chain conflict.**
Deleting the personal record to honour a GDPR erasure request changes the
Merkle root, so the chain fails verification at step 1 (acquired). Keeping the leaf
hash instead of the record lets the chain verify again, but the hash still
identifies the person. It is found by trying every age 0-120, on attempt 38.
A retained hash of low-entropy personal data is pseudonymous data under GDPR,
not anonymous.

Structure: `merkle()` and `sign_chain()` / `verify_chain()` are the chain;
`ab2013()` derives the 12 fields from the chain.
"""

from __future__ import annotations

import hashlib
import hmac
import json

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "27-data-provenance-training-governance"
KEY = b"pipeline-signing-key (stand-in for a C2PA certificate)"
RECORDS = [f"Record {i}: [PUBLIC-TEXT-{i}]  " for i in range(9)] + ["user [U-042] age 37"]
PERSONAL = 9


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def merkle(leaves):
    """Root over leaf hashes; a leaf may be kept as a bare hash after its record is erased."""
    level = [x if len(x) == 64 else h(x) for x in leaves]
    while len(level) > 1:
        level = [h("".join(level[i:i + 2])) for i in range(0, len(level), 2)]
    return level[0]


def steps(records):
    stripped = [r.strip() for r in records]
    clean = [r.lower() for r in stripped]
    return [("c2pa.acquired", records, records), ("c2pa.filtered: whitespace strip", records, stripped),
            ("c2pa.edited: lowercase", stripped, clean)], clean


def sign_chain(actions):
    chain, prev = [], ""
    for name, before, after in actions:
        body = {"action": name, "in": merkle(before), "out": merkle(after), "prev": prev}
        prev = hmac.new(KEY, json.dumps(body, sort_keys=True).encode(), "sha256").hexdigest()
        chain.append(dict(body, sig=prev))
    return chain


def verify_chain(chain, actions):
    """1-based number of the first step whose signature or commitment fails, or None."""
    for i, (step, fresh) in enumerate(zip(chain, sign_chain(actions)), 1):
        if step != fresh:
            return i
    return None


def normalise(field):
    return field.split(" (")[0]


def ab2013(ref, chain):
    cleaning = "; ".join(s["action"].split(": ")[1] for s in chain if ": " in s["action"])
    values = ["[SITE-A] public pages", "pretraining", "10 records", "text; unlabeled", "Y", "N",
              "Y", "N", f"{cleaning}, to normalise text", "2025-01; not ongoing", "2025-02", "N"]
    return {normalise(f): v for f, v in zip(ref.AB_2013_FIELDS, values)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    actions, clean = steps(RECORDS)
    chain = sign_chain(actions)
    fields = ab2013(ref, chain)
    tampered, _ = steps(RECORDS[:3] + ["Record 3: [EDITED]"] + RECORDS[4:])
    erased = [r for i, r in enumerate(RECORDS) if i != PERSONAL]
    kept_hash = RECORDS[:PERSONAL] + [h(RECORDS[PERSONAL])]
    guesses = [f"user [U-042] age {a}" for a in range(121)]
    rendered = ref.render_markdown({f: fields[normalise(f)] for f in ref.AB_2013_FIELDS})
    return {
        "steps": len(chain), "verifies": verify_chain(chain, actions),
        "tamper_at": verify_chain(chain, tampered),
        "bad_keys": sum(not f.isidentifier() for f in ref.AB_2013_FIELDS),
        "keys": len(fields), "item9": fields[normalise(ref.AB_2013_FIELDS[8])],
        "lines": rendered.count("\n- **"), "missing": rendered.count("(missing)"),
        "still_match": sum(h(a) == h(b) for a, b in zip(RECORDS, clean)),
        "strip_match": sum(h(a) == h(b) for a, b in zip(RECORDS, actions[1][2])),
        "erased_at": verify_chain(chain, steps(erased)[0]),
        "kept_ok": merkle(kept_hash) == chain[0]["in"],
        "attempt": next(i + 1 for i, g in enumerate(guesses) if h(g) == kept_hash[PERSONAL]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the manifest verifies end to end and renders through the reference unchanged",
            (result["steps"], result["verifies"], result["tamper_at"]) == (3, None, 1)
            and (result["bad_keys"], result["keys"], result["lines"], result["missing"]) == (8, 12, 12, 0)
            and result["item9"].startswith("whitespace strip; lowercase"),
            f"{result['steps']}-step chain fails at {result['verifies']}; one edited record fails at "
            f"step {result['tamper_at']}; {result['bad_keys']} of 12 reference names are not keys; "
            f"item 9 derived as '{result['item9']}'; {result['lines']} lines rendered, "
            f"{result['missing']} missing",
        ),
        practice.Check(
            "FINDING (technical barrier): cleaning breaks every source signature",
            (result["strip_match"], result["still_match"]) == (1, 0),
            f"records still hashing to their source value: {result['strip_match']} of 10 after the strip, "
            f"{result['still_match']} after the lowercase",
        ),
        practice.Check(
            "FINDING (legal barrier): erasure and an append-only chain conflict",
            result["erased_at"] == 1 and result["kept_ok"] and result["attempt"] == 38,
            f"erasing the personal record fails the chain at step {result['erased_at']}; keeping its "
            f"hash re-verifies: {result['kept_ok']}; the hash is reversed on guess {result['attempt']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
