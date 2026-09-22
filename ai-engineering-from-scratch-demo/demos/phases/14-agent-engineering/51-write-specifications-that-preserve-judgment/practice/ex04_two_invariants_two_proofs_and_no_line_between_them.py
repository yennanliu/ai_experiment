"""Exercise 4 — two invariants, two proofs, and no line between them.

    Add a proof receipt for every invariant.

Reading of the exercise: the shipped specification has **2** invariants and
**2** proofs, which looks like a receipt each until you ask which covers
which. Both lists are `list[str]` and nothing joins them, so the pairing is
in the reader's head.

**ANSWER: pairing them by hand covers 1 of the 2 invariants and exposes a
proof that belongs to neither.** "Diagnosis is read-only" is covered by "zero
production writes"; "every source is included in the audit record" is covered
by nothing, because "ten recorded incident replays" is a proof of the outcome
-- did it find the service -- rather than of the invariant. The gap is
visible only once the link is written down.

**FINDING: the lesson's own proof ladder says the replay is the wrong
layer.** The docs list unit, wire, journey, replay and audit-log proofs and
warn against accepting a lower layer for a higher claim. The uncovered
invariant is about the audit record, and the proof that would close it --
an audit log -- is **1** of the **5** named layers and **0** of the **2**
proofs shipped.

**FINDING: a proof per invariant changes the shape of the artifact, not just
its contents.** Recording the link means a proof is a mapping rather than a
list; here that turns **2** strings into **2** pairs plus **1** unmatched
entry. `validate` checks both lists are non-empty and would report **0**
issues for a specification whose proofs prove something else entirely.

**FINDING: an invariant with no proof is indistinguishable from one with
three.** `compile_contract` returns the contract with both lists intact and
**0** keys relating them, so a reviewer counting receipts gets the same
document whether coverage is **1** of **2** or **2** of **2**.

Structure: `COVERAGE` is the hand-written pairing; `coverage()` reports what
it leaves uncovered.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "51-write-specifications-that-preserve-judgment"
LAYERS = ["unit test", "wire test", "browser journey", "replay set", "audit log"]
# invariant -> the shipped proof that actually covers it
COVERAGE = {
    "diagnosis is read-only": "zero production writes",
    "every source is included in the audit record": "",
}
NEEDED = "an audit log listing every source consulted per replay"


def coverage(spec):
    rows = []
    for invariant in spec.invariants:
        proof = COVERAGE.get(invariant, "")
        rows.append({"invariant": invariant, "proof": proof,
                     "covered": bool(proof) and proof in spec.proof})
    used = {row["proof"] for row in rows if row["covered"]}
    return rows, [proof for proof in spec.proof if proof not in used]


def repaired(ref, spec):
    """The same specification with a receipt for the uncovered invariant."""
    return ref.Specification(spec.outcome, list(spec.invariants), list(spec.examples),
                             list(spec.non_goals), list(spec.decisions),
                             list(spec.proof) + [NEEDED])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spec = ref.example()
    rows, unmatched = coverage(spec)
    fixed = repaired(ref, spec)
    contract = ref.compile_contract(spec)
    stripped = ref.Specification(spec.outcome, list(spec.invariants), list(spec.examples),
                                 list(spec.non_goals), list(spec.decisions),
                                 ["a screenshot of the dashboard"])
    return {
        "invariants": len(spec.invariants), "proofs": len(spec.proof),
        "covered": sum(row["covered"] for row in rows),
        "uncovered": [row["invariant"] for row in rows if not row["covered"]],
        "unmatched": unmatched,
        "layers": len(LAYERS), "audit_layer": "audit log" in LAYERS,
        "audit_in_proofs": sum("audit" in proof for proof in spec.proof),
        "needed": NEEDED in fixed.proof, "fixed_proofs": len(fixed.proof),
        "issues": ref.validate(spec), "stripped_issues": ref.validate(stripped),
        "keys": sorted(contract),
        "relates": any("cover" in key or "receipt" in key for key in contract),
        "proof_type": ref.Specification.__annotations__["proof"],
        "checks_lists": inspect.getsource(ref.validate).count("if not getattr"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: pairing covers 1 of the 2 invariants and leaves a stray proof",
            all([result["invariants"] == 2, result["proofs"] == 2,
                 result["covered"] == 1,
                 result["uncovered"] == ["every source is included in the audit record"],
                 result["unmatched"] == ["ten recorded incident replays"]]),
            f"{result['covered']} of {result['invariants']} invariants has a matching "
            f"proof; {result['uncovered']} has none and {result['unmatched']} belongs to "
            "the outcome rather than to any invariant",
        ),
        practice.Check(
            "FINDING: the lesson's own ladder says the replay is the wrong layer",
            all([result["layers"] == 5, result["audit_layer"] is True,
                 result["audit_in_proofs"] == 0, result["needed"] is True,
                 result["fixed_proofs"] == 3]),
            f"the docs name {result['layers']} proof layers including an audit log, and "
            f"{result['audit_in_proofs']} of the shipped proofs is one; adding it takes the "
            f"list to {result['fixed_proofs']} entries and closes the gap",
        ),
        practice.Check(
            "FINDING: a proof per invariant changes the shape of the artifact",
            all([result["proof_type"] == "list[str]", result["issues"] == [],
                 result["stripped_issues"] == [], result["checks_lists"] == 1]),
            f"proof is a {result['proof_type']} and validate only tests the lists are "
            f"non-empty, so a specification whose proofs prove something else entirely "
            f"returns {len(result['stripped_issues'])} issues",
        ),
        practice.Check(
            "FINDING: an invariant with no proof looks like one with three",
            all([len(result["keys"]) == 6, result["relates"] is False]),
            f"compile_contract returns {result['keys']} and {0 if not result['relates'] else 1} "
            "keys relating invariants to proofs, so a reviewer counting receipts gets the "
            "same document at 1 of 2 coverage as at 2 of 2",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
