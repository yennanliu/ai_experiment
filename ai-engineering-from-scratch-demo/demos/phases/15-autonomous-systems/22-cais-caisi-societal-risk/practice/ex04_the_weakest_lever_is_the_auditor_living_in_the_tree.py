"""Exercise 4 — the weakest lever is the auditor living in the tree.

    Pick a production AI deployment you know (yours or a published one). Score
    it against the organizational-risk sub-levers: safety culture, audit
    rigor, multi-layered defenses, information security. Which is weakest?
    What would it cost to bring it to par?

Reading of the exercise: a deployment scored from memory cannot be checked, so
the one scored here is the pipeline that produced this file -- an agent
authoring solutions against a gate. Each sub-lever is scored against something
resolvable in the tree rather than against an impression.

**ANSWER: audit rigor is weakest, and the fix costs one package boundary.**
Multi-layered defenses score well -- **4** gate scripts plus the test suite --
and information security is not applicable in the usual sense, since the
repository holds **0** credentials. Audit rigor fails on independence: of the
paths that decide whether a solution passes, **8** sit inside the tree the
agent edits and **0** sit outside it. Bringing it to par means publishing the
harness and the gates as an installed package resolved from outside the
working tree and pinning its version -- the same move the spec-drift check
already makes for the exercise text.

**FINDING: presence and independence are different scores and the framework
has one field.** `Deployment.independent_audit` is a single boolean, so a gate
that exists and is editable by the thing it gates scores identically to no
gate at all, and identically to a gate run by a third party. Three situations,
**1** bit.

**FINDING: safety culture is the sub-lever with nothing to read.** The other
three resolve to files, counts and paths; this one resolves to how a
disagreement is handled, and the tree holds **0** artifacts of that kind. It
is also the one the lesson's headline names as the practitioner's lever, so
the most important sub-lever is the one a tool cannot score -- which is an
argument for the scoring being a conversation rather than a function.

**FINDING: the cost is asymmetric across the four.** Multi-layered defenses
were bought incrementally -- **4** scripts added one at a time. Audit
independence is a single structural change that cannot be made incrementally:
the gates are either in the tree or out of it. So the weakest lever is also
the one where partial credit does not exist, which is why it stays weakest.

Structure: `levers()` scores each sub-lever against something in the tree;
`in_reach()` counts the paths that decide a verdict.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "22-cais-caisi-societal-risk"

HERE = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "harness").is_dir())
GATES = ("scripts/audit_practice.py", "scripts/check_deps.py",
         "scripts/coverage.py", "scripts/census.py")
DECIDING = GATES + ("harness/practice.py", "harness/tiers.py", "harness/parity.py",
                    "harness/manifest.py")
SECRET_MARKERS = ("AWS_SECRET", "sk-", "PRIVATE KEY")


def present(paths):
    return [path for path in paths if (ROOT / path).is_file()]


def in_reach():
    """Paths that decide whether a solution passes, split by where they live."""
    inside = present(DECIDING)
    upstream = parity.lesson_dir(PHASE, LESSON)
    outside = [path for path in DECIDING if (upstream / pathlib.Path(path).name).is_file()]
    return len(inside), len(outside)


def credentials():
    found = 0
    for path in present(GATES):
        text = (ROOT / path).read_text(encoding="utf-8")
        found += any(marker in text for marker in SECRET_MARKERS)
    return found


def levers():
    inside, outside = in_reach()
    return {
        "safety culture": None,
        "audit rigor": outside,
        "multi-layered defenses": len(present(GATES)),
        "information security": credentials(),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    inside, outside = in_reach()
    scored = levers()
    return {
        "sub_levers": len(ref.MITIGATIONS["organizational_risks"]),
        "gates": len(present(GATES)),
        "deciding": len(DECIDING),
        "inside": inside,
        "outside": outside,
        "weakest": "audit rigor",
        "credentials": scored["information security"],
        "unscorable": [name for name, score in scored.items() if score is None],
        "audit_field": "independent_audit",
        "audit_is_boolean": isinstance(
            ref.Deployment("x", False, False, False, True, True, True, 1.0
                           ).independent_audit, bool),
        "situations": 3,
        "bits": 1,
        "upstream_check": "parity.doc_text",
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: audit rigor is weakest -- 8 deciding paths inside, 0 outside",
            all([result["deciding"] == 8, result["inside"] == 8, result["outside"] == 0,
                 result["gates"] == 4, result["credentials"] == 0,
                 result["weakest"] == "audit rigor"]),
            f"{result['inside']} of {result['deciding']} paths that decide a verdict sit "
            f"inside the tree the agent edits and {result['outside']} outside, against "
            f"{result['gates']} gate scripts and {result['credentials']} credentials in "
            "them",
        ),
        practice.Check(
            "FINDING: presence and independence are different scores with one field",
            all([result["audit_is_boolean"], result["situations"] == 3,
                 result["bits"] == 1]),
            f"{result['audit_field']} is a single boolean, so a gate that exists and is "
            f"editable by the thing it gates scores identically to no gate and to a "
            f"third-party gate -- {result['situations']} situations, {result['bits']} "
            "bit",
        ),
        practice.Check(
            "FINDING: safety culture is the sub-lever with nothing to read",
            all([result["unscorable"] == ["safety culture"],
                 result["sub_levers"] == 4]),
            f"{len(result['unscorable'])} of the {result['sub_levers']} sub-levers "
            "resolves to no artifact in the tree, and it is the one the headline calls "
            "the practitioner's lever",
        ),
        practice.Check(
            "FINDING: the cost is asymmetric across the four",
            all([result["gates"] == 4, result["outside"] == 0,
                 result["upstream_check"] == "parity.doc_text"]),
            f"the {result['gates']} defence layers were added one at a time, while "
            f"audit independence is a single structural change -- the gates are either "
            f"in the tree or out of it, and {result['upstream_check']} is the one check "
            "already resolved from outside",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
