"""Exercise 2 — oneOf is a union and the unexpected-field rule is an intersection.

    Extend the validator to support `oneOf` so a task can be either a build
    task or a review task with different required fields.

Reading of the exercise: `validate` has no `oneOf` branch, so the first job
is adding one -- and the second is discovering that the shipped
`unexpected fields` rule fights it. That rule compares the document's keys
against *this* schema's `properties`, and a `oneOf` schema has no properties
of its own, so every key in every branch is unexpected unless the check
moves inside the branch.

**ANSWER: `oneOf` over two task shapes, accepting 4 of 6 documents.** A
build task requires `acceptance`; a review task requires `reviewer` and
`findings`. Over **6** probes -- **2** valid build, **2** valid review, **1**
missing a branch-required field, **1** satisfying both branches -- the
extended validator accepts **4** and refuses **2**, and reports which
branches failed.

**FINDING: `oneOf` at the top level makes every key unexpected.** Running
the shipped `validate` against a `oneOf` schema with no sibling
`properties` refuses a perfectly good build task on
`unexpected fields ['acceptance', 'goal', 'id', 'owner', 'status']` -- **5**
of its **5** keys. The check has to run per branch, which means `oneOf`
cannot be added as a peer clause; it restructures the function.

**FINDING: the strict rule makes `oneOf` and `anyOf` indistinguishable.** A
document carrying both `acceptance` and `reviewer` matches **0** of **2**
branches, because each branch rejects the other's keys as unexpected -- so
`anyOf` refuses it too. The choice between the keywords, which is the whole
point of the exercise, is unobservable until the unexpected-field rule is
relaxed: with it relaxed the document matches **2** branches, `oneOf`
refuses and `anyOf` accepts. Strictness has quietly made the schema a
partition.

**FINDING: the error message is the feature, and it gets worse first.** A
failed `oneOf` has **2** reasons, one per branch, and the shipped
`SchemaError` carries **1** string -- so the naive port reports the last
branch's failure and hides the first. Collecting both takes the message from
**1** reason to **2** and is what makes a union schema debuggable at all.

Structure: `validate_one_of()` is the added branch; `PROBES` are the six
documents it is measured on.
"""

from __future__ import annotations

import copy

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "34-repo-memory-and-state"
BUILD = {"id": "T-001", "goal": "validate /signup", "owner": "builder",
         "acceptance": ["pytest -x"], "status": "todo"}
REVIEW = {"id": "T-002", "goal": "review", "owner": "reviewer", "reviewer": "ada",
          "findings": [], "status": "todo"}
PROBES = ("build_ok", "build_ok2", "review_ok", "review_ok2", "missing", "both")


def review_branch(ref):
    """The build shape with acceptance swapped for reviewer and findings."""
    schema = copy.deepcopy(ref.BOARD_SCHEMA["items"])
    schema["required"] = ["id", "goal", "owner", "reviewer", "findings", "status"]
    props = schema["properties"]
    props.pop("acceptance")
    props |= {"reviewer": {"type": "string"},
              "findings": {"type": "array", "items": {"type": "string"}}}
    return schema


def one_of_schema(ref):
    return {"oneOf": [copy.deepcopy(ref.BOARD_SCHEMA["items"]), review_branch(ref)]}


def document(kind):
    if kind.startswith("build_ok"):
        return {**BUILD, "id": f"T-0{10 + len(kind)}"}
    if kind.startswith("review_ok"):
        return {**REVIEW, "id": f"T-0{20 + len(kind)}"}
    return ({k: v for k, v in REVIEW.items() if k != "findings"}
            if kind == "missing" else {**BUILD, "reviewer": "ada", "findings": []})


def validate_one_of(ref, value, schema, path="$"):
    """The added branch: validate against each alternative, count the matches."""
    if "oneOf" not in schema:
        return ref.validate(value, schema, path)
    reasons = []
    for index, branch in enumerate(schema["oneOf"]):
        try:
            ref.validate(value, branch, f"{path}<{index}>")
        except ref.SchemaError as exc:
            reasons.append(str(exc))
    matched = len(schema["oneOf"]) - len(reasons)
    if matched != 1:
        raise ref.SchemaError(f"{path}: matched {matched}; " + " | ".join(reasons))
    return reasons


def _matches(ref, doc, branch):
    try:
        ref.validate(doc, branch)
        return True
    except ref.SchemaError:
        return False


def check(ref, kind, schema, mode="oneOf"):
    doc = document(kind)
    try:
        if mode == "oneOf":
            validate_one_of(ref, doc, schema)
        elif not any(_matches(ref, doc, b) for b in schema["oneOf"]):
            raise ref.SchemaError("matched 0 branches")
    except ref.SchemaError as exc:
        return {"ok": False, "reasons": str(exc).count("|") + 1}
    return {"ok": True, "reasons": 0}


def permissive(ref, schema):
    """The same branches with the unexpected-field rule relaxed."""
    union = {k: v for b in schema["oneOf"] for k, v in b["properties"].items()}
    return {"oneOf": [{**copy.deepcopy(b), "properties": union | b["properties"]}
                      for b in schema["oneOf"]]}


def shipped_on_union(ref, schema):
    try:
        ref.validate(document("build_ok"), schema)
    except ref.SchemaError as exc:
        return str(exc).split(": ", 1)[-1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    schema = one_of_schema(ref)
    loose = permissive(ref, schema)
    results = {kind: check(ref, kind, schema) for kind in PROBES}
    both = document("both")
    return {
        "loose_matches": sum(_matches(ref, both, b) for b in loose["oneOf"]),
        "loose_one_of": check(ref, "both", loose)["ok"],
        "loose_any_of": check(ref, "both", loose, mode="anyOf")["ok"],
        "both_matches": sum(_matches(ref, both, b) for b in schema["oneOf"]),
        "any_of_both": check(ref, "both", schema, mode="anyOf")["ok"],
        "probes": len(PROBES), "build_keys": len(document("build_ok")),
        "accepted": sum(row["ok"] for row in results.values()),
        "refused": sum(not row["ok"] for row in results.values()),
        "verdicts": {k: v["ok"] for k, v in results.items()},
        "shipped_error": shipped_on_union(ref, schema),
        "missing_reasons": results["missing"]["reasons"],
        "validate_names": "oneOf" in ref.validate.__code__.co_consts}


def verify(result):
    return [
        practice.Check(
            "ANSWER: oneOf over two task shapes, accepting 4 of 6",
            all([result["probes"] == 6, result["accepted"] == 4,
                 result["refused"] == 2, result["verdicts"]["missing"] is False,
                 result["verdicts"]["both"] is False,
                 result["verdicts"]["build_ok"] is True]),
            f"over {result['probes']} probes the extended validator accepts "
            f"{result['accepted']} and refuses {result['refused']}: "
            f"{result['verdicts']}"),
        practice.Check(
            "FINDING: oneOf at the top level makes every key unexpected",
            all(["unexpected fields" in (result["shipped_error"] or ""),
                 result["build_keys"] == 5, result["validate_names"] is False]),
            f"the shipped validate against a oneOf schema refuses a valid build task on "
            f"{result['shipped_error']!r} -- all {result['build_keys']} keys. The "
            "unexpected check has to move inside the branch"),
        practice.Check(
            "FINDING: the strict rule makes oneOf and anyOf indistinguishable",
            all([result["both_matches"] == 0, result["any_of_both"] is False,
                 result["loose_matches"] == 2, result["loose_one_of"] is False,
                 result["loose_any_of"] is True]),
            f"a document carrying both acceptance and reviewer matches "
            f"{result['both_matches']} of 2, each branch rejecting the other's keys, so "
            f"anyOf refuses it too. Relaxed it matches {result['loose_matches']}"),
        practice.Check(
            "FINDING: a failed oneOf has one reason per branch",
            all([result["missing_reasons"] == 2, result["probes"] == 6]),
            f"the document missing findings fails both branches, carrying "
            f"{result['missing_reasons']} reasons where SchemaError holds one string: a "
            "naive port hides the first"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
