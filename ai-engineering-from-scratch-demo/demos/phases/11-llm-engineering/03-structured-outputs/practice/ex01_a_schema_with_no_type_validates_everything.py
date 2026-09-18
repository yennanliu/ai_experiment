"""Exercise 1 — before oneOf exists, a schema without `type` validates everything.

    Extend the schema validator to support `oneOf` (the data must match exactly
    one of several schemas). This handles polymorphic outputs -- for example, a
    field that can be either a `Product` or a `Service` object with different
    shapes.

Reading of the exercise: `oneOf` is implemented on top of the lesson's own
`validate_schema`, so a branch matches exactly when the lesson's validator
reports no errors for it, and the union is scored against a labelled corpus of
Product-shaped, Service-shaped, both-shaped and neither-shaped documents.

**ANSWER: the extension is four lines, and the state it replaces is worse than
"unsupported".** `_validate` dispatches on `schema.get("type")` and has no
`else`, so a schema carrying only `oneOf` matches no branch and appends no
error: `validate_schema("anything", {"oneOf": [...]})` returns `[]` today.
Polymorphic schemas do not fail to validate -- they accept everything, silently.

**FINDING: with the extension, 5 of 12 labelled documents change verdict, and
all five were being accepted.** The shipped validator accepts 12 of 12. The
extension accepts the 7 that match exactly one branch, and rejects the 2 that
match neither and the 3 that match both.

**FINDING: "exactly one" is the hard half, and open objects break it.** The
lesson's validator ignores unknown keys, so a document with a Product's fields
*and* a Service's fields matches both branches, and so does any document whose
extra keys happen to satisfy the other branch's `required`. 3 of the 12 are
ambiguous for that reason, and `oneOf` rejects all three -- while a Product
carrying a stray `duration_days` survives, because Service also wants a `rate`.

**CONTROL: a discriminator fixes it and `additionalProperties` does not exist.**
Adding a required `kind` with a one-value `enum` to each branch takes the
ambiguous set from 3 to 0 and the accepted set from 7 to 10, because a document
can then satisfy only the branch it names. The lesson's validator has no
`additionalProperties` support at all, so closing the objects is not available.

Structure: `validate_one_of` is the extension, `CORPUS` the labelled documents,
`discriminated` the control schema.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "03-structured-outputs"
PRODUCT = {"type": "object",
           "properties": {"name": {"type": "string"}, "price": {"type": "number"}},
           "required": ["name", "price"]}
SERVICE = {"type": "object",
           "properties": {"name": {"type": "string"}, "rate": {"type": "number"},
                          "duration_days": {"type": "integer"}},
           "required": ["name", "rate"]}
# (document, how many branches it should match, label)
CORPUS = [
    ({"name": "Widget", "price": 9.99}, 1, "product"),
    ({"name": "Gadget", "price": 24.0}, 1, "product"),
    ({"name": "Audit", "rate": 250.0}, 1, "service"),
    ({"name": "Audit", "rate": 250.0, "duration_days": 3}, 1, "service"),
    ({"name": "Setup", "rate": 99.0, "duration_days": 1}, 1, "service"),
    ({"name": "Widget Pro", "price": 19.0}, 1, "product"),
    ({"name": "Bundle", "price": 9.99, "rate": 250.0}, 2, "both: priced and rated"),
    ({"name": "Kit", "price": 5.0, "rate": 1.0, "duration_days": 2}, 2, "both"),
    ({"name": "Plan", "price": 1.0, "rate": 2.0}, 2, "both"),
    ({"name": "Widget", "price": 9.99, "duration_days": 3}, 1, "product with a stray key"),
    ({"name": "Nameless"}, 0, "neither: no price and no rate"),
    ({"price": 9.99}, 0, "neither: no name"),
]


def matches(ref, document, schema):
    return not ref.validate_schema(document, schema)


def validate_one_of(ref, document, branches):
    """The extension: exactly one branch may accept, and the union is that rule."""
    hits = [i for i, branch in enumerate(branches) if matches(ref, document, branch)]
    if len(hits) == 1:
        return []
    if not hits:
        return [f": matches none of the {len(branches)} oneOf branches"]
    return [f": matches {len(hits)} oneOf branches {hits}, expected exactly one"]


def discriminated(schema, kind):
    """The control: a required one-value enum that only one branch can satisfy."""
    properties = dict(schema["properties"], kind={"type": "string", "enum": [kind]})
    return {**schema, "properties": properties, "required": [*schema["required"], "kind"]}


def tag(document, branches):
    """The document with the discriminator its intended branch would carry."""
    return dict(document, kind="product" if "price" in document else "service")


def run(ref, branches, corpus):
    hits = [sum(matches(ref, d, b) for b in branches) for d, _, _ in corpus]
    accepted = [not validate_one_of(ref, d, branches) for d, _, _ in corpus]
    return {"hits": hits, "accepted": sum(accepted),
            "ambiguous": sum(n > 1 for n in hits), "orphans": sum(n == 0 for n in hits)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    branches = [PRODUCT, SERVICE]
    bare = {"oneOf": branches}
    shipped = [not ref.validate_schema(d, bare) for d, _, _ in CORPUS]
    control_branches = [discriminated(PRODUCT, "product"), discriminated(SERVICE, "service")]
    control_corpus = [(tag(d, branches), n, label) for d, n, label in CORPUS]
    return {
        "shipped_accepts": sum(shipped), "corpus": len(CORPUS),
        "shipped_on_garbage": not ref.validate_schema("anything", bare),
        "expected": [n for _, n, _ in CORPUS],
        **run(ref, branches, CORPUS),
        "control": run(ref, control_branches, control_corpus),
    }


def verify(result):
    control = result["control"]
    return [
        practice.Check(
            "ANSWER: today a oneOf-only schema accepts everything, silently",
            all([result["shipped_accepts"] == result["corpus"], result["shipped_on_garbage"]]),
            f"`_validate` dispatches on schema.get('type') with no else, so a schema "
            f"carrying only `oneOf` matches no branch and appends no error. It accepts "
            f"{result['shipped_accepts']} of {result['corpus']} labelled documents -- "
            "including the two that match neither branch, and the string 'anything'",
        ),
        practice.Check(
            "ANSWER: the extension separates the corpus the way the labels say",
            all([result["hits"] == result["expected"], result["accepted"] == 7]),
            f"branch-match counts {result['hits']} equal the labelled counts, and "
            f"`oneOf` accepts the {result['accepted']} documents matching exactly one. "
            f"{result['shipped_accepts']} -> {result['accepted']} accepted: five "
            "documents change verdict, and all five were being let through",
        ),
        practice.Check(
            "FINDING: 'exactly one' is the hard half, because the objects are open",
            all([result["ambiguous"] == 3, result["orphans"] == 2]),
            f"{result['ambiguous']} of {result['corpus']} documents match both branches -- "
            f"the lesson's validator ignores unknown keys, so anything carrying both a "
            f"price and a rate satisfies Product and Service at once -- and "
            f"{result['orphans']} match neither. Those three are rejected for being too "
            "informative rather than for being malformed",
        ),
        practice.Check(
            "CONTROL: a discriminator takes the ambiguous set to zero",
            all([control["ambiguous"] == 0, control["accepted"] > result["accepted"]]),
            f"adding a required `kind` with a one-value enum to each branch moves "
            f"ambiguity {result['ambiguous']} -> {control['ambiguous']} and acceptance "
            f"{result['accepted']} -> {control['accepted']}, because a document can then "
            "satisfy only the branch it names",
        ),
        practice.Check(
            "FINDING: closing the objects is not available -- there is no additionalProperties",
            control["orphans"] == result["orphans"],
            "the other standard fix, additionalProperties: false, is not implemented by "
            f"`_validate` at all, so the {result['orphans']} documents matching no branch "
            "stay unmatched and the ambiguity has to be resolved by adding a field rather "
            "than by forbidding one. A discriminator is the only lever the lesson has",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
