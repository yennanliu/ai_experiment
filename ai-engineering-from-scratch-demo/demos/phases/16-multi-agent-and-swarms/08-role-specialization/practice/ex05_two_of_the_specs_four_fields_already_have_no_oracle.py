"""Exercise 5 — two of the spec's four fields already have no oracle.

    PwC's 7× accuracy gain came from verification loops. Hypothesize three
    tasks where adding a verifier would not help — where deterministic
    checking of correctness is impossible or prohibitively expensive.

Reading of the exercise: hypothesize from this module rather than in the
abstract, because its own `Spec` already contains two fields with no oracle
and one whose oracle is three test cases -- three worked examples of the three
classes, in forty lines of code.

**ANSWER: unbounded properties, erased properties, and judgements.**

*Unbounded* -- `description` says "the sum of two integers", a claim over an
infinite domain. The verifier checks **3** points. No finite suite decides it,
and the gap is not a bug in the suite: a lookup table over those three points
passes every one of them.

*Erased* -- `signature` says `add_two(a: int, b: int) -> int`. Python
annotations are not enforced at runtime, so `def add_two(*args)` satisfies the
verifier and contradicts the spec. The property is real and the artifact does
not carry it at the point where checking happens. Checking it means a second
tool, not a longer test suite.

*Judgements* -- is the code readable, is the design right, is this what the
user meant. The module already has the shape: `CriticReport` exists precisely
because these cannot be `VerifierReport`, and its verdict is a boolean over
**3** substring tests.

The measurement that makes this concrete: **2** of the Spec's **4** fields are
read by **0** roles, and the two that are read are exactly the two with a
mechanical oracle -- a name to compare and a list of cases to execute.

**FINDING: the verifier's power is exactly the length of the test list.** It
runs `spec.tests`, of which there are **3**, and reports pass or fail. An
artifact that returns the right answer on those three and `a * b` everywhere
else is `passed=True`. Adding the verifier does not raise accuracy on the
task; it raises accuracy on the sample, and the two are only the same thing
when the sample determines the task.

**FINDING: `description` is the field the user actually wrote, and it is the
one nothing reads.** `planner` puts the user's wish there and every other
field is a literal, so the only part of the spec that varies with the request
is the only part with no reader. Whatever the 7x was measured on, it was not
this.

**FINDING: the critic is the module's own admission.** Its three checks -- a
`def`, a `return`, a matching name -- are all decidable, so it is a weak
verifier rather than a different kind of role. The lesson's "critic vs
verifier" distinction is real, and the shipped critic sits on the wrong side
of it: **0** of its checks require judgement.

Structure: `oracles()` classifies each Spec field by what could check it;
`probes()` runs the two artifacts that pass without being right.
"""

from __future__ import annotations

import dataclasses
import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "08-role-specialization"
WISH = "A function that returns the sum of two integers."
WIDENED = "def add_two(*args):\n    return sum(args)\n"
MEMORISED = ("def add_two(a, b):\n"
             "    if (a, b) == (1, 2):\n        return 3\n"
             "    if (a, b) == (10, 20):\n        return 30\n"
             "    if (a, b) == (-5, 5):\n        return 0\n"
             "    return a * b\n")


def oracles(ref):
    """Which Spec fields any role reads, and therefore which have a mechanical check."""
    sources = [inspect.getsource(ref.critic), inspect.getsource(ref.verifier)]
    fields = [field.name for field in dataclasses.fields(ref.Spec)]
    return {field: sum(f"spec.{field}" in text for text in sources) for field in fields}


def probes(ref, spec):
    """Two artifacts that pass every check the pipeline has and are not right."""
    verdicts = {}
    for label, code in (("widened", WIDENED), ("memorised", MEMORISED)):
        artifact = ref.Artifact(code=code)
        verdicts[label] = (ref.critic(spec, artifact).approved,
                           ref.verifier(spec, artifact).passed)
    return verdicts


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spec = ref.planner(WISH)
    read = oracles(ref)
    critic_source = inspect.getsource(ref.critic)
    return {
        "fields": len(read), "read": read,
        "unread": sorted(field for field, count in read.items() if count == 0),
        "checked": sorted(field for field, count in read.items() if count),
        "tests": len(spec.tests),
        "verdicts": probes(ref, spec),
        "description_is_wish": spec.description == WISH,
        "literal_fields": sum(getattr(spec, field) == getattr(ref.planner("x"), field)
                              for field in ("task_name", "signature", "tests")),
        "critic_checks": critic_source.count("notes.append"),
        "critic_decidable": all(token in critic_source
                                for token in ('"def" not in', '"return" not in',
                                              "task_name not in")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: unbounded properties, erased properties, and judgements",
            all([result["unread"] == ["description", "signature"],
                 result["checked"] == ["task_name", "tests"],
                 result["fields"] == 4, result["tests"] == 3]),
            f"{len(result['unread'])} of the {result['fields']} Spec fields are read by "
            f"0 roles -- {' and '.join(result['unread'])} -- and the "
            f"{len(result['checked'])} that are read are exactly those with a mechanical "
            f"oracle: a name to compare and {result['tests']} cases to execute",
        ),
        practice.Check(
            "FINDING: the verifier's power is exactly the length of the test list",
            all([result["verdicts"]["memorised"] == (True, True),
                 result["verdicts"]["widened"] == (True, True),
                 result["tests"] == 3]),
            f"an artifact returning the right answer on the {result['tests']} listed "
            "cases and a * b everywhere else is passed=True, as is one widened to "
            "*args -- the verifier raises accuracy on the sample, which equals accuracy "
            "on the task only when the sample determines it",
        ),
        practice.Check(
            "FINDING: description is the field the user wrote and the one nothing reads",
            all([result["description_is_wish"], result["literal_fields"] == 3,
                 "description" in result["unread"]]),
            f"planner puts the wish in description and leaves the other "
            f"{result['literal_fields']} fields as literals, so the only part of the "
            "spec that varies with the request is the only part with no reader",
        ),
        practice.Check(
            "FINDING: the critic is the module's own admission",
            all([result["critic_checks"] == 3, result["critic_decidable"]]),
            f"its {result['critic_checks']} checks -- a def, a return, a matching name "
            "-- are all decidable, so it is a weak verifier rather than a different kind "
            "of role; the lesson's critic-versus-verifier distinction is real and the "
            "shipped critic sits on the wrong side of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
