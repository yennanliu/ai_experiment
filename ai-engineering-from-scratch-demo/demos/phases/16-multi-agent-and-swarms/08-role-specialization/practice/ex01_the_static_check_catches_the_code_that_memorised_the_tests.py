"""Exercise 1 — the static check catches the code that memorised the tests.

    Run `code/main.py` and observe how the verifier catches the bug the critic
    missed. Add a static-analysis check (count occurrences of `return`) as an
    additional verifier. What does it catch that the runtime test misses?

Reading of the exercise: build the artifact that defeats the runtime test
before writing the check, because otherwise there is no way to tell whether
the check is catching anything -- and the artifact that defeats it is the one
the check was clearly designed for.

**ANSWER: it catches code that memorised the test cases.** An executor
returning

    if (a, b) == (1, 2): return 3
    if (a, b) == (10, 20): return 30
    if (a, b) == (-5, 5): return 0
    return a * b

passes the critic's **3** checks and all **3** of the verifier's tests. The
correct implementation has **1** `return`; this one has **4**. The runtime
test cannot catch it, and not because it is weak -- the tests are the *only*
oracle the verifier has, so an artifact that satisfies all of them is
indistinguishable from a correct one by construction. The static check works
by reading a property of the *text* rather than of the behaviour, which is a
different axis and the reason it adds anything at all.

**FINDING: the verifier is not a sandbox.** Its docstring says "Run the code
in a sandbox namespace", and the call is `exec(art.code, ns, ns)` with a plain
dict. CPython injects `__builtins__` into any globals mapping that lacks it,
so `import os` inside an artifact succeeds. The role the lesson describes as
the deterministic, trustworthy one is the only role that executes attacker-
controlled text, with nothing removed.

**FINDING: the verdict hides the critic whenever the verifier also fails.**
`run_pipeline` branches `if approved and passed` / `elif not passed` /
`elif not approved`. The second arm catches every case where the verifier
failed, so the third can only fire when the verifier *passed* -- a run where
both roles object prints "verifier blocked ship" and the critic's notes never
reach the verdict, although they were computed.

**FINDING: the planner ignores the wish it is given.** `planner(user_wish)`
returns a `Spec` whose `task_name`, `signature` and `tests` are literals;
**1** of its **4** fields, `description`, carries the argument through. Two
entirely different wishes produce the same specification.

Structure: `memoriser()` is the artifact that defeats the tests;
`static_check()` is the additional verifier the exercise asks for.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "08-role-specialization"
MEMORISED = ("def add_two(a, b):\n"
             "    if (a, b) == (1, 2):\n        return 3\n"
             "    if (a, b) == (10, 20):\n        return 30\n"
             "    if (a, b) == (-5, 5):\n        return 0\n"
             "    return a * b\n")


def static_check(artifact, limit=1):
    """The additional verifier: more than one return is a branch table."""
    returns = artifact.code.count("return")
    return returns <= limit, returns


def escapes(ref, spec):
    """Whether an artifact can reach the interpreter from inside the verifier."""
    artifact = ref.Artifact(code="import os\nMARKER = os.name\n"
                                 "def add_two(a, b):\n    return a + b\n")
    namespace = {}
    exec(artifact.code, namespace, namespace)  # noqa: S102 - the point of the check
    return "__builtins__" in namespace, "MARKER" in namespace


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spec = ref.planner("A function that returns the sum of two integers.")
    other = ref.planner("A function that formats a postal address.")
    memo = ref.Artifact(code=MEMORISED)
    good = ref.executor_correct(spec)
    builtins_present, imported = escapes(ref, spec)
    pipeline = inspect.getsource(ref.run_pipeline)
    branches = [line.strip().split(":")[0]
                for line in pipeline.splitlines() if line.strip().startswith(("if ", "elif "))]
    return {
        "critic_checks": inspect.getsource(ref.critic).count("notes.append"),
        "tests": len(spec.tests),
        "memo_critic": ref.critic(spec, memo).approved,
        "memo_verifier": ref.verifier(spec, memo).passed,
        "memo_static": static_check(memo), "good_static": static_check(good),
        "sandbox_claimed": "sandbox" in inspect.getsource(ref.verifier),
        "builtins_present": builtins_present, "imported": imported,
        "branches": branches,
        "same_spec": (spec.task_name, spec.signature, spec.tests)
                     == (other.task_name, other.signature, other.tests),
        "carries_wish": sum(getattr(spec, field) != getattr(other, field)
                            for field in ("task_name", "signature", "description")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: it catches code that memorised the test cases",
            all([result["memo_critic"], result["memo_verifier"],
                 result["memo_static"] == (False, 4), result["good_static"] == (True, 1),
                 result["critic_checks"] == result["tests"] == 3]),
            f"the memorising artifact passes the critic's {result['critic_checks']} "
            f"checks and all {result['tests']} tests while carrying "
            f"{result['memo_static'][1]} returns against the correct version's "
            f"{result['good_static'][1]}; the tests are the verifier's only oracle, so "
            "anything satisfying them is indistinguishable from correct",
        ),
        practice.Check(
            "FINDING: the verifier is not a sandbox",
            all([result["sandbox_claimed"], result["builtins_present"],
                 result["imported"]]),
            "the docstring says 'Run the code in a sandbox namespace' and the call is "
            "exec(art.code, ns, ns) with a plain dict; CPython injects __builtins__ into "
            "any globals mapping that lacks it, so import os inside an artifact succeeds "
            "-- the role described as trustworthy is the one executing untrusted text",
        ),
        practice.Check(
            "FINDING: the verdict hides the critic whenever the verifier also fails",
            all([len(result["branches"]) == 3,
                 result["branches"][1] == "elif not vrep.passed"]),
            f"the branches are {result['branches']}; the second catches every run where "
            "the verifier failed, so the third can only fire when the verifier passed -- "
            "a run where both roles object prints 'verifier blocked ship' and the "
            "critic's notes never reach the verdict",
        ),
        practice.Check(
            "FINDING: the planner ignores the wish it is given",
            all([result["same_spec"], result["carries_wish"] == 1]),
            f"two entirely different wishes produce the same task_name, signature and "
            f"tests; {result['carries_wish']} of the Spec's 4 fields carries the "
            "argument through, and it is description, which no role reads",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
