"""Exercise 2 -- a lint gate blocks 23 of 30 green PRs, 12 of the 16 that kept full coverage, and every block comes from two recipes.

    Implement a "lint-clean" check: after migration, run a style linter (spotless for Java, ruff for Python). Fail the PR if new lint errors appear. Measure the coverage-preserved-but-style-regressed rate.

Reading of the exercise: the lesson's simulation has no source code, so
each of its 50 repos is paired with a generated Python 2 module
(`make_fixture`, seed 19: two imports and 8 snippets drawn from
`BODY_POOL`), migrated by the regex recipes in `RECIPES`, a stand-in for
libcst. Ruff cannot parse Python 2, so the linter is a stdlib stand-in for
two rules in ruff's default set, F401 (unused import) and E711 (comparison
to None), run on both sides. A PR that passes the lesson's `migrate`
(seed 19) fails the gate when the migrated file has an error the original
did not. "Coverage preserved" is read strictly: the migrated coverage is
not below the base. The rate depends on the snippet mix; which recipes
cause the errors does not.

**ANSWER: 12 of the 16 coverage-preserved passes regress on style (75%).**
Of all 30 green PRs the gate blocks 23. All 50 migrated files compile
under Python 3; the errors are 26 E711 and 14 F401 on `import string`.

**FINDING: the new errors come from two recipes.** `<>` to `!=` is
correct Python 3 and turns `result <> None` into an E711; rewriting
`string.join` and `string.upper` into str methods leaves `import string`
unused. With one more rule (`<> None` to `is not None`) the gate still
blocks 14 of the 30 passes, one F401 each.

**FINDING: 14 of the 30 passes lost coverage and still count as
preserved.** The lesson's gate fails a repo only below the base minus 2.0
points; the worst pass on seed 19 is -0.593. The doc says "dropped more
than 2%": over 1,000 seeds 55 passes dropped more than 2% of the base and
passed, while 8 repos were ever filed as `coverage_regression`.

Structure: `migrate_source()` applies the recipes; `lint()` is the two
rules; `new_errors()` is the multiset difference the gate fails on.
"""

from __future__ import annotations

import collections
import random
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "09-code-migration-agent"
SEED, BODY = 19, 8
HEADER = "import os\nimport string\n"
BODY_POOL = [  # Python 2 snippets a repo is built from
    "names = string.join(parts, ', ')", "title = string.upper(name)", "letters = string.ascii_letters",
    "home = os.path.expanduser('~')", 'print "done", count',
    "try:\n    value = int(raw)\nexcept ValueError, err:\n    value = 0", "for i in xrange(3):\n    total = i",
    "for k, v in table.iteritems():\n    last = k", "if table.has_key(key):\n    hit = 1",
    "if result <> None:\n    ok = 1",
]
RECIPES = [  # the deterministic Python 2 -> 3 rewrites (a libcst stand-in)
    (r"^(\s*)print (.+)$", r"\1print(\2)"), (r"except (\w+), (\w+):", r"except \1 as \2:"),
    (r"\bxrange\(", "range("), (r"\.iteritems\(\)", ".items()"), (r"(\w+)\.has_key\((\w+)\)", r"\2 in \1"),
    (r"string\.join\((\w+), ('[^']*')\)", r"\2.join(\1)"), (r"string\.upper\((\w+)\)", r"\1.upper()"),
    (r" <> ", " != "),
]
FIXED = [(r" <> None\b", " is not None")] + RECIPES  # the one-rule fix for the E711 regressions


def make_fixture(seed=SEED, repos=50):
    rng = random.Random(seed)
    return [HEADER + "\n".join(rng.choice(BODY_POOL) for _ in range(BODY)) + "\n" for _ in range(repos)]


def migrate_source(src, recipes=RECIPES):
    for pattern, repl in recipes:
        src = re.sub(pattern, repl, src, flags=re.M)
    return src


def lint(src):
    """Two rules from ruff's default set: F401 unused import, E711 comparison to None."""
    imports = re.findall(r"^import (\w+)$", src, flags=re.M)
    unused = [f"F401 {m}" for m in imports if not re.search(rf"\b{m}\.", src)]
    return unused + ["E711"] * len(re.findall(r"[!=]= None\b", src))


def new_errors(before, after):
    rest = list(before)
    return [e for e in after if e not in rest or rest.remove(e)]


def coverage_sweep(ref, seeds=1000):
    """How the code's 2-point gate compares with the doc's "dropped more than 2%"."""
    slipped = filed = 0
    for seed in range(seeds):
        rng = random.Random(seed)
        for a in (ref.migrate(repo, rng) for repo in ref.synth_bench(rng)):
            slipped += a.status == "pass" and a.coverage_final < 0.98 * a.coverage_base
            filed += a.failure_class == "coverage_regression"
    return {"slipped": slipped, "filed": filed}


def regressions(sources, recipes=RECIPES):
    """The gate: the new lint errors each migrated file has, and whether every file still compiles."""
    migrated = [migrate_source(src, recipes) for src in sources]
    compiled = sum(bool(compile(m, "<migrated>", "exec")) for m in migrated)
    return [new_errors(lint(s), lint(m)) for s, m in zip(sources, migrated)], compiled


def lesson_passes(ref):
    rng = random.Random(SEED)
    attempts = [ref.migrate(repo, rng) for repo in ref.synth_bench(rng)]
    return {i: a.coverage_final - a.coverage_base for i, a in enumerate(attempts) if a.status == "pass"}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sources = make_fixture()
    regressed, compiled = regressions(sources)
    fixed, _ = regressions(sources, FIXED)
    delta = lesson_passes(ref)
    strict = [i for i in delta if delta[i] >= 0]
    return {
        "compiled": compiled, "passes": len(delta),
        "gate_blocks": sum(bool(regressed[i]) for i in delta),
        "strict": len(strict), "strict_blocks": sum(bool(regressed[i]) for i in strict),
        "negative_delta": len(delta) - len(strict), "min_delta": round(min(delta.values()), 3),
        "kinds": dict(collections.Counter(e for i in delta for e in regressed[i])),
        "none_fixed_blocks": sum(bool(fixed[i]) for i in delta),
        **coverage_sweep(ref),
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: 12 of the 16 coverage-preserved passes regress on style (75%)",
            (r["strict"], r["strict_blocks"], r["passes"], r["gate_blocks"], r["compiled"]) == (16, 12, 30, 23, 50),
            f"{r['strict_blocks']}/{r['strict']} coverage-preserved passes gain lint errors; the gate "
            f"blocks {r['gate_blocks']}/{r['passes']} green PRs; {r['compiled']}/50 files compile",
        ),
        practice.Check(
            "FINDING: the new errors come from two recipes",
            r["kinds"] == {"E711": 26, "F401 string": 14} and r["none_fixed_blocks"] == 14,
            f"new errors on passes {r['kinds']}; with the `is not None` rule the gate blocks {r['none_fixed_blocks']}",
        ),
        practice.Check(
            "FINDING: 14 of the 30 passes lost coverage and still count as preserved",
            (r["negative_delta"], r["min_delta"], r["slipped"], r["filed"]) == (14, -0.593, 55, 8),
            f"{r['negative_delta']} passes below base (worst {r['min_delta']}); 1,000 seeds: "
            f"{r['slipped']} passes dropped >2% of base, {r['filed']} filed as coverage_regression",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
