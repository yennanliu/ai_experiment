"""Exercise 3 -- trimming cuts the agent's diff 34-60%, and every cut past the noise reverts a fix the tests never ran.

    Add a "minimal-diff" optimizer: after the agent's branch passes tests, trim unnecessary changes with a second pass. Report diff-size reduction.

Reading of the exercise: the lesson's `Attempt` carries no diff, so each of
the 19 repos the agent passed on `main()`'s seed 19 gets a generated
Python 2 module of five functions (`make_repo`) and the agent's branch for
it: every fix it needs plus noise (a renamed local, requoted strings,
comments). Four of the five functions are under test, the lesson's 80%
base coverage. The optimizer reverts changed lines back to the original
while the tests stay green, in two versions: one line at a time, and every
subset of the changed lines of one function at a time. Diff size is the
count of changed lines.

**ANSWER: the one-line pass cuts the diff from 214 lines to 142 (-33.6%),
the per-function pass to 85 (-60.3%); on Python 3.14, 135 (-36.9%) and 78
(-63.6%).** All 19 branches pass before trimming; there are 95 needed
fixes and 119 noise lines. The one-line pass keeps 57 noise lines: a
renamed local spans three lines, reverting any one of them alone breaks
the function, so all three stay. The per-function pass removes all 119.

**FINDING: both passes revert fixes in the untested function, and the
tests cannot see it.** 10 needed fixes (5 `has_key`, 3 `iteritems`, 2
`xrange`) are reverted because nothing calls that function, and in 10 of
the 19 repos the untested function then crashes. The four tested
functions still pass, so a coverage gate sees the same branch. The
per-function pass ends 10 lines below the 95 needed fixes; that gap is the
reverted fixes.

**FINDING: on Python 3.14 the Python 2 `except ValueError, err:` compiles,
so the trimmer reverts it too.** PEP 758 (https://peps.python.org/pep-0758/,
read 2026-09-29) allows unparenthesised exception lists, which turns the
line into `except (ValueError, err):`. It is a SyntaxError on 3.11-3.13
(measured on 3.12). On 3.14, 7 more fixes are reverted and 17 of the 19
repos break. The expected numbers are keyed on the interpreter version.

Structure: `make_repo()` builds five functions from `TEMPLATES` (a line is
a string the agent left alone or (Python 2, agent's line, label)), the
last one untested; `passes()` renders a branch with some lines reverted
and runs its tests; `one_by_one()` reverts single lines, `per_function()`
the largest green subset of each function's changed lines; `score()`
checks what they reverted against the labels.
"""

import collections
import functools
import itertools
import random
import sys

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "09-code-migration-agent"
SEED, FUNCS = 19, 5
PY314 = sys.version_info >= (3, 14)  # PEP 758: `except A, B:` parses from 3.14 on
EXPECT = {False: (142, 85, {"has_key": 5, "iteritems": 3, "xrange": 2}, 10),  # kept 1-line, per-fn; reverted; breaks
          True: (135, 78, {"except": 7, "has_key": 5, "iteritems": 3, "xrange": 2}, 17)}
TEMPLATES = {  # kind -> (lines, (test args, expected))
    "xrange": (["def {f}(n):", ("    total = 0", "    acc = 0", "noise"), ("    for i in xrange(n):",
                "    for i in range(n):", "need"), ("        total += i", "        acc += i", "noise"),
                ("    return total", "    return acc", "noise")], ((4,), 6)),
    "iteritems": (["def {f}(table):", ("    keys = []", "    keys = []  # py3", "noise"), ("    for k, v in table"
                   ".iteritems():", "    for k, v in table.items():", "need"), "        keys.append(k)",
                   "    return sorted(keys)"], (({"a": 1},), ["a"])),
    "has_key": (["def {f}(table, key):", ("    if table.has_key(key):", "    if key in table:", "need"),
                 ("        return 'hit'", '        return "hit"', "noise"), "    return 'miss'"],
                (({"k": 1}, "k"), "hit")),
    "except": (["def {f}(raw):", "    try:", "        return int(raw)", ("    except ValueError, err:",
                "    except ValueError as err:", "need"), ("        return -1", "        return -1  # py3", "noise")],
               (("x",), -1)),
    "print": (["def {f}(count):", "    if count < 0:", ('        print "negative", count', '        print("negative",'
               ' count)', "need"), "    return count"], ((2,), 2)),
}


def make_repo(rng):
    lines, calls = [], []
    for n, kind in enumerate(rng.choice(list(TEMPLATES)) for _ in range(FUNCS)):
        body, call = TEMPLATES[kind]
        for old, new, label in ((x, x, "") if isinstance(x, str) else x for x in body):
            lines.append((old.format(f=f"f{n}"), new.format(f=f"f{n}"), label, n, kind))
        calls.append((f"f{n}", *call))
    return lines, calls


def passes(lines, reverted, calls):
    source = "\n".join(old if i in reverted else new for i, (old, new, *_) in enumerate(lines)) + "\n"
    try:
        exec(compile(source, "<branch>", "exec"), namespace := {})
        return all(namespace[f](*args) == want for f, args, want in calls)
    except Exception:
        return False


def one_by_one(lines, changed, tests):
    keep_green = lambda rev, i: rev | {i} if passes(lines, rev | {i}, tests) else rev
    return functools.reduce(keep_green, changed, set())


def per_function(lines, changed, tests):
    reverted = set()
    for pool in ([i for i in changed if lines[i][3] == n] for n in range(FUNCS)):
        subsets = (set(g) for k in range(len(pool), 0, -1) for g in itertools.combinations(pool, k))
        reverted |= next((g for g in subsets if passes(lines, reverted | g, tests)), set())
    return reverted


def score(repos, trim):
    total = collections.Counter()
    for lines, calls in repos:
        changed = [i for i, line in enumerate(lines) if line[2]]
        reverted = trim(lines, changed, calls[:-1])
        total.update(changed=len(changed), kept=len(changed) - len(reverted),
                     need=[x[2] for x in lines].count("need"),
                     noise_kept=sum(lines[i][2] == "noise" for i in set(changed) - reverted),
                     untested_breaks=not passes(lines, reverted, calls[-1:]))
        total.update(f"wrong:{lines[i][4]}" for i in reverted if lines[i][2] == "need")
    return dict(total)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(SEED)
    attempts = [ref.migrate(repo, rng) for repo in ref.synth_bench(rng)]
    agent_passed = [a for a in attempts if a.status == "pass" and a.agent_turns > 0]
    repos = [make_repo(fixture) for fixture in [random.Random(SEED)] for _ in agent_passed]
    return {
        "repos": len(repos), "base_cov": sorted({a.coverage_base for a in agent_passed}),
        "branches_green": sum(passes(lines, set(), calls) for lines, calls in repos),
        "greedy": score(repos, one_by_one), "grouped": score(repos, per_function),
        "py2_except_compiles": passes([("try:\n    pass\nexcept ValueError, err:\n    pass",) * 2], set(), []),
    }


def verify(r):
    one, grp, (kept_one, kept_grp, reverted, breaks) = r["greedy"], r["grouped"], EXPECT[PY314]
    wrong = lambda row: {k[6:]: v for k, v in sorted(row.items()) if k.startswith("wrong:")}
    cut = lambda row: f"{row['changed']} -> {row['kept']} ({1 - row['kept'] / row['changed']:.1%} smaller)"
    return [
        practice.Check(
            "ANSWER: the one-line pass cuts the diff 214 -> 142 lines, the per-function pass to 85 (3.14: 135, 78)",
            (r["repos"], r["branches_green"], r["base_cov"], one["changed"], one["need"]) == (19, 19, [80.0], 214, 95)
            and (one["kept"], grp["kept"], one.get("noise_kept", 0), grp.get("noise_kept", 0))
            == (kept_one, kept_grp, 57, 0),
            f"1-line {cut(one)}, {one.get('noise_kept', 0)} noise kept; per-function {cut(grp)}; {one['need']} fixes",
        ),
        practice.Check(
            "FINDING: both passes revert fixes in the untested function, and the tests cannot see it",
            wrong(one) == wrong(grp) == reverted and one["untested_breaks"] == grp["untested_breaks"] == breaks,
            f"reverted fixes {wrong(grp)}; untested function crashes in {grp['untested_breaks']}/19",
        ),
        practice.Check(
            "FINDING: on Python 3.14 the Python 2 `except ValueError, err:` compiles, so it gets reverted",
            r["py2_except_compiles"] == PY314 and ("except" in wrong(grp)) == PY314,
            f"Python {sys.version_info[0]}.{sys.version_info[1]}: compiles {r['py2_except_compiles']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
