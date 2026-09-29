"""Exercise 2 — code_exec with stdout.

    Extend `code_exec` to capture stdout and accept a list of expected stdouts as targets.

Reading of the exercise: `code_exec_stdout(prediction, targets, extras)` keeps
the lesson's metric contract and its stripped-builtins namespace, adds a
`print` that writes into a buffer, and scores stdout against `targets`. When
`extras["inputs"]` is given, `f(inputs[i])` is called and its stdout must
equal `targets[i]`, and the score is the fraction that match. With no inputs
the whole program runs once and its stdout must equal any one target. When
`io_pairs` is given it scores return values the way the lesson does, so the
five shipped `code-exec` fixtures run through it too. Stdout is compared
after stripping trailing whitespace from each line.

**ANSWER: the extended metric scores stdout from both per-call and
whole-program runs.** A correct FizzBuzz scores 1.0 against per-call targets
and a program printing "hello" scores 1.0 against ["hi", "hello"]. A
FizzBuzz that prints "Fizz" on 5 scores 0.667. On the five shipped fixtures
with the toy's predictions it gives the same 1.0 the lesson's metric gives.

**FINDING: the shipped namespace has no `print`,** so a prediction that
prints fails with NameError and scores 0.0 even when its return values are
right.

**FINDING: the shipped `exec(prediction, safe_globals, local)` splits
globals from locals.** Top-level `def`s land in `local`, but functions look
names up in `safe_globals`. A correct recursive factorial and an `f` that
calls a helper `g` both score 0.0. This metric runs the code in one
namespace and scores both 1.0.

**FINDING: stripped builtins are not a sandbox.** The lesson says a
prediction "cannot reach the filesystem". A prediction that walks
`().__class__.__base__.__subclasses__()` to `os._wrap_close` calls
`os.getcwd()` and scores 1.0 under both the shipped metric and this one.
"""

from __future__ import annotations

import io
import os

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "49-lm-eval-harness"
SAFE = {"range": range, "len": len, "min": min, "max": max, "abs": abs, "int": int, "float": float, "str": str}
FIZZ = "def f(n):\n    print('Fizz' if n % 3 == 0 else 'Buzz' if n % 5 == 0 else n)\n"
FIZZ_BAD = "def f(n):\n    print('Fizz' if n % 3 == 0 or n % 5 == 0 else n)\n"
HELLO = "print('hello')\n"
FACT = "def f(x):\n    return 1 if x < 2 else x * f(x - 1)\n"
HELPER = "def g(x):\n    return x * 2\ndef f(x):\n    return g(x)\n"
PRINTS = "def f(x):\n    print(x)\n    return x * 2\n"
ESCAPE = (
    "def f(x):\n    c = [k for k in ().__class__.__base__.__subclasses__() if k.__name__ == '_wrap_close'][0]\n"
    "    return c.__init__.__globals__['getcwd']()\n"
)


_FAILED = object()


def _clean(text):
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _call(fn, x, buf):
    buf.seek(0), buf.truncate()
    try:
        return fn(x), _clean(buf.getvalue())
    except Exception:
        return _FAILED, None


def code_exec_stdout(prediction, targets, extras):
    buf = io.StringIO()
    env = {"__builtins__": {**SAFE, "print": lambda *a, **k: print(*a, **{**k, "file": buf})}}
    try:
        exec(prediction, env)  # one namespace, so recursion and helpers resolve
    except Exception:
        return 0.0
    if "io_pairs" not in extras and "inputs" not in extras:
        return 1.0 if _clean(buf.getvalue()) in {_clean(t) for t in targets} else 0.0
    return _score_calls(env.get("f"), targets, extras, buf) if callable(env.get("f")) else 0.0


def _score_calls(fn, targets, extras, buf):
    if "io_pairs" in extras:  # return values, as the lesson's metric scores them
        hits = [_call(fn, x, buf)[0] == want for x, want in extras["io_pairs"]]
    else:  # stdout of f(inputs[i]) against targets[i]
        hits = [_call(fn, x, buf)[1] == _clean(want) for x, want in zip(extras["inputs"], targets)]
    return sum(hits) / len(hits) if hits else 0.0


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fizz = {"inputs": [3, 5, 7]}
    fixtures = ref.build_code_task()
    toy = ref.ToyAdapter().generate([ex.prompt for ex in fixtures])
    both = [(round(ref.metric_code_exec(p, [], e), 3), round(code_exec_stdout(p, [], e), 3))
            for p, e in ((FACT, {"io_pairs": [[3, 6], [4, 24]]}), (HELPER, {"io_pairs": [[3, 6]]}),
                         (PRINTS, {"io_pairs": [[3, 6]]}), (ESCAPE, {"io_pairs": [[0, os.getcwd()]]}))]
    return {
        "fizz": code_exec_stdout(FIZZ, ["Fizz", "Buzz", "7"], fizz),
        "fizz_bad": round(code_exec_stdout(FIZZ_BAD, ["Fizz", "Buzz", "7"], fizz), 3),
        "hello": code_exec_stdout(HELLO, ["hi", "hello"], {}),
        "fixtures": [(ref.metric_code_exec(p, ex.targets, ex.extras), code_exec_stdout(p, ex.targets, ex.extras))
                     for p, ex in zip(toy, fixtures)],
        "fact": both[0], "helper": both[1], "prints": both[2], "escape": both[3],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: stdout is captured and scored per call and per program; shipped fixtures unchanged",
            (r["fizz"], r["fizz_bad"], r["hello"]) == (1.0, 0.667, 1.0) and r["fixtures"] == [(1.0, 1.0)] * 5,
            f"fizzbuzz {r['fizz']}, wrong fizzbuzz {r['fizz_bad']}, hello vs [hi, hello] {r['hello']}; "
            f"shipped vs extended on the 5 fixtures {r['fixtures']}",
        ),
        practice.Check(
            "FINDING: the shipped namespace has no print, so a printing prediction scores 0",
            r["prints"] == (0.0, 1.0),
            f"(shipped, extended) on a printing doubler: {r['prints']}",
        ),
        practice.Check(
            "FINDING: split globals/locals make recursion and helper functions score 0",
            (r["fact"], r["helper"]) == ((0.0, 1.0), (0.0, 1.0)),
            f"(shipped, extended): recursive factorial {r['fact']}, f calling g {r['helper']}",
        ),
        practice.Check(
            "FINDING: stripped builtins are not a sandbox: a prediction reaches os.getcwd()",
            r["escape"] == (1.0, 1.0),
            f"(shipped, extended) on the __subclasses__ escape: {r['escape']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
