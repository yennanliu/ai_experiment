"""Exercise 3 — a five percent threshold needs twenty cases to exist.

    Wire the eval suite into CI. Fail the build on >=5% regression.

Reading of the exercise: `ci_gate` ships and takes the threshold as an
argument, so wiring is one call. The exercise says "fail on >=5%" and the
code says `regression > regression_threshold`, which is a different rule --
and on the shipped four-case suite neither rule can express 5% at all,
because a pass rate over 4 cases moves in steps of 25.

**ANSWER: the gate wires in one line and cannot enforce the stated rule on
the shipped suite.** With the lesson's **4** cases the pass rate takes **5**
possible values and the smallest non-zero regression is **25.0%**, so every
threshold between **0%** and **25%** behaves identically. Growing the suite
to **20** cases makes **5.0%** reachable -- one case -- and the gate starts
distinguishing the two rules.

**FINDING: `>` is not `>=`, and a regression exactly at the threshold is
allowed.** Shown at **25.0%**, where the arithmetic is binary-exact: one
failure in four cases against a baseline of 1.0 gives `allow=True`, because
the comparison is strict. The exercise's rule blocks it. At 5% the same
comparison is additionally at the mercy of float representation --
`0.95 - 18/20` is **0.049999999999999934**, which is a second, quieter way
for a boundary regression to pass.

**FINDING: the baseline is an argument, and nothing stores one.** `main`
passes the literal `0.95`, `ci_gate` has **3** parameters and **0** of them
is a path, and the module defines **0** functions that read or write a
baseline. Running the suite twice cannot tell you whether anything changed,
which is the lesson's own "no baseline" pitfall present in the reference.

**FINDING: every case weighs the same, so a guardrail failure hides.** The
pass rate is an unweighted mean over categories, so on a suite of **39**
benchmark cases and **1** online guardrail, the guardrail failing alone is a
**2.5%** regression and the build is allowed. Weighting online cases at
**3x** makes the same failure **7.1%** and blocks it. The three layers the
lesson separates are flattened into one denominator.

Structure: `suite()` builds a suite of a given size; `sweep()` reports what
each threshold can and cannot see.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "30-eval-driven-agent-development"
THRESHOLD, BASELINE = 0.05, 0.95


def case(ref, cid, category, passes):
    """A case whose verdict is fixed, so the gate is what is under test."""
    return ref.EvalCase(
        cid=cid, category=category, description=cid,
        proposer=lambda feedback: "candidate",
        judge=lambda candidate: (passes, "fixed verdict"))


def suite(ref, size, failures=0, category="benchmark"):
    return [ref.evaluator_optimizer(
        case(ref, f"c{i:03d}", category, i >= failures)) for i in range(size)]


def rates(ref, size):
    """Every pass rate a suite of this size can produce."""
    return sorted({round(1 - failures / size, 4)
                   for failures in range(size + 1)})


def gate(ref, results, baseline=BASELINE, threshold=THRESHOLD):
    allow, message = ref.ci_gate(results, baseline_pass_rate=baseline,
                                 regression_threshold=threshold)
    return {"allow": allow, "message": message}


def weighted_rate(results, weights):
    total = sum(weights[row.category] for row in results)
    passed = sum(weights[row.category] for row in results if row.passed)
    return round(passed / total, 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    small, large = rates(ref, 4), rates(ref, 20)
    # 0.25 is binary-exact, so this isolates the operator from float rounding.
    boundary = suite(ref, 4, failures=1)
    mixed = suite(ref, 39) + suite(ref, 1, failures=1, category="online")
    weights = {"benchmark": 1, "custom": 1, "online": 3}
    return {
        "small_values": len(small), "small_step": round(100 * (1 - small[-2]), 1),
        "large_step": round(100 * (1 - large[-2]), 1),
        "gate_params": ref.ci_gate.__code__.co_argcount,
        "boundary_regression": round(100 * (1.0 - 3 / 4), 1),
        "boundary_allow": gate(ref, boundary, baseline=1.0,
                               threshold=0.25)["allow"],
        "strict_would_block": (1.0 - 3 / 4) >= 0.25,
        "uses_gte": ">=" in inspect.getsource(ref.ci_gate),
        "baseline_params": [n for n in ref.ci_gate.__code__.co_varnames[:3]
                            if "path" in n or "store" in n],
        "baseline_functions": [n for n, v in vars(ref).items()
                               if callable(v) and "baseline" in n],
        "mixed_cases": len(mixed),
        "mixed_regression": round(100 * (1.0 - 39 / 40), 1),
        "mixed_allow": gate(ref, mixed, baseline=1.0)["allow"],
        "weighted": round(100 * (1.0 - weighted_rate(mixed, weights)), 1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the gate cannot express 5% on a four-case suite",
            all([result["small_values"] == 5, result["small_step"] == 25.0,
                 result["large_step"] == 5.0, result["gate_params"] == 3]),
            f"over the lesson's 4 cases the pass rate takes {result['small_values']} "
            f"values and the smallest non-zero regression is {result['small_step']}%, so "
            f"every threshold between 0 and {result['small_step']}% behaves identically. "
            f"At 20 cases the step is {result['large_step']}% and the threshold becomes "
            "meaningful",
        ),
        practice.Check(
            "FINDING: > is not >=, and a regression of exactly 5% is allowed",
            all([result["boundary_regression"] == 25.0,
                 result["boundary_allow"] is True,
                 result["strict_would_block"] is True,
                 result["uses_gte"] is False]),
            f"at a binary-exact {result['boundary_regression']}% threshold -- one failure "
            f"in four cases -- ci_gate returns allow={result['boundary_allow']} because "
            f"the comparison is strict. The exercise's stated rule blocks it "
            f"({result['strict_would_block']}), and at 5% the same comparison is subject "
            "to float representation as well",
        ),
        practice.Check(
            "FINDING: the baseline is an argument and nothing stores one",
            all([result["baseline_params"] == [],
                 result["baseline_functions"] == [], result["gate_params"] == 3]),
            f"ci_gate has {result['gate_params']} parameters and "
            f"{len(result['baseline_params'])} of them names a path or a store, and the "
            f"module defines {len(result['baseline_functions'])} functions that read or "
            "write a baseline. Running the suite twice cannot say whether anything moved",
        ),
        practice.Check(
            "FINDING: every case weighs the same, so a guardrail failure hides",
            all([result["mixed_cases"] == 40, result["mixed_regression"] == 2.5,
                 result["mixed_allow"] is True, result["weighted"] == 7.1]),
            f"on 39 benchmark cases plus 1 online guardrail, the guardrail failing alone "
            f"is a {result['mixed_regression']}% regression -- allowed "
            f"({result['mixed_allow']}). Weighting online cases at 3x makes the same "
            f"failure {result['weighted']}% and blocks it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
