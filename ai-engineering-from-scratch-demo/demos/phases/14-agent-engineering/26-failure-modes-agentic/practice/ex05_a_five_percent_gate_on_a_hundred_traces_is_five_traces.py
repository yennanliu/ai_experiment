"""Exercise 5 — a five percent gate on a hundred traces is five traces.

    Wire one detector into a CI job: fail the build if >=5% of traces tag a
    mode.

Reading of the exercise: the gate is four lines and the question it raises is
how often it is right. At 100 traces a 5% threshold is five traces, which is
inside the sampling noise of a system whose true rate is near 5% -- so the
gate is built, and then run repeatedly against a fixed true rate to count how
often it flips for no reason.

**ANSWER: the gate works, and at a true rate of 5% it is a coin flip that
more data does not fix.** Over **200** seeded CI runs of **100** traces each
against a population whose true `tool_misuse` rate is exactly **0.05**, the
build fails **112** times and passes **88** -- **44.0%** of runs get the
minority verdict on identical code. Ten times the traces makes it **47.5%**,
*worse*, because a threshold set at the true rate converges to 50/50 as the
estimate sharpens. Moving the rate off the threshold is what fixes it: at
**0.02** the flap rate is **2.0%** and at **0.10** it is **4.5%**, against
**44.0%** at the threshold itself. A gate is only decisive where the
population is not.

**FINDING: gating "any mode" instead of one mode fails every build.** The
corpus tags **5** distinct modes, and the share of traces carrying at least
one tag is **72.0%** -- **14.4x** the threshold. A gate written against
`tag(trace)` rather than a named mode is not a quality bar, it is an
unconditional failure, which is the over-alerting pitfall arriving on day one.

**FINDING: the threshold is absolute, so it cannot see an improvement.** A
release that takes `scope_creep` from **29%** to **12%** still fails a 5%
gate, and one that takes it from **4%** to **4.9%** still passes -- so the
build is green while the number doubles toward the limit. The lesson's "no
baseline" pitfall is the reason: without a last-known-good, "getting worse"
is not expressible.

**FINDING: the detector the gate wraps decides what the gate can ever catch.**
Wiring `detect_context_loss` in gives **0** hits on a corpus containing
**14** writes to a path an explicit constraint forbade, so that build can
never fail. A CI gate inherits its detector's recall, and a gate on a detector that
never fires is a green light with a policy attached.

Structure: `ci_gate()` is the four-line job; `flap_rate()` reruns it against
a fixed true rate.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "26-failure-modes-agentic"
THRESHOLD, RUNS = 0.05, 200
TRUE_RATE = 0.05


def ci_gate(traces, tagger, mode, threshold=THRESHOLD):
    """Fail the build if at least `threshold` of traces carry `mode`."""
    hits = sum(mode in tagger(trace) for trace in traces)
    rate = hits / len(traces)
    return {"rate": rate, "failed": rate >= threshold, "hits": hits}


def bernoulli_traces(ref, rng, size, rate):
    """`size` traces, each independently carrying tool_misuse with `rate`."""
    traces = []
    for index in range(size):
        bad = rng.random() < rate
        args = {"file": "/tmp/f"} if bad else {"path": "/tmp/f"}
        traces.append(ref.Trace(tid=f"t{index}", user_request="read a file",
                                constraints=[],
                                steps=[ref.TraceStep("tool_call", "read_file", args)],
                                final_success_claim=False,
                                target_state_changed=False))
    return traces


def flap_rate(ref, size, rate=TRUE_RATE, runs=RUNS, seed=5):
    """How often the same code gets the opposite verdict from the same gate."""
    rng = random.Random(seed)
    failures = sum(ci_gate(bernoulli_traces(ref, rng, size, rate),
                           ref.tag, "tool_misuse")["failed"] for _ in range(runs))
    minority = min(failures, runs - failures)
    return {"failures": failures, "passes": runs - failures,
            "flap": round(100 * minority / runs, 1)}


def mixed_corpus(ref, size=100):
    """The demo's own seven shapes, cycled, so the rates are the demo's rates."""
    ok = {"path": "p", "content": "c", "query": "q"}
    rows = [("find the config file", ["do not modify any files"],
             [("search", ok, "ok"), ("write_file", ok, "ok")], True, True),
            ("list project files", [], [("magic_scanner", ok, "ok")], True, False),
            ("look up invoice 4711", [],
             [("search", ok, "error"), ("read_file", ok, "ok"),
              ("write_file", ok, "ok"), ("list_dir", ok, "ok")], True, True),
            ("update readme", ["do not modify src/"],
             [("read_file", ok, "ok"), ("write_file", ok, "ok"),
              ("write_file", {"path": "src/f.py", "content": "x"}, "ok")], True, True),
            ("read some file", [], [("read_file", {"file": "/tmp/f"}, "ok")],
             False, False),
            ("create a PR", [], [("search", ok, "ok")], True, False),
            ("find the config file", [], [("search", ok, "ok")], True, False)]
    traces = []
    for index in range(size):
        request, constraints, steps, claim, changed = rows[index % len(rows)]
        traces.append(ref.Trace(
            tid=f"m{index}", user_request=request, constraints=constraints,
            steps=[ref.TraceStep("tool_call", n, a, status=s) for n, a, s in steps],
            final_success_claim=claim, target_state_changed=changed))
    return traces


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    small, large = flap_rate(ref, 100), flap_rate(ref, 1000)
    away = {rate: flap_rate(ref, 100, rate=rate)["flap"] for rate in (0.02, 0.10)}
    mixed = mixed_corpus(ref)
    tagged = [ref.tag(trace) for trace in mixed]
    any_rate = round(100 * sum(bool(row) for row in tagged) / len(mixed), 1)
    creep = ci_gate(mixed, ref.tag, "scope_creep")
    return {
        "runs": RUNS, "small": small, "large": large, "threshold": THRESHOLD,
        "away": away,
        "modes": len({label for row in tagged for label in row}),
        "any_rate": any_rate, "ratio": round(any_rate / (100 * THRESHOLD), 1),
        "creep_rate": round(100 * creep["rate"], 1), "creep_failed": creep["failed"],
        "improved_fails": 0.12 >= THRESHOLD,
        "doubled_passes": 0.049 < THRESHOLD,
        "context_gate": ci_gate(mixed, ref.tag, "context_loss"),
        "violations": sum("src/f.py" in str(step.args) for trace in mixed
                          for step in trace.steps),
    }


def verify(result):
    small, large, gate = result["small"], result["large"], result["context_gate"]
    return [
        practice.Check(
            "ANSWER: at a true rate of 5% the gate is a coin flip more data cannot fix",
            all([small["failures"] == 112, small["passes"] == 88,
                 small["flap"] == 44.0, large["flap"] == 47.5,
                 result["away"] == {0.02: 2.0, 0.10: 4.5},
                 result["runs"] == 200]),
            f"over {result['runs']} seeded runs of 100 traces against a true rate of "
            f"{TRUE_RATE}, the build fails {small['failures']} times and passes "
            f"{small['passes']}: {small['flap']}% of runs get the minority verdict on "
            f"identical code. Ten times the traces makes it {large['flap']}%, and moving "
            f"the true rate off the threshold gives {result['away']}",
        ),
        practice.Check(
            "FINDING: gating any mode instead of one mode fails every build",
            all([result["modes"] == 5, result["any_rate"] == 72.0,
                 result["ratio"] == 14.4]),
            f"the corpus tags {result['modes']} distinct modes and {result['any_rate']}% "
            f"of traces carry at least one -- {result['ratio']}x the threshold. A gate "
            "written against tag(trace) rather than a named mode is an unconditional "
            "failure",
        ),
        practice.Check(
            "FINDING: the threshold is absolute, so it cannot see an improvement",
            all([result["creep_rate"] == 29.0, result["creep_failed"] is True,
                 result["improved_fails"] is True,
                 result["doubled_passes"] is True]),
            f"scope_creep sits at {result['creep_rate']}% and fails the gate "
            f"({result['creep_failed']}); a release that halves it to 12% still fails, "
            "and one that doubles 4% to 4.9% still passes. Without a last-known-good, "
            "'getting worse' is not expressible",
        ),
        practice.Check(
            "FINDING: the gate inherits its detector's recall",
            all([gate["hits"] == 0, gate["failed"] is False,
                 result["violations"] == 14]),
            f"wiring detect_context_loss in gives {gate['hits']} hits on a corpus "
            f"containing {result['violations']} writes to a forbidden path, so that build "
            "can never fail. A gate on a detector that never fires is a green light with "
            "a policy attached",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
