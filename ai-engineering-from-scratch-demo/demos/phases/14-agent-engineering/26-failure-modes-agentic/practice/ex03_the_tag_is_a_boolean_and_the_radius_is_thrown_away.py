"""Exercise 3 — the tag is a boolean and the radius is thrown away.

    Implement a "cascade radius" metric: given a failure at step N, how many
    downstream steps did it affect?

Reading of the exercise: `detect_cascading_errors` already walks the trace
counting steps after the first error, compares the count against **2**, and
then returns a string. The radius is computed and discarded, so implementing
the metric means keeping the number -- and then noticing that "downstream"
and "affected" are two different counts, because a step that never reads the
failed step's output was not affected by it.

**ANSWER: positional radius averages 3.1 and data-flow radius 1.6 over 10
cascades.** Counting every tool call after the first error gives radii
`[1, 2, 3, 4, 5, 1, 3, 4, 5, 3]`; counting only the calls whose arguments
carry a value derived from the failed step gives `[1, 1, 1, 2, 3, 1, 1, 2, 3,
1]`. Positional radius overstates blast by **1.9x** on the same traces --
"downstream" and "affected" are different questions.

**FINDING: the shipped threshold discards cascades of radius one.**
`downstream_ops >= 2` means a failure with exactly **1** step after it is not
a cascade, so **2** of **10** traces are tagged clean -- and both of them
really did carry the failed value into the next call. The lesson calls
cascading "the killer" and the detector's floor is two.

**FINDING: only the first error starts the count.** The loop sets
`saw_error = True` and never resets, so a trace with **3** separate errors
reports one cascade whose radius spans from the first to the end -- **9**
steps -- where per-error radii are **4**, **2** and **1**. One number for
three incidents is the over-alerting pitfall solved by under-counting.

**FINDING: the tag carries no position, so the radius cannot be recovered
downstream.** `tag()` returns `list[str]`; the module's **6** detectors
return **6** strings and **0** step indices. Two traces with radii **2** and
**5** produce byte-identical output, so a dashboard built on `tag` can rank
modes by frequency and never by blast.

Structure: `radii()` returns both counts per error; `CASCADES` are the traces
they are measured on.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "26-failure-modes-agentic"
POISON = "fabricated"
# (steps after the first error, how many of them read the poisoned value)
SHAPES = ((1, 1), (2, 1), (3, 1), (4, 2), (5, 3),
          (1, 1), (3, 1), (4, 2), (5, 3), (3, 1))


def build(ref, index, after, tainted):
    steps = [ref.TraceStep("tool_call", "search", {"query": "invoice"},
                           status="error", result=POISON)]
    for position in range(after):
        args = {"path": "/tmp/f", "content": POISON} if position < tainted \
            else {"path": "/tmp/g", "content": "clean"}
        steps.append(ref.TraceStep("tool_call", "write_file", args))
    return ref.Trace(tid=f"c{index:03d}", user_request="look up invoice 4711",
                     constraints=[], steps=steps, final_success_claim=True,
                     target_state_changed=True)


def radii(trace):
    """Positional radius, and the radius that survives a data-flow check."""
    positional = flow = 0
    seen_error = False
    for step in trace.steps:
        if step.kind != "tool_call":
            continue
        if step.status == "error" and not seen_error:
            seen_error = True
            continue
        if seen_error:
            positional += 1
            flow += POISON in str(step.args)
    return positional, flow


def per_error_radii(trace):
    """A radius for every error, not one for the first."""
    calls = [s for s in trace.steps if s.kind == "tool_call"]
    errors = [i for i, s in enumerate(calls) if s.status == "error"]
    spans = []
    for order, start in enumerate(errors):
        end = errors[order + 1] if order + 1 < len(errors) else len(calls)
        spans.append(end - start - 1)
    return spans


def multi_error(ref):
    """Three separate failures in one trace."""
    kinds = ["error", "ok", "ok", "ok", "ok", "error", "ok", "ok", "error", "ok"]
    steps = [ref.TraceStep("tool_call", "search", {"query": "x"},
                           status=kind, result=POISON if kind == "error" else "")
             for kind in kinds]
    return ref.Trace(tid="multi", user_request="look up invoice", constraints=[],
                     steps=steps, final_success_claim=True,
                     target_state_changed=True)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    traces = [build(ref, i, after, tainted)
              for i, (after, tainted) in enumerate(SHAPES)]
    measured = [radii(trace) for trace in traces]
    positional = [p for p, _ in measured]
    flow = [f for _, f in measured]
    tagged = [ref.detect_cascading_errors(trace) is not None for trace in traces]
    many = multi_error(ref)
    return {
        "cascades": len(SHAPES), "positional": positional, "flow": flow,
        "mean_positional": round(sum(positional) / len(positional), 1),
        "mean_flow": round(sum(flow) / len(flow), 1),
        "overstatement": round(sum(positional) / sum(flow), 1),
        "tagged": sum(tagged),
        "dropped": sum(not hit and f >= 1 for hit, f in zip(tagged, flow)),
        "many_positional": radii(many)[0], "many_errors": 3,
        "many_per_error": per_error_radii(many),
        "tag_type": type(ref.tag(traces[0])).__name__,
        "detectors": len(ref.DETECTORS),
        "identical": ref.tag(traces[1]) == ref.tag(traces[4]),
        "extremes": (positional[1], positional[4]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: positional radius averages 3.1 and data-flow radius 1.6",
            all([result["cascades"] == 10, result["mean_positional"] == 3.1,
                 result["mean_flow"] == 1.6, result["overstatement"] == 1.9,
                 result["positional"] == [1, 2, 3, 4, 5, 1, 3, 4, 5, 3],
                 result["flow"] == [1, 1, 1, 2, 3, 1, 1, 2, 3, 1]]),
            f"counting every call after the first error gives {result['positional']}; "
            f"counting only calls whose arguments carry the failed step's value gives "
            f"{result['flow']}. Positional radius overstates blast by "
            f"{result['overstatement']}x on the same traces",
        ),
        practice.Check(
            "FINDING: the shipped threshold discards cascades of radius one",
            all([result["tagged"] == 8, result["dropped"] == 2,
                 result["positional"][0] == 1]),
            f"downstream_ops >= 2 means a failure with one step after it is not a "
            f"cascade, so {result['cascades'] - result['tagged']} of "
            f"{result['cascades']} traces are tagged clean -- and {result['dropped']} of "
            "those really did propagate the failed value. The lesson calls cascading the "
            "killer and the detector's floor is two",
        ),
        practice.Check(
            "FINDING: only the first error starts the count",
            all([result["many_positional"] == 9, result["many_errors"] == 3,
                 result["many_per_error"] == [4, 2, 1]]),
            f"the loop sets saw_error and never resets, so a trace with "
            f"{result['many_errors']} separate errors reports one cascade of "
            f"{result['many_positional']} steps where the per-error radii are "
            f"{result['many_per_error']}. One number for three incidents",
        ),
        practice.Check(
            "FINDING: the tag carries no position",
            all([result["tag_type"] == "list", result["identical"] is True,
                 result["extremes"] == (2, 5), result["detectors"] == 6]),
            f"tag() returns a {result['tag_type']} of strings and the module's "
            f"{result['detectors']} detectors return zero step indices, so traces with "
            f"radii {result['extremes'][0]} and {result['extremes'][1]} produce identical "
            f"output ({result['identical']}). Modes can be ranked by frequency, never by "
            "blast",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
