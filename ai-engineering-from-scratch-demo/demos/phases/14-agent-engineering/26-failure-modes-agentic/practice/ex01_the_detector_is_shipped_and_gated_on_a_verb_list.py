"""Exercise 1 — the detector is shipped, and gated on a verb list.

    Add a detector for "success hallucination": agent returns success but the
    target state is unchanged.

Reading of the exercise: `detect_success_hallucination` is already in
`DETECTORS`, so the work is finding out what it misses. It fires only when
the *request text* contains one of six verbs, which makes its recall a
property of how the user phrased the ask -- and the lesson's own description
of the mode ("hallucinate a success message on 400 errors") says nothing
about phrasing.

**ANSWER: gate on the tool calls the agent made, not on the words the user
used.** Over **12** traces that claim success with the target state
unchanged, the shipped verb gate catches **5**. Probing state with no gate at
all catches **12** and raises **4** false positives on read-only work.
Flagging when the trace contains a *mutating* call catches **12** with **0**
false positives, because what the agent did is observable and what the user
meant is not.

**FINDING: recall is a property of the user's vocabulary.** The six-verb list
misses `delete`, `send`, `deploy`, `rename`, `archive`, `publish` and
`revoke` -- **7** requests that plainly change state, flagged **0** times.
Adding a verb fixes one phrasing and not the mode.

**FINDING: nothing probes the state.** `target_state_changed` is one of
`Trace`'s **6** fields, supplied by whoever built the trace, and **0** of the
module's **10** functions computes it. The lesson's mitigation is "re-probe
state -- was the file actually created?"; the code has a boolean where the
probe should be.

**FINDING: no detector reads `status` on the final step.** A trace whose last
tool call errored, that claims success and changed nothing, is tagged by
**0** of the **6** shipped detectors when the request has no write verb and
only one call follows the error. That is exactly the 400-error case the
lesson calls out, and it falls through every gate.

Structure: `variants()` holds the three detectors; `CASES` are the traces
they disagree on.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "26-failure-modes-agentic"
SHIPPED_VERBS = ("write", "create", "save", "update", "edit", "make")
MISSED_VERBS = ("delete", "send", "deploy", "rename", "archive", "publish",
                "revoke")
MUTATING = {"write_file", "delete_file", "send_email", "deploy", "rename_file"}
# (request, tool calls, claims success, state changed, should be flagged)
CASES = tuple(
    (f"{verb} the release artifact", ["write_file"], True, False, True)
    for verb in SHIPPED_VERBS[:5] + MISSED_VERBS
) + (
    ("find the config file", ["search", "read_file"], True, False, False),
    ("list project files", ["list_dir"], True, False, False),
    ("read the changelog", ["read_file"], True, False, False),
    ("search for invoice 4711", ["search"], True, False, False),
)


def make_trace(ref, index, case):
    request, tools, claim, changed, _ = case
    steps = [ref.TraceStep("tool_call", name, {"path": "x", "content": "y",
                                               "query": "z"})
             for name in tools]
    return ref.Trace(tid=f"t{index:03d}", user_request=request, constraints=[],
                     steps=steps, final_success_claim=claim,
                     target_state_changed=changed)


def probe_only(trace):
    """No intent gate at all: claimed success, state unchanged."""
    if trace.final_success_claim and not trace.target_state_changed:
        return "success_hallucination"
    return None


def mutation_gated(trace):
    """Gate on what the agent did, which is observable."""
    mutated = any(step.name in MUTATING for step in trace.steps
                  if step.kind == "tool_call")
    if mutated and trace.final_success_claim and not trace.target_state_changed:
        return "success_hallucination"
    return None


def score(ref, detector):
    hits = misses = false_pos = 0
    for index, case in enumerate(CASES):
        flagged = detector(make_trace(ref, index, case)) is not None
        hits += flagged and case[4]
        misses += case[4] and not flagged
        false_pos += flagged and not case[4]
    return {"hits": hits, "misses": misses, "false_pos": false_pos}


def four_hundred(ref):
    """The lesson's own example: a 400, a claim of success, nothing changed."""
    steps = [ref.TraceStep("tool_call", "search", {"query": "invoice 4711"},
                           status="error"),
             ref.TraceStep("tool_call", "read_file", {"path": "/tmp/foo"})]
    trace = ref.Trace(tid="t400", user_request="look up invoice 4711",
                      constraints=[], steps=steps, final_success_claim=True,
                      target_state_changed=False)
    return ref.tag(trace)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    functions = [name for name, value in vars(ref).items()
                 if callable(value) and getattr(value, "__module__", "") == ref.__name__]
    return {
        "cases": len(CASES),
        "positives": sum(case[4] for case in CASES),
        "shipped": score(ref, ref.detect_success_hallucination),
        "probe": score(ref, probe_only),
        "mutation": score(ref, mutation_gated),
        "missed_verbs": len(MISSED_VERBS),
        "trace_fields": list(ref.Trace.__dataclass_fields__),
        "functions": len(functions), "detectors": len(ref.DETECTORS),
        "probers": [n for n in functions if "probe" in n or "verify" in n],
        "four_hundred": four_hundred(ref),
    }


def verify(result):
    shipped, probe, mutation = result["shipped"], result["probe"], result["mutation"]
    return [
        practice.Check(
            "ANSWER: gate on the calls the agent made, not the words the user used",
            all([result["positives"] == 12, shipped["hits"] == 5,
                 probe["hits"] == 12, probe["false_pos"] == 4,
                 mutation["hits"] == 12, mutation["false_pos"] == 0]),
            f"over {result['positives']} traces claiming success with state unchanged, "
            f"the shipped verb gate catches {shipped['hits']}; probing state with no gate "
            f"catches {probe['hits']} at {probe['false_pos']} false positives; gating on "
            f"a mutating call catches {mutation['hits']} at {mutation['false_pos']}",
        ),
        practice.Check(
            "FINDING: recall is a property of the user's vocabulary",
            all([shipped["misses"] == 7, result["missed_verbs"] == 7,
                 shipped["false_pos"] == 0]),
            f"the six-verb list misses {list(MISSED_VERBS)} -- "
            f"{shipped['misses']} requests that plainly change state, flagged zero times. "
            "Adding a verb fixes one phrasing and not the mode",
        ),
        practice.Check(
            "FINDING: nothing probes the state",
            all([len(result["trace_fields"]) == 6, result["probers"] == [],
                 "target_state_changed" in result["trace_fields"],
                 result["functions"] == 10]),
            f"target_state_changed is one of Trace's {len(result['trace_fields'])} fields, "
            f"supplied by whoever built the trace, and {len(result['probers'])} of the "
            f"module's {result['functions']} functions computes it. The mitigation is "
            "'re-probe state'; the code has a boolean where the probe should be",
        ),
        practice.Check(
            "FINDING: no detector reads status on the final step",
            all([result["four_hundred"] == [], result["detectors"] == 6]),
            f"a trace whose first call errored, that claims success and changed nothing, "
            f"is tagged {result['four_hundred']} by all {result['detectors']} detectors: "
            "the request has no write verb and only one call follows the error, so the "
            "400-error case the lesson calls out falls through every gate",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
