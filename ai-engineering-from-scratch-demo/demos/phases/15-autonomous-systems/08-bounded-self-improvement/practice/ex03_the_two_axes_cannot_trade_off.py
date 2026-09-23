"""Exercise 3 — the two axes cannot trade off.

    Disable the multi-objective constraint. Show the loop converges on the
    performance axis while a safety axis drops.

Reading of the exercise: the demonstration it asks for does not exist in this
simulator, and establishing that is a stronger answer than producing a seed
where it looks like it does. So the claim is tested exhaustively rather than
sampled: every reachable agent is scored on both axes, and the two axes are
checked for a region where they disagree.

**ANSWER: safety does not drop, and cannot.** With the multi gate off the
loop accepts **142** of 200 cycles and finishes at perf **1.00** and safety
**1.00**. Over all **2801** operator sequences up to length **4**, the
highest perf any sequence with safety below 1.0 reaches is **0.0** -- there
is no agent anywhere in the reachable space that is good at the task and bad
on safety.

**FINDING: one tool carries the whole safety axis.** Of the **7** operators,
`reverse` is the only one that breaks the filter, and it does so by
destroying the string the filter searches. It also takes perf to **0.0**, so
the axis that is supposed to be independent is a strict sub-case of the axis
it is supposed to constrain.

**FINDING: the filter is applied after the operators, not before.**
`apply` runs the whole sequence and only then scans for the four danger
substrings, so "safety" here measures whether the transform left the pattern
legible. `upper` and `title` both survive it -- the scan lowercases first --
and `reverse` defeats it. A filter that can be defeated by reordering
characters is testing the operators, not the input.

**FINDING: the gate that is disabled was rejecting the other axis.** With
every gate on, its **66** rejections split **46** perf-only, **20** both and
**0** safety-only. Turning it off therefore removes a perf floor of **0.25**,
which is why acceptances rise from **96** to **142** -- the observable effect
of disabling the *safety* constraint is entirely on the performance axis.

Structure: `space()` scores every reachable agent on both axes; `replay()`
is `run`'s loop with the gate configuration as an argument.
"""

from __future__ import annotations

import inspect
import itertools
import random

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "08-bounded-self-improvement"

SEED, CYCLES, MAX_OPS = 5, 200, 4
ALL_ON = dict(invariant=True, anchor=True, multi=True, regress=True)


def space(ref, max_ops=MAX_OPS):
    """(perf, safety) for every operator sequence up to `max_ops`."""
    names = sorted(ref.APPROVED_TOOL_MANIFEST) + ["nop"]
    return [(ref.perf_score(ref.Agent(ops=list(seq))),
             ref.safety_score(ref.Agent(ops=list(seq))))
            for length in range(max_ops + 1)
            for seq in itertools.product(names, repeat=length)]


def single_tools(ref):
    names = sorted(ref.APPROVED_TOOL_MANIFEST) + ["nop"]
    return {name: (ref.perf_score(ref.Agent(ops=[name])),
                   ref.safety_score(ref.Agent(ops=[name])))
            for name in names}


GATES = ("invariant", "anchor", "multi", "regress")


def failing_gate(ref, gates, candidate, history):
    """The first gate a candidate fails, in the order `run` applies them."""
    perf, safety = ref.perf_score(candidate), ref.safety_score(candidate)
    checks = (("invariant", ref.gate_invariant(candidate)),
              ("anchor", ref.gate_anchor(candidate)),
              ("multi", ref.gate_multi(perf, safety)),
              ("regress", ref.gate_regression(history, perf)))
    for name, passed in checks:
        if gates[name] and not passed:
            return name
    return None


def replay(ref, gates, edit=True):
    """`run`'s loop, returning the state it prints and the state it does not."""
    random.seed(SEED)
    agent, accepted = ref.Agent(), []
    history, rejects = [ref.perf_score(agent)], dict.fromkeys(GATES, 0)
    scores = []
    for _cycle in range(CYCLES):
        candidate = ref.mutate(agent, edit)
        gate = failing_gate(ref, gates, candidate, history)
        scores.append((gate, ref.perf_score(candidate), ref.safety_score(candidate)))
        if gate:
            rejects[gate] += 1
            continue
        agent = candidate
        accepted.append(ref.perf_score(agent))
        history.append(accepted[-1])
    return {"accepted": len(accepted), "rejects": rejects, "agent": agent,
            "history": history, "scores": scores, "perfs": accepted}


def multi_reasons(scores):
    """Why each multi rejection fired, by axis."""
    counts = dict.fromkeys(("perf", "safety", "both"), 0)
    for gate, perf, safety in scores:
        if gate == "multi":
            counts["both" if perf < 0.25 and safety < 1.0
                   else ("perf" if perf < 0.25 else "safety")] += 1
    return counts


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = space(ref)
    tools = single_tools(ref)
    guarded, unguarded = replay(ref, ALL_ON), replay(ref, dict(ALL_ON, multi=False))
    off_agent, body = unguarded["agent"], inspect.getsource(ref.apply)
    return {
        "unguarded": [unguarded["accepted"], ref.perf_score(off_agent),
                      ref.safety_score(off_agent)],
        "guarded": [guarded["accepted"]],
        "searched": len(rows),
        "max_ops": MAX_OPS,
        "unsafe_best_perf": max(perf for perf, safety in rows if safety < 1.0),
        "tradeoff_region": sum(perf >= 0.25 and safety < 1.0 for perf, safety in rows),
        "tools": len(tools),
        "unsafe_tools": [name for name, (_p, safety) in tools.items() if safety < 1.0],
        "unsafe_tool_perf": [tools[name][0] for name, (_p, s) in tools.items() if s < 1.0],
        "filter_after_ops": body.index("dangerous =") > body.index("for op in agent.ops"),
        "filter_lowercases": "s.lower()" in body,
        "reasons": multi_reasons(guarded["scores"]),
        "perf_floor": 0.25,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: safety does not drop, and cannot",
            all([result["unguarded"] == [142, 1.0, 1.0],
                 result["unsafe_best_perf"] == 0.0,
                 result["tradeoff_region"] == 0, result["searched"] == 2801]),
            f"with the multi gate off the loop accepts {result['unguarded'][0]} cycles "
            f"and ends at perf {result['unguarded'][1]} and safety "
            f"{result['unguarded'][2]}; across {result['searched']} sequences "
            f"{result['tradeoff_region']} have perf at or above 0.25 with safety below "
            "1.0",
        ),
        practice.Check(
            "FINDING: one tool carries the whole safety axis",
            all([result["tools"] == 7, result["unsafe_tools"] == ["reverse"],
                 result["unsafe_tool_perf"] == [0.0]]),
            f"{len(result['unsafe_tools'])} of {result['tools']} operators breaks the "
            f"filter -- {result['unsafe_tools'][0]} -- and it scores "
            f"{result['unsafe_tool_perf'][0]} on perf, so the safety axis is a "
            "sub-case of the one it should constrain",
        ),
        practice.Check(
            "FINDING: the filter is applied after the operators, not before",
            all([result["filter_after_ops"], result["filter_lowercases"]]),
            "apply runs the whole sequence and only then scans for the danger "
            "substrings, lowercasing first -- so upper and title survive it and reverse "
            "defeats it by reordering characters",
        ),
        practice.Check(
            "FINDING: the gate that is disabled was rejecting the other axis",
            all([result["reasons"] == {"perf": 46, "safety": 0, "both": 20},
                 result["guarded"][0] == 96, result["unguarded"][0] == 142]),
            f"its rejections split {result['reasons']}, so turning it off removes a "
            f"perf floor of {result['perf_floor']} and acceptances rise "
            f"{result['guarded'][0]} to {result['unguarded'][0]} -- the observable "
            "effect of disabling the safety constraint is on the performance axis",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
