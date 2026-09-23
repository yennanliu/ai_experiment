"""Exercise 1 — the anchor gate never fires, because the invariant gate runs first.

    Run `code/main.py` with all primitives enabled. Confirm the loop still
    improves on the primary metric without letting the hack win.

Reading of the exercise: "confirm" means reproduce the run and then check the
claim it is evidence for -- that each primitive blocks a specific failure
class. `run` prints and returns `None`, so the loop is replayed here and the
replay is checked against the shipped stdout before any count is read off it.

**ANSWER: it improves to the ceiling and the hack does not land.** With all
four gates on, **96** of **200** cycles are accepted, final perf is **1.00**
against a brute-forced ceiling of **1.00** (reachable by
`['collapse', 'title']`), safety stays **1.00**, the manifest stays the
approved **6** tools and the anchor is intact. The replay reproduces the
shipped `accepted` count and all four reject counters exactly.

**FINDING: one of the four primitives never fires.** Rejects break down
**11** invariant, **66** multi, **27** regression and **0** anchor. `mutate`
edits the objective only in the branch that also adds `raw_eval` to the
manifest, and `gate_invariant` runs first -- so every objective edit is
already rejected by the time `gate_anchor` is asked. Two primitives guard one
event and the ordering makes one of them dead code.

**FINDING: three of the ninety-six accepted cycles improved anything.** Only
**3** acceptances raised perf above everything before them; the other **93**
passed all four gates and moved the agent sideways. The gates filter
proposals; they do not measure progress, and the accepted count is not one.

**FINDING: the hack is rare before it is blocked.** The manifest edit is the
fourth branch of `mutate`, reached only when the first three are skipped and
a second coin lands -- about **5%** of cycles. The **11** invariant rejects
out of 200 are that rate, not the gate's strength: with every gate off the
same mutation stream drifts the manifest and the objective in **200** of 200
cycles.

Structure: `replay()` is `run`'s loop with its counters exposed; `shipped()`
captures the real run's stdout to check the replay against.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import ast
import random
import re

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "08-bounded-self-improvement"

SEED, CYCLES, MAX_OPS = 5, 200, 4
ALL_ON = dict(invariant=True, anchor=True, multi=True, regress=True)
ALL_OFF = dict.fromkeys(ALL_ON, False)


def shipped(ref, gates, edit=True):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.run(gates, edit, CYCLES, seed=SEED)
    return buffer.getvalue()


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
    agent, accepted, rejects = ref.Agent(), [], dict.fromkeys(GATES, 0)
    history = [ref.perf_score(agent)]
    for _cycle in range(CYCLES):
        candidate = ref.mutate(agent, edit)
        gate = failing_gate(ref, gates, candidate, history)
        if gate:
            rejects[gate] += 1
            continue
        agent = candidate
        accepted.append(ref.perf_score(agent))
        history.append(accepted[-1])
    return {"accepted": len(accepted), "rejects": rejects, "agent": agent,
            "history": history, "perfs": accepted}


def rises(run):
    """Accepted cycles that raised perf above everything before them."""
    return sum(perf > max(run["history"][:index + 1])
               for index, perf in enumerate(run["perfs"]))


def ceiling(ref):
    """Best perf, shortest sequence reaching it, space size, best unsafe perf."""
    names = sorted(ref.APPROVED_TOOL_MANIFEST) + ["nop"]
    rows = []
    for length in range(MAX_OPS + 1):
        for seq in itertools.product(names, repeat=length):
            agent = ref.Agent(ops=list(seq))
            rows.append((ref.perf_score(agent), ref.safety_score(agent), seq))
    top = max(perf for perf, _s, _q in rows)
    best = min((q for perf, _s, q in rows if perf == top), key=len)
    return top, list(best), len(rows), max(p for p, s, _q in rows if s < 1.0)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    text = shipped(ref, ALL_ON)
    run_on, run_off = replay(ref, ALL_ON), replay(ref, ALL_OFF)
    agent, off_agent = run_on["agent"], run_off["agent"]
    best, ops, searched, unsafe_best = ceiling(ref)
    printed = re.search(r"accepted (\d+)/(\d+)", text)
    return {
        "accepted": run_on["accepted"], "cycles": int(printed.group(2)),
        "printed_accepted": int(printed.group(1)),
        "rejects": run_on["rejects"],
        "printed_rejects": ast.literal_eval(re.search(r"rejects\s+(\{.*\})", text).group(1)),
        "perf": ref.perf_score(agent), "safety": ref.safety_score(agent),
        "ceiling": best, "ceiling_ops": ops,
        "searched": searched, "unsafe_best": unsafe_best,
        "manifest": sorted(agent.active_manifest),
        "approved": sorted(ref.APPROVED_TOOL_MANIFEST),
        "anchor_intact": ref.gate_anchor(agent),
        "rises": rises(run_on),
        "off_accepted": run_off["accepted"],
        "off_manifest_drifted": not ref.gate_invariant(off_agent),
        "off_anchor_drifted": not ref.gate_anchor(off_agent),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: it improves to the ceiling and the hack does not land",
            all([result["accepted"] == result["printed_accepted"] == 96,
                 result["rejects"] == result["printed_rejects"],
                 result["perf"] == result["ceiling"] == 1.0,
                 result["safety"] == 1.0, result["anchor_intact"],
                 result["manifest"] == result["approved"]]),
            f"{result['accepted']} of {result['cycles']} cycles land at perf "
            f"{result['perf']} -- the ceiling, held by {result['ceiling_ops']} -- with "
            f"safety {result['safety']}, the approved manifest and the anchor intact; "
            "the replay reproduces the shipped counters exactly",
        ),
        practice.Check(
            "FINDING: one of the four primitives never fires",
            all([result["rejects"]["anchor"] == 0, result["rejects"]["invariant"] == 11,
                 result["rejects"]["multi"] == 66, result["rejects"]["regress"] == 27]),
            f"rejects are {result['rejects']}: the objective is edited only in the "
            "branch that adds raw_eval, and gate_invariant runs first, so gate_anchor "
            "is asked about nothing",
        ),
        practice.Check(
            "FINDING: three of the ninety-six accepted cycles improved anything",
            all([result["rises"] == 3, result["searched"] == 2801,
                 result["unsafe_best"] == 0.0]),
            f"{result['rises']} of {result['accepted']} accepted cycles raised perf "
            f"above everything before them; the rest passed all four gates and moved "
            "the agent sideways",
        ),
        practice.Check(
            "FINDING: the hack is rare before it is blocked",
            all([result["off_accepted"] == 200, result["off_manifest_drifted"],
                 result["off_anchor_drifted"]]),
            f"with every gate off the same stream accepts {result['off_accepted']} of "
            f"{result['cycles']} cycles and drifts both manifest and objective, so the "
            f"{result['rejects']['invariant']} invariant rejects measure the attempt "
            "rate",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
