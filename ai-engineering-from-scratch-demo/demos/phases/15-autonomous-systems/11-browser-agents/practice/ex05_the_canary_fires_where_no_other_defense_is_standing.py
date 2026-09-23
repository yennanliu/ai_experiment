"""Exercise 5 — the canary fires where no other defense is standing.

    Design a memory canary for a browser-agent setting. What would you store,
    where, and what triggers the alarm?

Reading of the exercise: three questions, and the third is the one that
decides the design -- a canary is only useful if its trigger sits somewhere
the other defenses do not. So the design is stated and then placed on the
lesson's own 3-by-4 grid to see which cells it is alone in.

**ANSWER: a credential-shaped token, in the store the agent reads
credentials from, tripping on any outbound payload that contains it.** Not on
an endpoint, not on a page, not on an approval -- on the *value* leaving.
Placed on the grid, the token appears in the posted body in **3** of **12**
cells and in **0** benign cells: precision **1.0** with no rule about URLs.

**FINDING: it is the only defense standing in those three cells.** The cells
where the secret leaves are naive-vs-visible, naive-vs-fragment and
sanitizer-vs-fragment. The read/write boundary stops **0** of them, because
in each the boundary is not enabled -- which is the point: the canary is the
layer that still reports when the configured defenses are the ones that
failed.

**FINDING: the simulator has nowhere to put it.** `run_agent` writes the
secret into the body from a module constant and reads **0** credential
stores, so there is no cookie jar, profile or saved-card page to seed. The
"where" half of the exercise is unanswerable against this code, and that
absence is the finding: a canary the agent never reads is a canary that is
never carried.

**FINDING: the trigger has no call site.** `run_agent` *returns* an
`AgentResult` with **4** fields and sends nothing, so the egress point the
alarm would hook does not exist in this module. The deliverable of a canary
design is therefore the checkpoint, not the token -- and it is strictly
detection: the payload has already been composed by the time the value is
visible.

Structure: `leaks()` walks the grid and asks which cells carry the secret;
`shape()` checks that a canary can be made to look like the real thing.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "11-browser-agents"

CANARY = "USER_SESSION_TOKEN=sk-canary-0000"


def leaks(ref):
    """{(case, defense): whether the posted body carries the secret}."""
    return {(name, defense): ref.run_agent(page, defense).posted_body == ref.SENSITIVE
            for name, page in ref.CASES for defense in ref.DEFENSES}


def benign_cells(cells):
    return {key: carried for key, carried in cells.items() if key[0] == "benign page"}


def stopped_by_boundary(ref, cells):
    """Of the leaking cells, how many the boundary would have caught."""
    return sum(1 for (name, defense) in cells
               if cells[(name, defense)] and defense in ("rw_boundary", "both"))


def shape(ref):
    """Can a canary be made indistinguishable from the real secret?"""
    real, fake = ref.SENSITIVE, CANARY
    return (real.split("=")[0] == fake.split("=")[0],
            real.split("=")[1][:3] == fake.split("=")[1][:3])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cells = leaks(ref)
    agent = inspect.getsource(ref.run_agent)
    return {
        "cells": len(cells),
        "leaking": sorted(key for key, carried in cells.items() if carried),
        "benign_leaks": sum(benign_cells(cells).values()),
        "precision": 1.0 if sum(benign_cells(cells).values()) == 0 else 0.0,
        "boundary_would_catch": stopped_by_boundary(ref, cells),
        "reads_a_store": agent.count("store") + agent.count("cookie") + agent.count("profile"),
        "secret_from_constant": "target_body = SENSITIVE" in agent,
        "result_fields": len(ref.AgentResult.__dataclass_fields__),
        "sends": agent.count("requests.") + agent.count("urlopen") + agent.count("send("),
        "returns": agent.count("return AgentResult"),
        "prefix_matches": shape(ref)[0],
        "value_prefix_matches": shape(ref)[1],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the value leaving is the trigger -- 3 of 12 cells, 0 benign",
            all([result["cells"] == 12, len(result["leaking"]) == 3,
                 result["benign_leaks"] == 0, result["precision"] == 1.0,
                 result["prefix_matches"], result["value_prefix_matches"]]),
            f"the token appears in the posted body in {len(result['leaking'])} of "
            f"{result['cells']} cells -- {result['leaking']} -- and "
            f"{result['benign_leaks']} benign cells, at precision "
            f"{result['precision']}, with a shape matching the real secret",
        ),
        practice.Check(
            "FINDING: it is the only defense standing in those three cells",
            result["boundary_would_catch"] == 0,
            f"the read/write boundary stops {result['boundary_would_catch']} of the "
            "leaking cells, because in each of them it is the defense that is not "
            "enabled -- the canary reports when the configured layers are the ones that "
            "failed",
        ),
        practice.Check(
            "FINDING: the simulator has nowhere to put it",
            all([result["reads_a_store"] == 0, result["secret_from_constant"]]),
            f"run_agent reads {result['reads_a_store']} credential stores and writes the "
            "secret in from a module constant, so there is no cookie jar or profile to "
            "seed -- a canary the agent never reads is never carried",
        ),
        practice.Check(
            "FINDING: the trigger has no call site",
            all([result["sends"] == 0, result["returns"] >= 1,
                 result["result_fields"] == 4]),
            f"run_agent performs {result['sends']} sends and returns an AgentResult of "
            f"{result['result_fields']} fields, so the egress point the alarm hooks does "
            "not exist here -- the deliverable is the checkpoint, not the token",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
