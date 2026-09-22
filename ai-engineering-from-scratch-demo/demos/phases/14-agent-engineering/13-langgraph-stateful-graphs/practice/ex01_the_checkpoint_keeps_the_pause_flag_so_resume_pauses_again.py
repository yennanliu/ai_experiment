"""Exercise 1 — the checkpoint keeps the pause flag, so resume pauses again.

    Add a conditional edge from `classify` to `end` when classification
    confidence is below a threshold. Resume the run after a human sets `route`
    manually.

Reading of the exercise: `_classify` has no confidence -- its `else` branch
returns `sales` -- so the edge needs a classifier that reports one first. The
resume half is the more interesting one, because `Runner.run` saves the
checkpoint *before* it pops `_pause_reason`, so the durable copy of the state
is the one that cannot be resumed from.

**ANSWER: a confidence-gated edge to END, and a manual resume.** At a
threshold of **0.5** a gibberish input stops at `classify` with **0** ticket
issued; setting `route` by hand and resuming carries it through to
`sent BUG-...` in **2** further nodes.

**FINDING: the shipped classifier has no way to be unsure.** Its `else`
branch returns `sales`, so `zzz qqq` is routed to sales with confidence
nowhere and a ticket raised. **3** of the classifier's **4** branches are
keyword matches and the fourth is a default that looks like a match.

**FINDING: the saved checkpoint cannot be resumed from.** `run` calls
`checkpointer.save(...)` and *then* `state.pop("_pause_reason")`, so the
stored state still carries the flag: resuming from `load_latest` pauses at
the same node again, forever. The only resumable copy is the one attached to
the exception, which is in memory and dies with the process.

**FINDING: resume re-runs the node it paused at.** Coming back into
`human_gate` executes it a second time, so `step` reaches **5** over a **4**-
node path. Any node with a side effect either has to be idempotent or the
runner has to resume at the *next* node -- and the shipped `resume_from` is
the caller's guess either way.

Structure: `confident_classify()` supplies the missing confidence; the graph
is the lesson's own `build_graph()` with one conditional edge changed.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "13-langgraph-stateful-graphs"
THRESHOLD = 0.5
KEYWORDS = {"refund": ("refund", "money back"), "bug": ("crash", "bug", "error"),
            "sales": ("pricing", "quote")}
GIBBERISH = "zzz qqq"
REAL = "the CLI crashes on ctrl-c"


def confident_classify(state):
    """The lesson's rules, plus the number they were already deciding on."""
    text = state["input"].lower()
    hits = {route: sum(term in text for term in terms)
            for route, terms in KEYWORDS.items()}
    route, best = max(hits.items(), key=lambda row: row[1])
    total = sum(hits.values())
    return {"route": route if best else "unknown",
            "confidence": best / total if total else 0.0,
            "step": state.get("step", 0) + 1}


def gated_graph(ref):
    graph = ref.build_graph()
    graph.nodes["classify"] = confident_classify
    graph.edges["classify"] = []
    graph.add_conditional_edges(
        "classify", router=lambda s: ("low" if s["confidence"] < THRESHOLD
                                      else s["route"]),
        targets={"low": ref.END, "refund": "refund", "bug": "bug", "sales": "sales"})
    return graph


def run(ref, graph, session, state, **kwargs):
    runner = ref.Runner(graph, kwargs.pop("checkpointer"))
    try:
        return runner.run(session, state, **kwargs), None
    except ref.PausedAtNode as paused:
        return paused.state, paused.node


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    graph, ckpt = gated_graph(ref), ref.InMemoryCheckpointer()
    low, _ = run(ref, graph, "low", {"input": GIBBERISH, "step": 0,
                                     "human_approval": True}, checkpointer=ckpt)
    fixed = {**low, "route": "bug", "human_approval": True}
    resumed, _ = run(ref, graph, "low", {}, checkpointer=ckpt,
                     resume_from="bug", state_override=fixed)
    shipped = ref.InMemoryCheckpointer()
    paused_state, node = run(ref, ref.build_graph(), "s1",
                             {"input": REAL, "step": 0, "human_approval": False},
                             checkpointer=shipped)
    saved_node, saved_state = shipped.load_latest("s1")
    from_store, store_node = run(ref, ref.build_graph(), "s1", {},
                                 checkpointer=shipped, resume_from="human_gate",
                                 state_override={**saved_state,
                                                 "human_approval": True})
    from_exc, _ = run(ref, ref.build_graph(), "s1", {}, checkpointer=shipped,
                      resume_from="human_gate",
                      state_override={**paused_state, "human_approval": True})
    return {
        "low_confidence": low["confidence"], "low_route": low["route"],
        "low_ticket": low.get("ticket"), "low_steps": low["step"],
        "resumed_output": resumed.get("output"), "resumed_steps": resumed["step"],
        "shipped_route": ref._classify({"input": GIBBERISH})["route"],
        "shipped_ticket": ref._sales({"input": GIBBERISH})["ticket"][:3],
        "branches": len(KEYWORDS) + 1,
        "paused_node": node, "saved_node": saved_node,
        "saved_has_flag": "_pause_reason" in saved_state,
        "store_repauses": store_node, "exception_output": from_exc.get("output"),
        "final_step": from_exc["step"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a low-confidence input ends at classify and resumes by hand",
            all([result["low_confidence"] == 0.0, result["low_route"] == "unknown",
                 result["low_ticket"] is None, result["low_steps"] == 1,
                 result["resumed_output"] == "sent BUG-zzz qqq", result["resumed_steps"] == 4]),
            f"at a threshold of {THRESHOLD} the gibberish input stops at classify with "
            f"confidence {result['low_confidence']}, route {result['low_route']!r} and "
            f"no ticket. Setting route by hand and resuming carries it to "
            f"{result['resumed_output']!r} at step {result['resumed_steps']}",
        ),
        practice.Check(
            "FINDING: the shipped classifier has no way to be unsure",
            all([result["shipped_route"] == "sales", result["shipped_ticket"] == "SAL",
                 result["branches"] == 4]),
            f"_classify's else branch returns {result['shipped_route']!r}, so gibberish "
            f"is routed to sales and a {result['shipped_ticket']} ticket is raised. "
            f"{result['branches'] - 1} of its {result['branches']} branches are keyword "
            "matches and the fourth is a default that looks like one",
        ),
        practice.Check(
            "FINDING: the saved checkpoint cannot be resumed from",
            all([result["paused_node"] == "human_gate",
                 result["saved_node"] == "human_gate",
                 result["saved_has_flag"] is True,
                 result["store_repauses"] == "human_gate",
                 result["exception_output"] == "sent BUG-the CLI cras"]),
            f"run saves the checkpoint and then pops _pause_reason, so the stored state "
            f"still carries it ({result['saved_has_flag']}) and resuming from "
            f"load_latest pauses at {result['store_repauses']!r} again. Only the copy "
            f"attached to the exception resumes, reaching {result['exception_output']!r}",
        ),
        practice.Check(
            "FINDING: resume re-runs the node it paused at",
            all([result["final_step"] == 5]),
            f"coming back into human_gate executes it a second time, so step reaches "
            f"{result['final_step']} over a 4-node path. A node with a side effect has "
            "to be idempotent or the runner has to resume at the next node, and "
            "resume_from is the caller's guess either way",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
