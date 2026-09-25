"""Exercise 1 — all three orchestrators produce the same transcript.

    Run `code/main.py` three times with different agent policies. Observe how
    the orchestrator choice changes which agents run.

Reading of the exercise: run it and diff the three pools, because the
instruction presupposes the thing to be observed. With the policies the
lesson ships, the observation is that nothing changes -- the demo's closing
line is refuted by its own output.

**ANSWER: it does not change which agents run, and only one of the three
orchestrators can.** Static, handoff-driven and LLM-selected each produce the
same **3** messages, in the same order, with the same content: **0** of the
**3** pairs differ. Sweeping the one field an agent policy controls -- the
researcher's `handoff` target -- over all **4** values it can take, the static
and round-robin transcripts are identical in **4** of 4, and the handoff
orchestrator diverges in **3**. "Only the orchestrator choice changes who
speaks when" is true of one orchestrator out of three.

**FINDING: the static orchestrator stores handoffs and reads none.** After a
static run, **3** of **3** messages in the pool carry a `handoff` key, and
`StaticOrchestrator.run` never mentions the word. It walks `self.order`. So
every policy change an agent can express is invisible to it, which is why it
tracks the round-robin selector exactly: neither reads what the agents decide.

**FINDING: a self-handoff ends by running out of budget, quietly.** Pointing
the researcher at itself makes the handoff orchestrator emit **10** messages
-- exactly `max_steps` -- and return normally. Nothing in the pool, the return
value or the state records that the loop was cut off rather than finished, so
a cycle and a completion are indistinguishable to the caller.

**FINDING: `max_steps` means two different things.** `StaticOrchestrator`
slices `self.order[:max_steps]`, so it caps the *plan*; the other two count
iterations with `range(max_steps)`, capping the *run*. **3** orchestrators,
**2** meanings, one parameter name -- and the slicing one silently drops the
tail of a longer order instead of stopping at the limit.

Structure: `transcript()` runs one orchestrator over a team whose researcher
hands off to a chosen target; `sweep()` covers every target.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "04-primitive-model"
ORDER = ["researcher", "writer", "reviewer"]
TARGETS = ORDER + ["done"]
BUDGET = 10


def team_handing_off_to(ref, target):
    """The shipped team, with the researcher's one policy decision made explicit."""
    team = ref.make_team()
    team["researcher"] = ref.Agent(
        "researcher", "Gather facts.",
        lambda _state, to=target: {"content": "note 1", "handoff": to})
    return team


def orchestrators(ref):
    """The three the lesson ships, built the way `main()` builds them."""
    return {"static": ref.StaticOrchestrator(ORDER),
            "handoff": ref.HandoffOrchestrator("researcher"),
            "llm": ref.LLMSelectorOrchestrator("researcher", ref.round_robin_selector)}


def transcript(ref, orchestrator, team):
    """Who spoke, in order, and what they said."""
    state = ref.SharedState()
    orchestrator.run(team, state)
    return [(m["from"], m["content"], m.get("handoff")) for m in state.snapshot()]


def shipped(ref):
    """The three pools `main()` prints, compared message for message."""
    pools = {name: transcript(ref, orchestrator, ref.make_team())
             for name, orchestrator in orchestrators(ref).items()}
    names = list(pools)
    pairs = [(a, b) for index, a in enumerate(names) for b in names[index + 1:]]
    return pools, sum(pools[a] != pools[b] for a, b in pairs), len(pairs)


def sweep(ref):
    """Vary the researcher's handoff target; record which orchestrators move."""
    rows = {}
    for target in TARGETS:
        rows[target] = {name: [m[0] for m in transcript(ref, orchestrator,
                                                        team_handing_off_to(ref, target))]
                        for name, orchestrator in orchestrators(ref).items()}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    pools, differing, pairs = shipped(ref)
    rows = sweep(ref)
    static_run = inspect.getsource(ref.StaticOrchestrator.run)
    return {
        "messages": len(pools["static"]), "differing": differing, "pairs": pairs,
        "identical": pools["static"] == pools["handoff"] == pools["llm"],
        "targets": len(TARGETS),
        "static_tracks_llm": sum(row["static"] == row["llm"] for row in rows.values()),
        "handoff_moves": sum(row["handoff"] != rows["writer"]["handoff"]
                             for row in rows.values()),
        "carry_handoff": sum("handoff" in m[2:] or m[2] is not None
                             for m in pools["static"]),
        "static_reads_handoff": "handoff" in static_run,
        "cycle_length": len(rows["researcher"]["handoff"]), "budget": BUDGET,
        "sliced": src.count("self.order[:max_steps]"),
        "counted": src.count("range(max_steps)"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: it does not change which agents run, and only one of three can",
            all([result["identical"], result["differing"] == 0, result["pairs"] == 3,
                 result["messages"] == 3, result["static_tracks_llm"] == 4,
                 result["handoff_moves"] == 3]),
            f"the three orchestrators produce the same {result['messages']} messages -- "
            f"{result['differing']} of {result['pairs']} pairs differ; across all "
            f"{result['targets']} handoff targets the static and round-robin transcripts "
            f"match in {result['static_tracks_llm']} of {result['targets']}, and only "
            f"the handoff orchestrator moves, in {result['handoff_moves']}",
        ),
        practice.Check(
            "FINDING: the static orchestrator stores handoffs and reads none",
            all([result["carry_handoff"] == 3, not result["static_reads_handoff"]]),
            f"after a static run {result['carry_handoff']} of {result['messages']} "
            "messages carry a handoff key and StaticOrchestrator.run never mentions the "
            "word -- it walks self.order, so every decision an agent can express is "
            "invisible to it",
        ),
        practice.Check(
            "FINDING: a self-handoff ends by running out of budget, quietly",
            result["cycle_length"] == result["budget"],
            f"pointing the researcher at itself makes the handoff orchestrator emit "
            f"{result['cycle_length']} messages -- exactly max_steps -- and return "
            "normally; nothing records that the loop was cut off, so a cycle and a "
            "completion look the same to the caller",
        ),
        practice.Check(
            "FINDING: max_steps means two different things",
            all([result["sliced"] == 1, result["counted"] == 2]),
            f"StaticOrchestrator slices self.order[:max_steps], capping the plan, while "
            f"the other {result['counted']} count iterations with range(max_steps), "
            "capping the run -- one parameter name, two meanings, and the slicing one "
            "silently drops the tail of a longer order",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
