"""Exercise 5 — the demo's CTDE runs its critic on every deploy step.

    Apply CTDE as a design pattern to a hypothetical LLM-agent system (e.g.,
    research agent + summarizer + coder). What is the joint information
    available at design time that is not available at runtime?

Reading of the exercise: the question is answered for the LLM system in the
README, and measured here on the lesson's own grid. The joint information is
whatever the critic reads that an actor cannot. Its value is what a policy
limited to its own observation loses without it.

**ANSWER: the other agents' state and the joint outcome.** On the grid, the
critic reads *both* positions, and an actor has only its own position plus
the pellets. In the LLM system, the design-time joint record is every
agent's full context and output on a task, together with whether the final
answer was right. From that record you can see that the summarizer drops
the caveat the coder needed, or that research and coder both fetched the
same page. At runtime the coder sees only the summary. So CTDE here means
using the joint traces to design each agent's prompt, schema and hand-off
convention, and then running each agent on its own input alone. A policy
restricted that way on the grid recovers 34% of the coordination gap: the
best convention fixed at design time scores 3.048 steps, against 3.212
independent and 2.726 with the joint state.

**FINDING: the shipped "CTDE" is centralized execution.** `run_maddpg_style`
says "Deploy-time only the actors run", yet it calls `_assigned_targets` --
which reads both agents' positions -- on 1363 of the 1363 steps it takes
over the demo's 500 episodes, i.e. on every step. The demo's own LLM
reading, "the router decides which sub-agent advances", is the same thing: a
router consulted at runtime is the critic deployed, not training discipline.

**FINDING: no convention tried closes the remaining 66%.** Three role
conventions, each using only the agent's own index and the pellets, score
3.154 (row-major), 3.048 (column-major) and 3.088 (diagonal). None reaches
2.726. What they lack is which pellet the *other* agent is nearer to, and
that is a fact about the other agent's position. Three conventions do not
prove that no local policy could close the gap. But if the LLM system needs
that kind of fact, the design has to send a message at runtime, because
CTDE only moves information that can be settled before the task starts.

Structure: `convention()` is a decentralized actor pair -- agent 0 takes the
first pellet under a fixed ordering, agent 1 the last -- run on the
reference `Env` and `step_toward`; `critic_calls()` counts how often the
shipped runner reads the joint state.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "20-marl-maddpg-qmix-mappo"
TRIALS = 500
ORDERINGS = {"row-major": lambda p: p, "column-major": lambda p: (p[1], p[0]),
             "diagonal": lambda p: (p[0] + p[1], p)}


def convention(ref, env, key, max_steps=50):
    """Each agent sees only itself and the pellets; roles were fixed at design time."""
    steps = 0
    while not env.done and steps < max_steps:
        order = sorted(env.pellets_remaining, key=key)
        env.agent0 = ref.step_toward(env.agent0, order[0])
        env.agent1 = ref.step_toward(env.agent1, order[-1])
        env.collect_if_on_pellet()
        steps += 1
    return steps


def critic_calls(ref):
    """(critic calls, steps taken) by run_maddpg_style over the demo's episodes."""
    calls, original = [0], ref._assigned_targets

    def counted(env):
        calls[0] += 1
        return original(env)

    ref._assigned_targets = counted
    try:
        steps = sum(ref.run_maddpg_style(ref.Env.new(random.Random(i))) for i in range(TRIALS))
    finally:
        ref._assigned_targets = original
    return calls[0], steps


def mean(ref, run):
    return round(sum(run(ref.Env.new(random.Random(i))) for i in range(TRIALS)) / TRIALS, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    local = {name: mean(ref, lambda env, k=key: convention(ref, env, k))
             for name, key in ORDERINGS.items()}
    return {"local": local, "independent": mean(ref, ref.run_independent),
            "joint": mean(ref, ref.run_maddpg_style), "calls": critic_calls(ref),
            "claims_actors_only": "Deploy-time only the actors run"
                                  in ref.run_maddpg_style.__doc__}


def verify(result):
    best = min(result["local"].values())
    ind, joint = result["independent"], result["joint"]
    recovered = (ind - best) / (ind - joint)
    calls, steps = result["calls"]
    return [
        practice.Check(
            "ANSWER: the other agents' state and the joint outcome",
            best == 3.048 and round(recovered, 2) == 0.34,
            f"with only its own observation the best design-time convention scores {best} "
            f"steps against {ind} independent and {joint} with the joint state -- "
            f"{recovered:.0%} of the gap",
        ),
        practice.Check(
            "FINDING: the shipped CTDE is centralized execution",
            result["claims_actors_only"] and calls == steps == 1363,
            f"the docstring says only the actors run at deploy; _assigned_targets reads both "
            f"positions on {calls} of {steps} steps",
        ),
        practice.Check(
            "FINDING: no convention tried closes the remaining 66%",
            all(v > joint for v in result["local"].values()),
            f"conventions score {result['local']}; none reaches {joint}, because which "
            "pellet the other agent is nearer to is not in an actor's observation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
