"""Exercise 2 — nothing in the Agent primitive can say "not yet".

    Implement a fourth orchestrator type: a queue-driven one where agents poll
    shared state for work. What deadlock can happen, and how do you detect it?

Reading of the exercise: write the poller, then find where readiness has to
live -- because polling needs an agent to be able to decline a turn, and
`Agent.run` is typed to return a message every time.

**ANSWER: a zero-append sweep is ambiguous, and the deadlock signal is that
sweep plus a non-empty waiting set.** A healthy team -- writer needs a
researcher message, reviewer needs a writer message -- stalls on sweep **2**
with **0** agents still waiting: that is completion. A circular team stalls on
sweep **1** with **3** still waiting: that is deadlock. Both look identical if
you only count appends, which is the obvious detector and the wrong one. One
sweep is enough either way, before any timeout expires, because the pool is
the only input any policy reads.

**FINDING: readiness cannot live in the agent.** All **3** shipped policies
contain exactly **1** `return` and **0** early exits: every one of them
produces a message unconditionally. `Message` is a bare `dict` with no "no
work for me" shape, so the preconditions the poller needs are a table the
orchestrator keeps on the side -- a fifth thing, in a lesson whose argument is
that there are four.

**FINDING: with the shipped policies the fourth orchestrator changes nothing
either.** Because no agent can decline, every sweep runs everyone, and the
queue orchestrator produces the same **3** messages in the same order as the
other three. The lesson's demo would print a fourth identical pool.

**FINDING: polling order decides the verdict.** Sweeping the writer before the
researcher makes `writer_policy`'s "Draft with no research yet." branch
reachable for the first time -- no shipped orchestrator can reach it. The
reviewer then approves by searching for the substring `"summarizing"` in the
writer's own output, finds nothing, and returns **needs revision**. The
approval gate is a word the writer happens to use, so changing who polls first
changes whether the work passes review.

Structure: `QueueOrchestrator` is the exercise; `ready` is the precondition
table it has to keep because the Agent primitive has nowhere to put one.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "04-primitive-model"
ORDER = ["researcher", "writer", "reviewer"]
POLICIES = ("researcher_policy", "writer_policy", "reviewer_policy")


def spoke(state, name):
    """Has this agent put anything in the pool yet?"""
    return any(m["from"] == name for m in state.snapshot())


PRECONDITIONS = {"researcher": lambda state: True,
                 "writer": lambda state: spoke(state, "researcher"),
                 "reviewer": lambda state: spoke(state, "writer")}

DEADLOCKED = {"researcher": lambda state: spoke(state, "reviewer"),
              "writer": lambda state: spoke(state, "researcher"),
              "reviewer": lambda state: spoke(state, "writer")}
OPEN = {name: (lambda state: True) for name in ORDER}


class QueueOrchestrator:
    """The fourth orchestrator: sweep the poll order, run whoever is ready, once each."""

    def __init__(self, ready, poll_order=None):
        self.ready, self.poll_order = ready, poll_order or ORDER

    def run(self, team, state, max_steps=10):
        """Sweep until nothing appends; report that sweep and who never ran."""
        done = set()
        for sweep in range(max_steps):
            appended = 0
            for name in self.poll_order:
                if name not in done and self.ready[name](state):
                    state.append(team[name].run(state))
                    done.add(name)
                    appended += 1
            if appended == 0:
                return sweep + 1, sorted(set(self.poll_order) - done)
        return None, sorted(set(self.poll_order) - done)


def pool(ref, orchestrator):
    """Run an orchestrator over a fresh team and report who spoke."""
    team, state = ref.make_team(), ref.SharedState()
    outcome = orchestrator.run(team, state)
    return [m["from"] for m in state.snapshot()], state.snapshot(), outcome or (None, [])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    static, _, _ = pool(ref, ref.StaticOrchestrator(ORDER))
    queued, messages, healthy = pool(ref, QueueOrchestrator(PRECONDITIONS))
    _, _, blocked = pool(ref, QueueOrchestrator(DEADLOCKED))
    writer_first, unresearched, _ = pool(
        ref, QueueOrchestrator(OPEN, poll_order=["writer", "reviewer", "researcher"]))
    sources = {name: inspect.getsource(getattr(ref, name)) for name in POLICIES}
    return {
        "agents": len(ORDER), "static": static, "queued": queued,
        "same_as_static": queued[:len(static)] == static,
        "messages": len(messages), "healthy": healthy, "blocked": blocked,
        "returns": sum(body.count("return ") for body in sources.values()),
        "early_exits": sum(body.count("return None") for body in sources.values()),
        "policies": len(sources),
        "message_is_dict": "Message = dict" in inspect.getsource(ref),
        "no_research": any("no research yet" in m["content"] for m in unresearched),
        "verdict": next(m["content"] for m in reversed(unresearched)
                        if m["from"] == "reviewer"),
        "gate": "summarizing" in inspect.getsource(ref.reviewer_policy),
        "writer_first": writer_first[0],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a zero-append sweep is ambiguous; the signal is that plus a waiting set",
            all([list(result["blocked"]) == [1, sorted(ORDER)],
                 list(result["healthy"]) == [2, []], result["agents"] == 3]),
            f"the circular team stalls on sweep {result['blocked'][0]} with "
            f"{len(result['blocked'][1])} agents still waiting, while the healthy one "
            f"stalls on sweep {result['healthy'][0]} with {len(result['healthy'][1])} -- "
            "so a zero-append sweep alone is ambiguous, and the deadlock signal is that "
            "sweep plus a non-empty waiting set",
        ),
        practice.Check(
            "FINDING: readiness cannot live in the agent",
            all([result["returns"] == 3, result["early_exits"] == 0,
                 result["policies"] == 3, result["message_is_dict"]]),
            f"all {result['policies']} shipped policies contain exactly "
            f"{result['returns'] // result['policies']} return and "
            f"{result['early_exits']} early exits, and Message is a bare dict with no "
            "'no work for me' shape -- so preconditions become a table the orchestrator "
            "keeps on the side, a fifth thing in a lesson about four",
        ),
        practice.Check(
            "FINDING: with the shipped policies the fourth orchestrator changes nothing",
            all([result["same_as_static"], result["messages"] == 3]),
            f"because no agent can decline, every sweep runs everyone and the queue "
            f"orchestrator produces the same {result['messages']} messages in the same "
            f"order as the static one ({', '.join(result['queued'])})",
        ),
        practice.Check(
            "FINDING: polling order decides the verdict",
            all([result["writer_first"] == "writer", result["no_research"],
                 result["gate"], "needs revision" in result["verdict"]]),
            f"sweeping the writer first makes the 'no research yet' branch reachable, "
            f"which no shipped orchestrator can do; the reviewer then searches the "
            f"writer's own output for the substring 'summarizing', fails, and returns "
            f"{result['verdict']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
