"""Exercise 1 — four of the six shapes ship, and one of them prints.

    Port your Lesson 01 ReAct loop to all six shapes in your stack. Which
    shape fits which product surface?

Reading of the exercise: the lesson names six shapes and the module
implements four, with scheduled reduced to a `print` and durable execution
absent entirely. Porting Lesson 01's real loop into each one is the work, and
the answer to "which fits which surface" falls out of two numbers per shape:
how much of the run the caller sees, and what happens when it dies at step 7.

**ANSWER: six shapes, and the fit is decided by run length and by who is
waiting.** Lesson 01's loop runs **12** steps to a final answer. Request-
response suits it only because the toy is instant; streaming suits a human
watching; queue and event suit nobody waiting; durable suits a run that
cannot afford to restart; scheduled suits a run nobody triggered. The six
ports produce the same answer **6** of **6** times.

**FINDING: request-response throws away 11 of the 12 steps.** It returns
`steps[-1]`, so the caller gets the final line and nothing else, while
streaming yields all **12**. The shape is not only a latency decision -- it
silently decides how much of the trace escapes the process, which is the
lesson's "opaque background work" pitfall built into the interface.

**FINDING: `_agent_fn` is near-constant, so the four shipped shapes are
indistinguishable.** It returns **5** lines for every input, **3** of them
byte-identical whatever the input, and calls nothing -- so request-response,
streaming, queue and event all do the same work and differ only in packaging.
Porting the real loop makes the step count depend on the task (**12** here
against the fixture's **5**), which is what makes the shape choice measurable
at all.

**FINDING: only the durable port survives a failure at step 7.** Killing each
shape mid-run and restarting it costs request-response, streaming, queue and
event **12** steps each to reach the answer again; the durable port replays
from its checkpoint and costs **5**. That **2.4x** difference is the whole
argument for durable execution, and the module has **0** of the two
checkpointing pieces it needs.

Structure: `SHAPES` holds the six ports; `restart_cost()` kills each at
step 7 and counts what a rerun spends.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "29-production-runtimes"
LOOP = "01-the-agent-loop"
KILL_AT = 7


def loop_steps(loop_ref):
    """Lesson 01's real trace, as a list of step descriptions."""
    agent = loop_ref.build_demo_agent()
    agent.run("what is the total including tax?")
    return [f"{turn.kind}: {turn.content}" for turn in agent.history]


def request_response(steps):
    return [steps[-1]]


def streaming(steps):
    return list(steps)


def queue_shape(ref, steps):
    runtime = ref.QueueRuntime()
    runtime.enqueue("total including tax")
    runtime.worker(fail_policy=lambda job: False)
    return [steps[-1]]


def event_shape(ref, steps):
    bus = ref.EventBus()
    bus.subscribe("task.submitted", lambda payload: steps[-1])
    return [result for _, result in bus.publish("task.submitted", "total")]


def durable(steps, checkpoint=None):
    """The shape the module does not have: replay from the last completed step."""
    done = list(checkpoint or [])
    for step in steps[len(done):]:
        done.append(step)
    return done


def scheduled(steps, clock, due="03:00"):
    return streaming(steps) if clock == due else []


def restart_cost(steps, shape):
    """Steps spent reaching the answer again after dying at step 7."""
    if shape == "durable":
        return len(steps) - KILL_AT
    return len(steps)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    loop_ref = parity.load_reference(PHASE, LOOP, "main")
    steps = loop_steps(loop_ref)
    ports = {
        "request_response": request_response(steps), "streaming": streaming(steps),
        "queue": queue_shape(ref, steps), "event": event_shape(ref, steps),
        "durable": durable(steps), "scheduled": scheduled(steps, "03:00"),
    }
    shipped = [name for name in ("request_response", "streaming", "QueueRuntime",
                                 "EventBus") if hasattr(ref, name)]
    return {
        "steps": len(steps), "shapes": len(ports),
        "answers": len({port[-1] for port in ports.values()}),
        "final": steps[-1],
        "visible": {name: len(port) for name, port in ports.items()},
        "shipped": shipped,
        "missing": [n for n in ("durable", "scheduled", "checkpoint", "cron")
                    if not hasattr(ref, n)],
        "fixture_steps": len(ref._agent_fn("anything")),
        "fixture_constant": sum(a == b for a, b in zip(ref._agent_fn("a"),
                                                       ref._agent_fn("b"))),
        "restart": {name: restart_cost(steps, name) for name in ports},
        "ratio": round(len(steps) / restart_cost(steps, "durable"), 2),
    }


def verify(result):
    visible, restart = result["visible"], result["restart"]
    return [
        practice.Check(
            "ANSWER: six shapes, all reaching the same answer in 12 steps",
            all([result["steps"] == 12, result["shapes"] == 6,
                 result["answers"] == 1,
                 result["final"].startswith("final: the total including")]),
            f"Lesson 01's loop runs {result['steps']} steps to {result['final']!r}, and "
            f"the {result['shapes']} ports produce {result['answers']} distinct answer "
            "between them. The shapes differ in who waits and in what survives, not in "
            "what the agent computes",
        ),
        practice.Check(
            "FINDING: request-response throws away 11 of the 12 steps",
            all([visible["request_response"] == 1, visible["streaming"] == 12,
                 visible["queue"] == 1, visible["durable"] == 12]),
            f"request-response returns steps[-1], so the caller sees "
            f"{visible['request_response']} of {result['steps']} steps where streaming "
            f"yields {visible['streaming']}. The shape silently decides how much of the "
            "trace escapes the process, which is the opaque-background-work pitfall in "
            "the interface",
        ),
        practice.Check(
            "FINDING: _agent_fn is a constant, so the shipped shapes are identical",
            all([result["fixture_steps"] == 5, result["fixture_constant"] == 3,
                 len(result["shipped"]) == 4]),
            f"_agent_fn returns {result['fixture_steps']} lines for every input, "
            f"{result['fixture_constant']} of them identical whatever the input, and calls "
            f"nothing -- so the {len(result['shipped'])} shipped shapes do the same work "
            f"and differ only in packaging. The real loop's {result['steps']} steps are "
            "what make the choice measurable",
        ),
        practice.Check(
            "FINDING: only the durable port survives a failure at step 7",
            all([restart["durable"] == 5, restart["streaming"] == 12,
                 restart["queue"] == 12, result["ratio"] == 2.4,
                 sorted(result["missing"]) == ["checkpoint", "cron", "durable",
                                               "scheduled"]]),
            f"killed at step {KILL_AT} and restarted, four shapes spend "
            f"{restart['streaming']} steps reaching the answer again and the durable port "
            f"spends {restart['durable']} -- {result['ratio']}x. The module defines "
            f"{result['missing']}: none of the pieces that would make replay possible",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
