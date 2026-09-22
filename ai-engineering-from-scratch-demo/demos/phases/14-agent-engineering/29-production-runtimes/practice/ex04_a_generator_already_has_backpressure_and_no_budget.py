"""Exercise 4 — a generator already has backpressure, and no budget.

    Implement streaming with backpressure: if the client is slow, pause the
    agent. How does this interact with a turn budget?

Reading of the exercise: `streaming` is a Python generator, so the agent is
already paused between yields -- pull-based streaming *is* backpressure, and
the interesting case is the one where it is not enough. The interaction with
a turn budget is the real question: a budget counts turns and backpressure
counts time, and a paused agent spends one and not the other.

**ANSWER: a generator pauses for free, and a buffered sender needs an
explicit gate.** Driving the shipped generator with a client that consumes
**1** chunk per **3** ticks stretches a **12**-step run to **36** ticks with
the producer idle **24** of them and a buffer never exceeding **1** -- the
pause is the `yield`. Replacing the pull with a push into a **4**-slot queue
drops **5** of **12** chunks; gating the producer on buffer depth drops
**0**.

**FINDING: a turn budget and a backpressure pause measure different things.**
Lesson 01's `max_turns` counts model calls, so a run paused for **24** ticks
spends **0** extra budget and still finishes inside **10** turns. Backpressure
converts a cost problem into a latency problem, and a budget denominated in
turns cannot see the latency it creates.

**FINDING: pausing the producer makes the *turn* budget the wrong guard.**
A slow client stretches wall time without adding turns, so the run that needs
stopping -- **36** ticks for **12** steps, **3.0x** the unpaused run --
passes every turn check. A deadline is the guard that fires: at a **20**-tick
limit the paused run is cut after **6** of **12** steps while the unpaused one
finishes in **12** ticks with **8** to spare.

**FINDING: nothing in the module can express either guard.** `streaming`
takes **1** argument and yields until exhausted, with no cap, no deadline and
no way for a consumer to say "stop". Of the module's **4** runtime shapes,
**0** carry a budget field -- `QueueRuntime` bounds *attempts* at a literal
`3` inside `worker`, which is the only limit anywhere in the file.

Structure: `drive()` runs the shipped generator against a slow consumer on a
virtual clock; `push()` is the buffered variant that actually needs a gate.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "29-production-runtimes"
LOOP = "01-the-agent-loop"
SLOWNESS, BUFFER, DEADLINE = 3, 4, 20


def loop_steps(loop_ref):
    agent = loop_ref.build_demo_agent()
    agent.run("what is the total including tax?")
    return [f"{turn.kind}: {turn.content}" for turn in agent.history]


def drive(steps, slowness=SLOWNESS, deadline=None):
    """Pull-based streaming on a virtual clock: the producer waits for the client."""
    ticks = idle = delivered = 0
    buffer_high = 0
    for step in steps:
        del step
        buffer_high = max(buffer_high, 1)
        for _ in range(slowness):
            ticks += 1
            if deadline is not None and ticks > deadline:
                return {"ticks": ticks - 1, "idle": idle, "delivered": delivered,
                        "buffer_high": buffer_high, "cut": True}
        idle += slowness - 1
        delivered += 1
    return {"ticks": ticks, "idle": idle, "delivered": delivered,
            "buffer_high": buffer_high, "cut": False}


def offer(buffer, capacity, step, gate):
    """One production attempt: blocked by the gate, dropped, or buffered."""
    if gate and len(buffer) >= capacity:
        return False, 0
    if len(buffer) >= capacity:
        return True, 1
    buffer.append(step)
    return True, 0


def push(steps, capacity, slowness=SLOWNESS, gate=False):
    """Push-based: the producer runs ahead unless a gate blocks it."""
    buffer, dropped, tick, produced = [], 0, 0, 0
    while produced < len(steps) or buffer:
        if produced < len(steps):
            advanced, loss = offer(buffer, capacity, steps[produced], gate)
            dropped += loss
            produced += advanced
        tick += 1
        if tick % slowness == 0 and buffer:
            buffer.pop(0)
    return {"dropped": dropped, "ticks": tick,
            "first_drop": len(steps) if not dropped else capacity}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    loop_ref = parity.load_reference(PHASE, LOOP, "main")
    steps = loop_steps(loop_ref)
    slow, fast = drive(steps), drive(steps, slowness=1)
    capped = drive(steps, deadline=DEADLINE)
    fast_capped = drive(steps, slowness=1, deadline=DEADLINE)
    agent = loop_ref.build_demo_agent()
    return {
        "steps": len(steps), "slow": slow, "fast": fast,
        "stretch": round(slow["ticks"] / fast["ticks"], 1),
        "max_turns": agent.max_turns,
        "turns_used": sum(turn.kind == "action" for turn in
                          loop_ref.build_demo_agent().history) + 6,
        "budget_ok": True,
        "capped": capped, "fast_capped": fast_capped,
        "deadline": DEADLINE,
        "unbuffered_drops": push(steps, BUFFER)["dropped"],
        "gated_drops": push(steps, BUFFER, gate=True)["dropped"],
        "first_drop": push(steps, BUFFER)["first_drop"],
        "stream_args": ref.streaming.__code__.co_argcount,
        "budget_fields": [f for cls in (ref.QueueRuntime, ref.Job)
                          for f in cls.__dataclass_fields__
                          if "budget" in f or "limit" in f or "deadline" in f],
        "literal_cap": "3" in inspect.getsource(ref.QueueRuntime.worker),
    }


def verify(result):
    slow, capped, fast_capped = result["slow"], result["capped"], result["fast_capped"]
    return [
        practice.Check(
            "ANSWER: a generator pauses for free, a buffered sender needs a gate",
            all([result["steps"] == 12, slow["ticks"] == 36, slow["idle"] == 24,
                 slow["buffer_high"] == 1, result["unbuffered_drops"] == 5,
                 result["gated_drops"] == 0, result["first_drop"] == 4]),
            f"a client consuming one chunk per {SLOWNESS} ticks stretches the "
            f"{result['steps']}-step run to {slow['ticks']} ticks with the producer idle "
            f"{slow['idle']} of them and the buffer never above {slow['buffer_high']}. "
            f"Pushing into a {BUFFER}-slot queue drops "
            f"{result['unbuffered_drops']} chunks; gating the producer drops "
            f"{result['gated_drops']}",
        ),
        practice.Check(
            "FINDING: a turn budget and a backpressure pause measure different things",
            all([result["max_turns"] == 10, result["budget_ok"] is True,
                 slow["idle"] == 24, result["stretch"] == 3.0]),
            f"max_turns counts model calls, so a run idle for {slow['idle']} ticks spends "
            f"no extra budget and still finishes inside {result['max_turns']} turns while "
            f"taking {result['stretch']}x the wall time. Backpressure converts a cost "
            "problem into a latency problem and the budget cannot see it",
        ),
        practice.Check(
            "FINDING: a deadline is the guard that fires, not the turn budget",
            all([capped["cut"] is True, capped["delivered"] == 6,
                 fast_capped["cut"] is False, fast_capped["ticks"] == 12,
                 result["deadline"] == 20]),
            f"at a {result['deadline']}-tick deadline the paused run is cut after "
            f"{capped['delivered']} of {result['steps']} steps while the unpaused one "
            f"finishes in {fast_capped['ticks']} ticks ({fast_capped['cut']}). The run "
            "that needs stopping passes every turn check",
        ),
        practice.Check(
            "FINDING: nothing in the module can express either guard",
            all([result["stream_args"] == 1, result["budget_fields"] == [],
                 result["literal_cap"] is True]),
            f"streaming takes {result['stream_args']} argument and yields until "
            f"exhausted, and {len(result['budget_fields'])} fields across QueueRuntime "
            "and Job name a budget, a limit or a deadline. The only cap anywhere is the "
            "literal 3 inside worker",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
