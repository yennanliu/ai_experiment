"""Exercise 3 — the handler times itself and calls the number instantiation.

    Benchmark: measure agent instantiation latency on your stack. Does Agno's
    2μs matter to your workload?

Reading of the exercise: a wall-clock number measured here would describe
this machine, so the question is answered arithmetically instead -- **2μs**
against a stated per-request cost, which is a ratio anyone can recompute.
What *is* measured on this machine is structural: how much work
instantiation actually is, and what the shipped handler's microsecond figure
is a figure for.

**ANSWER: it matters below about 200μs of per-request work and nowhere
else.** At the lesson's **2μs**, instantiation is **0.0004%** of a **500**ms
model call, **0.02%** of a **10**ms local call and **2.0%** of a **100μs**
in-process one. For it to reach **1%** of a request, the rest of the request
has to finish in **198μs** -- which excludes anything that touches a network.

**FINDING: the shipped handler measures the handler, not instantiation.**
`agno_request_handler` starts its timer *after* the agent exists and includes
`agent.run(prompt)` plus two session appends inside it. The microsecond
figure in its return string is therefore the whole request, and the agent is
constructed by the caller before the clock starts.

**FINDING: instantiation is two attribute assignments.** `AgnoAgent` is a
dataclass with **2** fields and no `__post_init__`, so constructing one does
the work of a two-element tuple. A number that small is a statement about
the *absence* of a constructor -- framework comparisons at this scale
measure how much setup a framework insists on, not how fast it runs.

**FINDING: the session store is the part that scales with traffic.**
`AgnoSession` holds `dict[str, list[str]]` and `append` never trims, so
**1000** turns across **100** sessions leave **1000** strings resident. Per
agent the lesson quotes **3.75 KiB**; per *session* the toy has no bound at
all, which is the number a stateless-backend design actually has to answer
for.

Structure: `share()` is the arithmetic; everything else reads the shipped
handler and the shipped session store.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "18-agno-and-mastra-runtimes"
INSTANTIATION_US = 2.0
WORKLOADS = {"model call 500ms": 500_000.0, "local model 10ms": 10_000.0,
             "in-process 100us": 100.0}


def share(overhead, request_us):
    """Instantiation as a percentage of one request."""
    return round(100 * overhead / (overhead + request_us), 4)


def breakeven(overhead, target_percent):
    """The per-request budget at which overhead reaches `target_percent`."""
    return round(overhead * (100 - target_percent) / target_percent, 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    session = ref.AgnoSession()
    agent = ref.AgnoAgent(name="a", fn=ref._agno_agent_fn)
    reply = ref.agno_request_handler(session, agent, "s1", "hello")
    for index in range(1000):
        session.append(f"s{index % 100}", f"turn {index}")
    handler_source = inspect.getsource(ref.agno_request_handler)
    before_timer = handler_source.split("start = time.perf_counter_ns()")[0]
    timed = handler_source.split("start = time.perf_counter_ns()")[1]
    return {
        "shares": {name: share(INSTANTIATION_US, cost)
                   for name, cost in WORKLOADS.items()},
        "breakeven_1pc": breakeven(INSTANTIATION_US, 1.0),
        "overhead": INSTANTIATION_US,
        "reply_has_us": reply.endswith("us)"),
        "constructs_before_timer": before_timer.count("AgnoAgent("),
        "timed_calls": timed.count("agent.run(") + timed.count("session.append("),
        "agent_fields": list(ref.AgnoAgent.__dataclass_fields__),
        "post_init": hasattr(ref.AgnoAgent, "__post_init__"),
        "sessions": len(session._turns),
        "resident": sum(len(turns) for turns in session._turns.values()),
        "trims": inspect.getsource(ref.AgnoSession.append).count("pop"),
        "quoted_kib": 3.75,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 2us matters below about 200us of per-request work",
            all([result["shares"]["model call 500ms"] == 0.0004,
                 result["shares"]["local model 10ms"] == 0.02,
                 result["shares"]["in-process 100us"] == 1.9608,
                 result["breakeven_1pc"] == 198.0]),
            f"at {result['overhead']}us, instantiation is "
            f"{result['shares']['model call 500ms']}% of a 500ms model call, "
            f"{result['shares']['local model 10ms']}% of a 10ms local one and "
            f"{result['shares']['in-process 100us']}% of a 100us in-process one. To "
            f"reach 1% the rest of the request must finish in "
            f"{result['breakeven_1pc']}us",
        ),
        practice.Check(
            "FINDING: the shipped handler measures the handler, not instantiation",
            all([result["reply_has_us"] is True,
                 result["constructs_before_timer"] == 0,
                 result["timed_calls"] == 3]),
            f"agno_request_handler starts its timer after the agent exists -- "
            f"{result['constructs_before_timer']} constructions before it -- and times "
            f"{result['timed_calls']} calls inside it, the agent run plus two session "
            "appends. The microseconds in its reply are the whole request",
        ),
        practice.Check(
            "FINDING: instantiation is two attribute assignments",
            all([result["agent_fields"] == ["name", "fn"],
                 result["post_init"] is False,
                 len(result["agent_fields"]) == 2]),
            f"AgnoAgent is a dataclass with {len(result['agent_fields'])} fields "
            f"{result['agent_fields']} and no __post_init__ "
            f"({result['post_init']}), so constructing one does the work of a two-"
            "element tuple. A number that small measures how much setup a framework "
            "insists on, not how fast it runs",
        ),
        practice.Check(
            "FINDING: the session store is the part that scales with traffic",
            all([result["sessions"] == 100, result["resident"] == 1002,
                 result["trims"] == 0, result["quoted_kib"] == 3.75]),
            f"AgnoSession.append never trims ({result['trims']} pops), so 1000 turns "
            f"across {result['sessions']} sessions leave {result['resident']} strings "
            f"resident. The lesson quotes {result['quoted_kib']} KiB per agent; per "
            "session the toy has no bound, and that is the number a stateless backend "
            "has to answer for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
