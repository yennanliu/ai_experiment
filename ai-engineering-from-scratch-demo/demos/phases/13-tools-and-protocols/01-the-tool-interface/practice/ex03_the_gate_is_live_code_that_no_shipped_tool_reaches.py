"""Exercise 3 — the gate is live code that no shipped tool reaches.

    Classify each tool in the harness as pure or consequential. Add a
    `consequential: true` flag to the registry entries that need it, and change
    the loop to print a "would confirm with user" line whenever a consequential
    tool is chosen. This is the shape of the confirmation gate every production
    host needs.

Reading of the exercise: the classification is done first, and it decides how
much of the rest there is to do -- none of the three tools needs the flag, and
the loop already prints the line. So the exercise's two build steps are already
in the lesson, and what is worth measuring is whether they ever run. They do
not. The flag is flipped on a live tool here to show the branch is reachable at
all, then restored (D5: the lesson is imported, not edited).

**ANSWER: all three are pure, and none needs the flag.** `add` is a function of
its arguments, `get_weather` reads a literal dict, and neither writes anything.
So `consequential` stays **false** on all three, and the count of registry
entries that need it is **0**.

**FINDING: `get_time` is not consequential, and it is not pure either.** It
calls `datetime.now`, so it reads state outside its arguments and cannot be
replayed -- while "The trust split" lists `get_current_time` under "Pure.
Read-only, deterministic, no side effects". The exercise's dichotomy has no cell
for it: "pure or consequential"
collapses two independent axes -- *reads* external state and *mutates* external
state -- into one, and `get_time` is the tool that reads without mutating. The
flag that matters for a confirmation gate is the mutation axis; the flag that
matters for caching and retry is the other one, and the lesson has neither.

**FINDING: the gate is live code that nothing reaches.** Running the lesson's
own four demo queries prints the GATE line **0** times. Flipping
`get_weather.consequential` to true and re-running the same query prints it
**1** time, so the branch works -- it is guarded by a flag no shipped tool sets.

**FINDING: and the circuit breaker is unreachable too.** `MAX_TURNS` is **5**,
but the deepest turn any demo query reaches is **2**: `fake_decide` returns
`content` as soon as the last history entry has role `tool`, so the loop always
exits on its second pass. "LOOP TERMINATED" prints **0** times. Both of the
loop's safety features are written, and neither executes.

Structure: `gate_lines` runs one query through the lesson's loop and counts GATE
lines, `deepest_turn` reads the turn numbers back out of the same transcript,
and `with_flag` flips one tool's flag around a call and restores it.
"""

from __future__ import annotations

import contextlib
import inspect
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "01-the-tool-interface"
DEMO_QUERIES = ("please add 7 and 35", "what time is it?",
                "tell me the weather in Bengaluru", "write me a haiku about tea")
WEATHER_QUERY = "tell me the weather in Bengaluru"
PROBE_ARGS = {"add": {"a": 1, "b": 2}, "get_time": {}, "get_weather": {"city": "Tokyo"}}


def transcript(ref, query):
    """The lesson's own loop output for one query."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.run_loop(query)
    return buffer.getvalue()


def gate_lines(ref, query):
    return transcript(ref, query).count("GATE")


def deepest_turn(ref, queries=DEMO_QUERIES):
    """The highest TURN number the loop reaches across the demo queries."""
    numbers = [int(line.split()[1]) for query in queries
               for line in transcript(ref, query).splitlines() if line.startswith("TURN")]
    return max(numbers)


def with_flag(ref, name, query):
    """GATE lines for `query` while `name` is marked consequential, flag restored after."""
    tool = {t.name: t for t in ref.REGISTRY}[name]
    tool.consequential = True
    try:
        return gate_lines(ref, query)
    finally:
        tool.consequential = False


def reads_clock(ref):
    return "datetime.now" in inspect.getsource(ref.tool_get_time)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    flags = {tool.name: tool.consequential for tool in ref.REGISTRY}
    replay = {tool.name: tool.executor(dict(PROBE_ARGS[tool.name]))
              == tool.executor(dict(PROBE_ARGS[tool.name])) for tool in ref.REGISTRY}
    shipped_gates = sum(gate_lines(ref, query) for query in DEMO_QUERIES)
    return {
        "flags": flags, "need_flag": sum(flags.values()),
        "replayable": replay,
        "reads_clock": reads_clock(ref),
        "shipped_gates": shipped_gates,
        "flipped_gates": with_flag(ref, "get_weather", WEATHER_QUERY),
        "restored": {t.name: t.consequential for t in ref.REGISTRY}["get_weather"],
        "max_turns": ref.MAX_TURNS,
        "deepest_turn": deepest_turn(ref),
        "terminated": sum(transcript(ref, query).count("LOOP TERMINATED")
                          for query in DEMO_QUERIES),
        "queries": len(DEMO_QUERIES),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: all three are pure, and none needs the flag",
            all([result["flags"] == {"add": False, "get_time": False, "get_weather": False},
                 result["need_flag"] == 0, result["replayable"]["add"],
                 result["replayable"]["get_weather"]]),
            f"add is a function of its arguments and get_weather reads a literal dict; "
            f"neither writes anything, and both replay identically. consequential stays "
            f"{result['flags']}, so {result['need_flag']} registry entries need the flag",
        ),
        practice.Check(
            "FINDING: get_time is not consequential, and it is not pure either",
            all([result["reads_clock"], not result["flags"]["get_time"]]),
            "tool_get_time calls datetime.now, so it reads state outside its arguments and "
            "cannot be replayed -- while 'The trust split' lists get_current_time under "
            "'Pure. Read-only, deterministic, no side effects'. 'Pure or consequential' "
            "collapses two independent axes -- reads external state, and mutates it -- into "
            "one, and get_time is the tool that reads without mutating. A confirmation gate "
            "keys off the mutation axis; caching and retry key off the other one, and the "
            "lesson has neither",
        ),
        practice.Check(
            "FINDING: the gate is live code that nothing reaches",
            all([result["shipped_gates"] == 0, result["flipped_gates"] == 1,
                 not result["restored"]]),
            f"the lesson's own {result['queries']} demo queries print the GATE line "
            f"{result['shipped_gates']} times. Flipping get_weather.consequential and "
            f"re-running the same query prints it {result['flipped_gates']} time, so the "
            "branch works -- it is guarded by a flag no shipped tool sets",
        ),
        practice.Check(
            "FINDING: and the circuit breaker is unreachable too",
            all([result["max_turns"] == 5, result["deepest_turn"] == 2,
                 result["terminated"] == 0]),
            f"MAX_TURNS is {result['max_turns']}, but the deepest turn any demo query reaches "
            f"is {result['deepest_turn']}: fake_decide returns content as soon as the last "
            f"history entry has role tool, so the loop always exits on its second pass. "
            f"'LOOP TERMINATED' prints {result['terminated']} times. Both of the loop's "
            "safety features are written, and neither executes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
