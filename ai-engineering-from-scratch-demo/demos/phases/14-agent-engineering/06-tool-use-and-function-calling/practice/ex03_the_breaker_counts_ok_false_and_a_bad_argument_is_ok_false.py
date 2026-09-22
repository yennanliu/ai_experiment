"""Exercise 3 — the breaker counts ok=False, and a bad argument is ok=False.

    Add a per-tool timeout and a circuit breaker (refuse the tool for 60s
    after 3 consecutive failures). What does this change about how the model
    recovers?

Reading of the exercise: `ToolDef` already carries `timeout_s`, and nothing
reads it, so the timeout is added by enforcing a field that already exists.
Duration is supplied per call as a declared cost against a virtual clock
rather than measured, so the result does not depend on the machine it runs
on. The recovery question is then about what the model can tell apart, which
is decided by `ToolResult`.

**ANSWER: a timeout and a breaker over the lesson's registry.** A tool with
`timeout_s=1.0` called at cost **3.0** returns a timeout error; **3**
consecutive timeouts open the breaker, and the **4th** call is refused
without reaching the tool: **0** executions across **4** calls. Advance the
clock **60** seconds and the next call runs again.

**FINDING: `timeout_s` is declared and never read.** `ToolDef` has **5**
fields including `timeout_s`, and the source of `dispatch` mentions it **0**
times. A field on the public type is a promise; this one had no enforcement
anywhere in the module.

**FINDING: the breaker fires on failures the tool never saw.** Consecutive
failures are counted from `ok=False`, and `ok=False` also covers validation
errors. **3** calls missing a required argument execute the tool **0** times
and open the breaker anyway -- after which a *correct* call is refused. The
model's own malformed output takes the tool offline.

**FINDING: recovery gets a third failure kind and no field to carry it.**
Validation errors, timeouts and an open circuit are **3** different
instructions -- fix the arguments, retry smaller, wait -- and all **3**
arrive as `ok=False` with free text in `content`. `ToolResult` has **3**
fields and none of them is a reason code, so a model that responds to failure
by editing arguments will keep editing them at an open circuit.

Structure: `Guarded` wraps the lesson's `ToolRegistry`; `Clock` is virtual,
so nothing here measures wall time.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "06-tool-use-and-function-calling"
SCHEMA = {"type": "object", "required": ["a", "b"],
          "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
COOLDOWN, THRESHOLD, TIMEOUT = 60.0, 3, 1.0


class Clock:
    """Virtual time: a declared cost per call, so no result depends on the host."""

    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds


class Guarded:
    """The lesson's registry plus the two things the exercise asks for."""

    def __init__(self, ref, inner, clock):
        self.ref, self.inner, self.clock = ref, inner, clock
        self.failures, self.open_until, self.executions = {}, {}, 0

    def dispatch(self, call, cost=0.0):
        if self.clock.now < self.open_until.get(call.name, 0.0):
            wait = self.open_until[call.name] - self.clock.now
            return self._record(call, False, f"circuit open for {call.name}, "
                                             f"retry after {wait:.0f}s")
        limit = self.inner._tools[call.name].timeout_s
        if cost > limit:
            self.clock.advance(limit)
            return self._record(call, False, f"timeout after {limit}s")
        before = self.inner.dispatch(call)
        self.executions += before.ok
        self.clock.advance(cost)
        return self._record(call, before.ok, before.content)

    def _record(self, call, ok, content):
        count = 0 if ok else self.failures.get(call.name, 0) + 1
        self.failures[call.name] = count
        if count >= THRESHOLD:
            self.open_until[call.name] = self.clock.now + COOLDOWN
        return self.ref.ToolResult(call.tool_use_id, ok, content)


def build(ref):
    clock, inner = Clock(), ref.ToolRegistry()
    inner.register(ref.ToolDef(name="add", description="Add two integers.",
                               input_schema=SCHEMA, executor=ref.add, timeout_s=TIMEOUT))
    return Guarded(ref, inner, clock), clock


def drive(ref, guard, clock, args, cost, calls=4):
    """`calls` attempts, then one more after the cooldown has expired."""
    rows = [guard.dispatch(ref.ToolCall(f"u{i}", "add", dict(args)), cost)
            for i in range(calls)]
    during = guard.executions
    clock.advance(COOLDOWN)
    rows.append(guard.dispatch(ref.ToolCall("after", "add", {"a": 1, "b": 2}), 0.1))
    return rows, during


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    slow_guard, slow_clock = build(ref)
    slow, slow_runs = drive(ref, slow_guard, slow_clock, {"a": 1, "b": 2}, cost=3.0)
    bad_guard, bad_clock = build(ref)
    bad, bad_runs = drive(ref, bad_guard, bad_clock, {"a": 1}, cost=0.1)
    kinds = [slow[0].content, bad[0].content, slow[3].content]
    return {
        "slow_contents": [row.content for row in slow],
        "slow_executions": slow_runs,
        "recovered": (slow[4].ok, slow[4].content),
        "bad_executions": bad_runs,
        "bad_open": bad[3].content.startswith("circuit open"),
        "bad_recovered": bad[4].ok,
        "tooldef_fields": list(ref.ToolDef.__dataclass_fields__),
        "dispatch_mentions": inspect.getsource(ref.ToolRegistry.dispatch).count("timeout"),
        "result_fields": list(ref.ToolResult.__dataclass_fields__),
        "kinds": kinds, "kind_flags": {slow[0].ok, bad[0].ok, slow[3].ok},
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three timeouts open the breaker, and the fourth call never runs",
            all([result["slow_executions"] == 0,
                 all(row.startswith("timeout after") for row in result["slow_contents"][:3]),
                 result["slow_contents"][3].startswith("circuit open"),
                 result["recovered"] == (True, "3")]),
            f"a tool with timeout_s=1.0 called at cost 3.0 returns "
            f"{result['slow_contents'][0]!r} three times, then "
            f"{result['slow_contents'][3]!r} on the fourth -- the tool body ran "
            f"{result['slow_executions']} times. After 60 virtual seconds the next call "
            f"returns {result['recovered'][1]!r}",
        ),
        practice.Check(
            "FINDING: timeout_s is declared and never read",
            all([result["tooldef_fields"] == ["name", "description", "input_schema",
                                              "executor", "timeout_s"],
                 result["dispatch_mentions"] == 0]),
            f"ToolDef carries {result['tooldef_fields']} and the source of dispatch "
            f"mentions timeout {result['dispatch_mentions']} times. The field was a "
            "promise on the public type with no enforcement anywhere in the module",
        ),
        practice.Check(
            "FINDING: the breaker fires on failures the tool never saw",
            all([result["bad_executions"] == 0, result["bad_open"] is True,
                 result["bad_recovered"] is True]),
            f"three calls missing a required argument execute the tool "
            f"{result['bad_executions']} times and open the breaker anyway "
            f"({result['bad_open']}), so a correct call is refused until the cooldown "
            "expires. The model's own malformed output takes a healthy tool offline",
        ),
        practice.Check(
            "FINDING: three failure kinds, one boolean and one free-text field",
            all([len(set(result["kinds"])) == 3, result["kind_flags"] == {False},
                 result["result_fields"] == ["tool_use_id", "ok", "content"]]),
            f"the three kinds are {result['kinds']} -- fix the arguments, retry smaller, "
            f"wait -- and all arrive as ok in {result['kind_flags']} with the difference "
            f"only in text. ToolResult carries {result['result_fields']}, so a model that "
            "answers failure by editing arguments keeps editing them at an open circuit",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
