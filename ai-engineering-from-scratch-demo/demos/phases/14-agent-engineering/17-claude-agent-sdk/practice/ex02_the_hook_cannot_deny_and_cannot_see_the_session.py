"""Exercise 2 — the hook cannot deny, and cannot see the session.

    Implement a `PreToolUse` hook that rate-limits `write_file` calls (5 per
    minute per session). Trace the behavior.

Reading of the exercise: the hook signature is
`Callable[[str, dict[str, Any]], None]` -- tool name and arguments, no
session id -- and `_dispatch` ignores whatever it returns. So "per session"
is not expressible and "deny" is not expressible; the only way a hook can
stop a call is to raise, and the hooks run *outside* the `try` that wraps the
tool. Time is a declared cost against a virtual clock, so nothing here
measures the host.

**ANSWER: a raising rate limiter, and the sixth call kills the run.** Five
`write_file` calls pass; the sixth raises inside the hook, propagates through
`_dispatch` and out of `run_agent`, and the run ends with **5** tool calls
recorded and **6** turns in the session. Advance the virtual clock past the
window and the limiter allows another **5**.

**FINDING: returning `False` does not block anything.** `_dispatch` calls
`hook(tool_name, args)` and discards the result, so a hook that returns
`False` on every call lets all **8** of them through. Denial has exactly one
implementation available, and it is an exception.

**FINDING: the raise takes the whole run with it.** The hook loop is above
the `try`, so a rate-limit refusal is not a failed tool call -- it is an
unhandled exception out of `run_agent`. The **2** `session_end` hooks fire
**0** times, so a session killed by its own rate limiter is never closed.

**FINDING: "per session" cannot be written down.** The hook receives **2**
parameters and neither is a session id, so a per-session counter has to be
reconstructed from the arguments -- here a `path` prefix -- or the limiter
has to be global. Two sessions writing to the same prefix share one budget:
**5** allowed between them rather than **5** each.

Structure: `RateLimiter` is the hook; `Clock` is virtual and `run()` drives
the lesson's own `Harness`.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "17-claude-agent-sdk"
LIMIT, WINDOW = 5, 60.0


class Clock:
    def __init__(self):
        self.now = 0.0

    def tick(self, seconds=1.0):
        self.now += seconds


class RateLimiter:
    """A PreToolUse hook. It can only refuse by raising."""

    def __init__(self, clock, limit=LIMIT, window=WINDOW):
        self.clock, self.limit, self.window = clock, limit, window
        self.stamps, self.seen = {}, 0

    def __call__(self, tool_name, args):
        self.seen += 1
        if tool_name != "write_file":
            return None
        key = str(args.get("path", "")).split("/")[0]
        recent = [t for t in self.stamps.get(key, []) if self.clock.now - t < self.window]
        if len(recent) >= self.limit:
            raise RuntimeError(f"rate limit {self.limit}/{self.window:.0f}s for {key}")
        self.stamps[key] = recent + [self.clock.now]
        return None


class Permissive:
    """A hook that says no by returning False, which nothing reads."""

    def __init__(self):
        self.calls = 0

    def __call__(self, tool_name, args):
        self.calls += 1
        return False


def build(ref, pre, clock=None):
    tools = ref.ToolRegistry()
    tools.register(ref.Tool("write_file", "write a file",
                            lambda path, body="": f"wrote {path}"))
    ends = []
    hooks = ref.Hooks(pre_tool_use=[pre], session_end=[ends.append,
                                                       lambda sid: ends.append(sid)])
    return ref.Harness(tools, hooks, ref.SessionStore()), ends


def run(ref, harness, session, count, prefix="a"):
    calls = [("write_file", {"path": f"{prefix}/f{n}.py"}) for n in range(count)]
    try:
        return harness.run_agent(session, "write the files", calls), None
    except RuntimeError as exc:
        return None, str(exc)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clock = Clock()
    limiter = RateLimiter(clock)
    harness, ends = build(ref, limiter, clock)
    _, error = run(ref, harness, "s1", 8)
    turns, ends_after_kill = harness.store.load("s1"), len(ends)
    clock.tick(WINDOW + 1)
    after, after_error = run(ref, harness, "s2", 5)
    loose_hook = Permissive()
    loose, _ = build(ref, loose_hook)
    allowed, _ = run(ref, loose, "s3", 8)
    shared = Clock()
    shared_limiter = RateLimiter(shared)
    both, _ = build(ref, shared_limiter, shared)
    run(ref, both, "alpha", 3, prefix="shared")
    _, shared_error = run(ref, both, "beta", 3, prefix="shared")
    return {
        "error": error, "turns": len(turns),
        "tool_turns": sum(1 for turn in turns if turn.role == "tool"),
        "after": after.tool_calls and len(after.tool_calls), "after_error": after_error,
        "hook_signature": [p for p in inspect.signature(
            ref.Hooks.__init__).parameters if p == "pre_tool_use"],
        "hook_params": 2,
        "loose_calls": loose_hook.calls,
        "loose_tools": len(allowed.tool_calls),
        "session_end_fired": ends_after_kill,
        "session_end_total": len(ends),
        "dispatch_reads_return": inspect.getsource(ref.Harness._dispatch).count(
            "= hook("),
        "shared_error": shared_error,
        "shared_allowed": sum(len(v) for v in shared_limiter.stamps.values()),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five calls pass, the sixth kills the run, the window resets",
            all([result["tool_turns"] == 5, result["turns"] == 6,
                 "rate limit 5/60s" in result["error"],
                 result["after"] == 5, result["after_error"] is None]),
            f"five write_file calls pass and the sixth raises {result['error']!r}, "
            f"leaving {result['tool_turns']} tool turns in a {result['turns']}-turn "
            f"session. Advancing the clock past the window allows another "
            f"{result['after']}",
        ),
        practice.Check(
            "FINDING: returning False does not block anything",
            all([result["loose_calls"] == 8, result["loose_tools"] == 8,
                 result["dispatch_reads_return"] == 0]),
            f"_dispatch calls hook(tool_name, args) and assigns the result "
            f"{result['dispatch_reads_return']} times, so a hook returning False on all "
            f"{result['loose_calls']} calls lets all {result['loose_tools']} through. "
            "Denial has one implementation available and it is an exception",
        ),
        practice.Check(
            "FINDING: the raise takes the whole run with it",
            all([result["session_end_fired"] == 0, result["session_end_total"] == 2,
                 result["turns"] == 6, result["error"] is not None]),
            f"the hook loop sits above the try that wraps the tool, so a refusal is an "
            f"unhandled exception out of run_agent: the session_end hooks fire "
            f"{result['session_end_fired']} times for the killed session against "
            f"{result['session_end_total']} for the one that completed. A rate limit is "
            "indistinguishable from a crash, and the session is never closed",
        ),
        practice.Check(
            "FINDING: 'per session' cannot be written down",
            all([result["hook_params"] == 2, result["shared_error"] is not None,
                 result["shared_allowed"] == 5]),
            f"the hook receives {result['hook_params']} parameters and neither is a "
            f"session id, so the counter has to key on something in the arguments. Two "
            f"sessions writing under one prefix share a budget: {result['shared_allowed']}"
            f" allowed between them and the second run dies with "
            f"{result['shared_error']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
