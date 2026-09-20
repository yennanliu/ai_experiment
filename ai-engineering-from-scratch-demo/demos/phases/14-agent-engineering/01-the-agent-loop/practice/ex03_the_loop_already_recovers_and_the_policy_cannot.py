"""Exercise 3 — the loop already recovers and the policy cannot.

    Extend `ToyLLM` so it sometimes returns an `Action` with a malformed
    argument dict. Make the loop recover by feeding back an error observation.
    This is the shape of 2026 CRITIC-style correction (Lesson 5).

Reading of the exercise: feeding an error back as an observation is already
shipped -- `ToolRegistry.dispatch` catches `TypeError` and returns a string,
and `AgentLoop` stores it like any other observation. So "make the loop
recover" cannot mean the loop; it means the policy, and `ToyLLM.respond`
takes `history` and never reads it. The extension is therefore two policies
over one malformed script: one that ignores the feedback and one that does not.

**ANSWER: a malformed script and a critic that reads the observation back.**
**3** of the **6** planned calls carry a malformed argument dict. Without
recovery the loop survives all **6**, records **3** `error:` observations and
finishes with `missing:tax` -- an unstored key read back as a value. With
recovery the same script dispatches **9**, still **3** errors, and the final
read returns `18.0`.

**FINDING: `bad args` also reports faults the arguments did not cause.**
`dispatch` has one `except TypeError`, and it cannot tell a binding failure
from a `TypeError` raised inside the tool. `calculator(expression=...)` and
`calculator(expr=120)` both come back prefixed `error: bad args for
calculator` -- the first is the model's fault, the second is the tool's, and
the model is told to fix its arguments either way.

**FINDING: the shipped policy has no path to recovery.** `ToyLLM.respond`
names `history` in its signature and **0** times in its body, so the script
advances identically whatever the observations say. The critic retries **3**
times; `ToyLLM` retries **0**.

**FINDING: `error:` is produced at two layers and the history has one field.**
`calculator` returns `error: illegal character in expr` through a *successful*
dispatch, while an unknown tool returns `error: unknown tool` through a failed
one. Both land as `Turn(kind='action')` with the string in `observation`, so
telling a rejected call from an executed one that disliked its input means
string-matching a prefix.

Structure: `MalformedLLM` is the script, `CriticLLM` adds the one branch that
reads the last observation; the lesson's `AgentLoop` runs both unmodified.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "01-the-agent-loop"
STEPS = (                                     # (name, args as emitted, corrected args)
    ("kv_set", {"key": "base", "value": "120"}, None),
    ("calculator", {"expression": "120 * 0.15"}, {"expr": "120 * 0.15"}),
    ("kv_set", {"key": "rate", "value": "0.15"}, None),
    ("calculator", {"expr": 138}, {"expr": "120 + 18.0"}),
    ("kv_set", {"key": "tax"}, {"key": "tax", "value": "18.0"}),
    ("kv_get", {"key": "tax"}, None),
)
PROBES = {
    "unknown key": ("calculator", {"expression": "1+1"}),
    "inside the tool": ("calculator", {"expr": 120}),
    "missing argument": ("kv_set", {"key": "tax"}),
    "rejected input": ("calculator", {"expr": "__import__('os')"}),
    "unknown tool": ("no_such_tool", {}),
}


class MalformedLLM:
    """The script the exercise asks for: three of six calls do not bind."""

    recovers = False

    def __init__(self, steps):
        self.steps, self.cursor, self.pending, self.retries = list(steps), 0, None, 0

    def _retry(self, history):
        last = history[-1]
        failed = last.kind == "action" and (last.observation or "").startswith("error:")
        return self.pending if (self.recovers and self.pending and failed) else None

    def respond(self, history):
        retry = self._retry(history)
        if retry is not None:
            self.pending, self.retries = None, self.retries + 1
            return {"kind": "action", "thought": "the observation named the fault",
                    "action": retry[0], "args": retry[1]}
        if self.cursor >= len(self.steps):
            return {"kind": "finish", "content": "done"}
        name, args, fix = self.steps[self.cursor]
        self.cursor += 1
        self.pending = (name, fix) if fix else None
        return {"kind": "action", "thought": f"call {name}", "action": name, "args": args}


class CriticLLM(MalformedLLM):
    """CRITIC-shaped: one branch that reads the observation it just caused."""

    recovers = True


def registry(ref):
    tools, store = ref.ToolRegistry(), ref.KVStore()
    tools.register("calculator", ref.calculator)
    tools.register("kv_set", store.set)
    tools.register("kv_get", store.get)
    return tools


def drive(ref, llm):
    loop = ref.AgentLoop(llm=llm, tools=registry(ref), max_turns=20)
    loop.run("store the parts of the total and read the tax back")
    seen = [turn.observation for turn in loop.history if turn.kind == "action"]
    return {"dispatched": len(seen), "errors": [o for o in seen if o.startswith("error:")],
            "last": seen[-1], "retries": llm.retries,
            "error_kinds": sorted({t.kind for t in loop.history
                                   if (t.observation or "").startswith("error:")})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tools = registry(ref)
    probes = {label: tools.dispatch(ref.ToolCall(name, args))
              for label, (name, args) in PROBES.items()}
    body = inspect.getsource(ref.ToyLLM.respond).split("\n", 1)[1]
    return {
        "plain": drive(ref, MalformedLLM(STEPS)),
        "critic": drive(ref, CriticLLM(STEPS)),
        "malformed": sum(1 for *_, fix in STEPS if fix),
        "probes": probes,
        "bad_args": sorted(k for k, v in probes.items()
                           if v.startswith("error: bad args for")),
        "history_reads": body.count("history"),
    }


def verify(result):
    plain, critic = result["plain"], result["critic"]
    return [
        practice.Check(
            "ANSWER: three malformed calls, survived without recovery and fixed with it",
            all([result["malformed"] == 3, plain["dispatched"] == 6,
                 len(plain["errors"]) == 3, plain["last"] == "missing:tax",
                 critic["dispatched"] == 9, len(critic["errors"]) == 3,
                 critic["last"] == "18.0"]),
            f"{result['malformed']} of 6 planned calls are malformed. Without recovery the "
            f"loop dispatches {plain['dispatched']}, records {len(plain['errors'])} error "
            f"observations and reads back {plain['last']!r}; with recovery it dispatches "
            f"{critic['dispatched']} and reads back {critic['last']!r}",
        ),
        practice.Check(
            "FINDING: 'bad args' also reports faults the arguments did not cause",
            all([result["bad_args"] == ["inside the tool", "missing argument", "unknown key"],
                 "not iterable" in result["probes"]["inside the tool"],
                 "unexpected keyword" in result["probes"]["unknown key"]]),
            f"{len(result['bad_args'])} probes come back under one prefix: "
            f"{result['bad_args']}. 'unknown key' is a binding failure and 'inside the "
            "tool' is a TypeError the tool raised on a value that bound fine -- dispatch "
            "has one except TypeError and cannot tell them apart",
        ),
        practice.Check(
            "FINDING: the shipped policy has no path to recovery",
            all([result["history_reads"] == 0, plain["retries"] == 0,
                 critic["retries"] == 3]),
            f"ToyLLM.respond takes history and names it {result['history_reads']} times in "
            f"its body, so the script advances identically whatever comes back. The critic "
            f"retries {critic['retries']} times on the same script; the shipped policy "
            f"retries {plain['retries']}",
        ),
        practice.Check(
            "FINDING: 'error:' is produced at two layers and the history has one field",
            all([result["probes"]["rejected input"] == "error: illegal character in expr",
                 result["probes"]["unknown tool"].startswith("error: unknown tool"),
                 plain["error_kinds"] == ["action"]]),
            f"a rejected input returns {result['probes']['rejected input']!r} through a "
            f"successful dispatch and an unknown tool returns "
            f"{result['probes']['unknown tool']!r} through a failed one. Both land as "
            f"kind={plain['error_kinds']} with the string in observation, so the difference "
            "survives only as a prefix a reader has to match",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
