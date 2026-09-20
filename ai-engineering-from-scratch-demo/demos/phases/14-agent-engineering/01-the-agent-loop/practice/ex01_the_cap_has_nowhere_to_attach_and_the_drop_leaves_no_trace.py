"""Exercise 1 — the cap has nowhere to attach and the drop leaves no trace.

    Add a `max_tool_calls_per_turn` cap. What breaks if the model issues three
    calls but you only execute the first two?

Reading of the exercise: the shipped `AgentLoop` reads exactly one `action`
per reply, so "three calls in one turn" is not a state it can represent. The
cap therefore cannot live in the loop; it lives in the adapter that flattens a
provider's parallel calls into the shipped reply shape. Putting it there is
what makes the breakage measurable, because the dropped calls are discarded
*before* the loop -- and the loop is the only thing that writes history.

**ANSWER: a `max_tool_calls_per_turn` cap in the flattening adapter.** With
`cap=2` over a script whose first turn issues **3** calls, **4** of the **5**
requested calls are dispatched and the history holds **4** action turns. The
dropped `kv_set(key='tax')` appears **0** times anywhere in the transcript.

**FINDING: the dropped call is read back as data, not as an error.** The
later `kv_get(key='tax')` returns `missing:tax`, and `ToolRegistry.dispatch`
only marks failures by prefixing `error:`. So **0** of the **4** observations
look like errors while **1** of them is wrong. The agent's next thought is
conditioned on a string that reads like a fact.

**FINDING: `max_turns` stops being a tool-call budget.** In the shipped loop a
turn is a call, so `max_turns=10` caps dispatches at **10**. Once a turn
carries `cap` calls the same budget admits `10 * cap`, and the budget itself
starts truncating turns mid-flight: uncapped with `max_turns=4` the run
dispatches **4** of **5** and ends `budget exhausted`, which is the exercise's
breakage arriving from the other end.

**FINDING: there is nowhere in the shipped types to put the cap.**
`ToolCall` has **2** fields, `Turn` has **4**, and a reply's `action` is a
`str`. No shipped type holds a list of calls, so no shipped type can hold a
count of them either.

Structure: `FlatteningLLM` is the adapter under test; `run()` drives the
lesson's own `AgentLoop` and `ToolRegistry` against it, unmodified.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "01-the-agent-loop"
SCRIPT = (
    {"thought": "record the three components",
     "calls": (("kv_set", {"key": "base", "value": "120"}),
               ("kv_set", {"key": "rate", "value": "0.15"}),
               ("kv_set", {"key": "tax", "value": "18.0"}))},
    {"thought": "read the parts of the total back",
     "calls": (("kv_get", {"key": "base"}),
               ("kv_get", {"key": "tax"}))},
)
REQUESTED = sum(len(entry["calls"]) for entry in SCRIPT)


class FlatteningLLM:
    """Where a provider's parallel calls become the loop's one-action replies."""

    def __init__(self, script, cap):
        self.queue = [(entry["thought"], name, args) for entry in script
                      for name, args in entry["calls"][:cap]]
        self.dropped = [(name, args) for entry in script
                        for name, args in entry["calls"][cap:]]
        self.cursor = 0

    def respond(self, history):
        if self.cursor >= len(self.queue):
            return {"kind": "finish", "content": "done"}
        thought, name, args = self.queue[self.cursor]
        self.cursor += 1
        return {"kind": "action", "thought": thought, "action": name, "args": args}


def run(ref, cap, max_turns=12):
    tools, store = ref.ToolRegistry(), ref.KVStore()
    tools.register("kv_set", store.set)
    tools.register("kv_get", store.get)
    llm = FlatteningLLM(SCRIPT, cap)
    loop = ref.AgentLoop(llm=llm, tools=tools, max_turns=max_turns)
    final = loop.run("record the three components, then read them back")
    return loop, llm, final


def observations(history):
    return [turn.observation for turn in history if turn.kind == "action"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    capped, llm, _ = run(ref, cap=2)
    squeezed, _, squeezed_final = run(ref, cap=3, max_turns=4)
    seen = observations(capped.history)
    return {
        "requested": REQUESTED, "dispatched": len(seen), "dropped": llm.dropped,
        "action_turns": len(seen),
        "trace_mentions_dropped": sum(f"{t.content}{t.tool_call}{t.observation}"
                                      .count("'tax', 'value'") for t in capped.history),
        "observations": seen,
        "errors": [text for text in seen if text.startswith("error:")],
        "tax_read": seen[-1],
        "budget_dispatched": len(observations(squeezed.history)),
        "budget_final": squeezed_final,
        "toolcall_fields": list(ref.ToolCall.__dataclass_fields__),
        "turn_fields": list(ref.Turn.__dataclass_fields__),
        "action_type": type(SCRIPT[0]["calls"][0][0]).__name__,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the cap lives in the adapter, and 4 of 5 calls reach the loop",
            all([result["requested"] == 5, result["dispatched"] == 4,
                 result["action_turns"] == 4, len(result["dropped"]) == 1,
                 result["dropped"][0][0] == "kv_set",
                 result["trace_mentions_dropped"] == 0]),
            f"{result['requested']} calls requested, {result['dispatched']} dispatched, "
            f"{result['action_turns']} action turns, and "
            f"{result['trace_mentions_dropped']} mentions of the dropped "
            f"{result['dropped'][0][0]} -- the loop cannot record a call it never saw",
        ),
        practice.Check(
            "FINDING: the dropped call comes back as data, not as an error",
            all([result["tax_read"] == "missing:tax", result["errors"] == [],
                 not result["tax_read"].startswith("error:")]),
            f"the read of the dropped key returns {result['tax_read']!r}; dispatch marks "
            f"failures with an 'error:' prefix, so {len(result['errors'])} of "
            f"{len(result['observations'])} observations look like errors while 1 is wrong",
        ),
        practice.Check(
            "FINDING: max_turns stops being a tool-call budget",
            all([result["budget_dispatched"] == 4,
                 result["budget_final"] == "budget exhausted",
                 result["budget_dispatched"] < result["requested"]]),
            f"uncapped under max_turns=4 the run dispatches {result['budget_dispatched']} "
            f"of {result['requested']} and ends {result['budget_final']!r}. A turn was a "
            "call; with a cap of c the same budget admits 10*c and truncates mid-turn",
        ),
        practice.Check(
            "FINDING: no shipped type holds a list of calls",
            all([result["toolcall_fields"] == ["name", "args"],
                 result["turn_fields"] == ["kind", "content", "tool_call", "observation"],
                 result["action_type"] == "str"]),
            f"ToolCall carries {result['toolcall_fields']}, Turn carries "
            f"{result['turn_fields']}, and a reply's action is a {result['action_type']}. "
            "Nothing shipped holds a list of calls, so the cap is an adapter setting",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
