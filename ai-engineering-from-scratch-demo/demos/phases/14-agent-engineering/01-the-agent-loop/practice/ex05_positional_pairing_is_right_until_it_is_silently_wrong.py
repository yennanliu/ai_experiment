"""Exercise 5 — positional pairing is right until it is silently wrong.

    Add a `tool_use_id` correlator like the Anthropic schema so parallel tool
    calls can return out of order. Why do Anthropic, OpenAI, and Bedrock all
    require it?

Reading of the exercise: the shipped loop cannot exhibit the bug the id
prevents. `AgentLoop.run` dispatches and appends in one breath, so a call and
its observation are paired by construction and no order exists to get wrong.
The correlator is therefore built against a batch that returns out of order,
and "why do all three require it" is answered by measuring what the
alternative -- pairing by position -- actually does when the order changes.

**ANSWER: a `tool_use_id` correlator over three calls returned in reverse.**
Keyed by id, **3** of **3** observations land on the call that produced them.
Keyed by position, **1** of **3** does. Both transcripts are well-formed:
**3** action turns, **3** non-empty observations, **0** errors.

**FINDING: the mis-pairing is invisible to every check the loop can make.**
All three tools return `str`, and here all three results are numeric strings,
so a type check catches **0** of the **2** swaps and a "does it look like a
number" check catches **0**. The only evidence that `120 * 0.15` returned
`138.0` is the id that was dropped.

**FINDING: the id has to be minted before the tool runs.**
`ToolRegistry.dispatch` takes a `ToolCall` and returns a bare `str` -- no
channel for an identifier on the way out -- and `ToolCall` carries **2**
fields, neither an id. So the correlator cannot be added at the tool layer;
it belongs on the request, which is exactly where Anthropic, OpenAI and
Bedrock all put it.

**FINDING: the shipped loop hides the need for it.** The module names
`tool_use_id` **0** times and the sequential run pairs **3** of **3**
correctly with no correlator at all. One call per turn makes position a
correct key, and it stays correct right up to the first turn that carries two.

Structure: `dispatch_batch` is the parallel edge; `by_id` and `by_position`
are the two correlators, scored against the same returned-in-reverse batch.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "01-the-agent-loop"
BATCH = (
    ("toolu_01", "calculator", {"expr": "120 * 0.15"}),
    ("toolu_02", "kv_get", {"key": "base"}),
    ("toolu_03", "calculator", {"expr": "120 + 18.0"}),
)


class BatchLLM:
    """One batch, emitted as the shipped loop wants it: one call per turn."""

    def __init__(self, batch):
        self.batch, self.cursor = list(batch), 0

    def respond(self, history):
        if self.cursor >= len(self.batch):
            return {"kind": "finish", "content": "done"}
        _, name, args = self.batch[self.cursor]
        self.cursor += 1
        return {"kind": "action", "thought": f"call {name}", "action": name, "args": args}


def registry(ref):
    tools, store = ref.ToolRegistry(), ref.KVStore()
    store.set("base", "120")
    tools.register("calculator", ref.calculator)
    tools.register("kv_get", store.get)
    return tools


def dispatch_batch(ref, tools, batch):
    """What a provider returns: result blocks, each carrying the id it answers."""
    return [(call_id, tools.dispatch(ref.ToolCall(name, args)))
            for call_id, name, args in batch]


def by_id(batch, returned):
    index = dict(returned)
    return [(call_id, index[call_id]) for call_id, _, _ in batch]


def by_position(batch, returned):
    return [(call_id, result) for (call_id, _, _), (_, result) in zip(batch, returned)]


def turns(ref, batch, paired):
    named = {call_id: (name, args) for call_id, name, args in batch}
    return [ref.Turn(kind="action", content=named[call_id][0],
                     tool_call=ref.ToolCall(*named[call_id]), observation=result)
            for call_id, result in paired]


def scored(truth, paired):
    return sum(1 for call_id, result in paired if truth[call_id] == result)


def looks_numeric(text):
    return text.replace(".", "", 1).isdigit()


def shape(ref, positional, truth):
    """What the transcript looks like once the wrong observations are attached."""
    rows = turns(ref, BATCH, positional)
    seen = [turn.observation for turn in rows]
    wrong = [result for call_id, result in positional if truth[call_id] != result]
    return {"action_turns": len(rows), "observations": seen,
            "errors": [o for o in seen if o.startswith("error:")],
            "type_mismatches": sum(1 for o in seen if not isinstance(o, str)),
            "wrong_but_numeric": sum(1 for o in wrong if looks_numeric(o))}


def sequential(ref, truth):
    """The shipped loop, one call per turn: position is a correct key here."""
    loop = ref.AgentLoop(llm=BatchLLM(BATCH), tools=registry(ref), max_turns=8)
    loop.run("compute the tax and the total, and read the base back")
    seen = [turn.observation for turn in loop.history if turn.kind == "action"]
    return sum(1 for got, (call_id, _, _) in zip(seen, BATCH) if got == truth[call_id])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    returned = dispatch_batch(ref, registry(ref), BATCH)
    truth, out_of_order = dict(returned), list(reversed(returned))
    positional = by_position(BATCH, out_of_order)
    result = {
        "keyed_correct": scored(truth, by_id(BATCH, out_of_order)),
        "positional_correct": scored(truth, positional),
        "size": len(BATCH), "results": [value for _, value in returned],
        "dispatch_returns": inspect.signature(ref.ToolRegistry.dispatch).return_annotation,
        "toolcall_fields": list(ref.ToolCall.__dataclass_fields__),
        "id_fields": [f for f in ref.Turn.__dataclass_fields__ if "id" in f],
        "id_mentions": inspect.getsource(ref).count("tool_use_id"),
        "sequential_correct": sequential(ref, truth),
    }
    result.update(shape(ref, positional, truth))
    return result


def verify(result):
    return [
        practice.Check(
            "ANSWER: keyed by id 3 of 3 land, keyed by position 1 of 3",
            all([result["keyed_correct"] == 3, result["positional_correct"] == 1,
                 result["size"] == 3, result["action_turns"] == 3,
                 all(result["observations"]), result["errors"] == []]),
            f"over {result['size']} calls returned in reverse, the id correlator pairs "
            f"{result['keyed_correct']} correctly and the positional one "
            f"{result['positional_correct']}, on a transcript still holding "
            f"{result['action_turns']} action turns and {len(result['errors'])} errors",
        ),
        practice.Check(
            "FINDING: the mis-pairing is invisible to every check the loop can make",
            all([result["type_mismatches"] == 0, result["wrong_but_numeric"] == 2,
                 result["results"] == ["18.0", "120", "138.0"]]),
            f"the batch returns {result['results']} -- all str, all numeric -- so a type "
            f"check catches {result['type_mismatches']} of the 2 swaps and a numeric check "
            f"catches {2 - result['wrong_but_numeric']}. The id was the only evidence",
        ),
        practice.Check(
            "FINDING: the id has to be minted before the tool runs",
            all([result["dispatch_returns"] == "str",
                 result["toolcall_fields"] == ["name", "args"], result["id_fields"] == []]),
            f"dispatch returns a bare {result['dispatch_returns']}, so there is no channel "
            f"for an identifier on the way out; ToolCall carries "
            f"{result['toolcall_fields']} and Turn carries {len(result['id_fields'])} id "
            "fields. The correlator cannot live at the tool layer -- it lives on the "
            "request, which is where all three provider schemas put it",
        ),
        practice.Check(
            "FINDING: one call per turn makes position a correct key, until it is not",
            all([result["id_mentions"] == 0, result["sequential_correct"] == 3]),
            f"the lesson's module names tool_use_id {result['id_mentions']} times and the "
            f"sequential run pairs {result['sequential_correct']} of {result['size']} with "
            "no correlator at all, because dispatching and appending happen in one "
            "statement. Position is a correct key right up to the first turn with two calls",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
