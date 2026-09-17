"""Exercise 4 — the loop already ships, and it checks Answer before Action.

    **Build a ReAct loop**: Extend the pipeline with a calculator tool. When the
    model generates a math expression, execute it with Python's `eval()` (in a
    sandbox) and feed the result back. Measure if tool-grounded reasoning
    outperforms pure CoT.

Reading of the exercise: `react_solve` already is this loop -- calculator,
`eval`, observation fed back -- so the exercise is read as *audit the loop the
lesson shipped*, driven through the `client` parameter every solver takes. Four
scripted replies, each isolating one branch, plus the guard on `eval`.

**ANSWER: the tool never fires for a model that also states its answer.** The
loop tests `Answer:` before `Action:` and returns on the first hit, so a reply
holding both -- which is what a capable model writes -- ends the run in 1 call
with 0 observations. The arm labelled "tool-grounded" runs pure CoT.

**FINDING: when the tool does fire it returns a float.** `eval("48 + 48/2")` is
`72.0` under true division, so the observation reads `Observation: 72.0`. A
model that echoes its own tool output answers "72.0" against an expected "72",
which Exercise 1 showed is graded as wrong.

**FINDING: "(in a sandbox)" is not one.** `eval(expr, {"__builtins__": {}}, {})`
removes names, not capability: `().__class__.__mro__[1].__subclasses__()`
evaluates to roughly 600 classes under exactly that guard, and `9**9**9` is a
syntactically valid expression the guard cannot see. A calculator needs a
grammar, not an empty namespace.

**FINDING: a turn with neither Action nor Answer is a silent five-step no-op.**
Nothing is appended to the conversation, so the next iteration re-sends the same
messages; at temperature 0 the model repeats. After `max_steps` the fallback
runs `extract_answer` over the concatenated assistant text and returns the last
number it finds -- `'3'`, from the words "3 parts".

**ANSWER: the comparison the exercise asks for cannot be read off the return.**
All three exits -- tool-grounded, one-shot, exhausted -- return the same
`(answer, text)` pair, and the tool-grounded one's text is the final assistant
turn, which holds no observation. Only the client's call log separates them, and
the client is not returned.

Structure: `Stub` is the scripted client and records what it was sent; `ROUTES`
are the four replies, one per branch of the loop.
"""

from __future__ import annotations

import ast
import types

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "02-few-shot-cot"
GUARD = ({"__builtins__": {}}, {})
REACH = "().__class__.__mro__[1].__subclasses__()"
ROUTES = {
    "both": ["Thought: compute it.\nAction: calculate 48 + 48/2\nAnswer: 72"],
    "tool": ["Thought: compute it.\nAction: calculate 48 + 48/2", "Answer: 72."],
    "error": ["Thought: go.\nAction: calculate 1/0", "Answer: 5"],
    "noop": ["Thought: I am thinking about the 3 parts of this."],
}


class Stub:
    """A scripted `client` that records the turns it was sent."""

    def __init__(self, replies):
        self.replies, self.calls, self.sent = list(replies), 0, []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.sent.append(kwargs["messages"][-1]["content"])
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def route(ref, replies):
    stub = Stub(replies)
    answer, trace = ref.react_solve("Q?", stub, "model")
    observations = [line for line in stub.sent if line.startswith("Observation")]
    return {"answer": answer, "calls": stub.calls, "observations": observations,
            "trace_has_observation": "Observation" in trace, "trace": trace[:40]}


def sandbox():
    """What the lesson's guard still reaches, and what it cannot see at all."""
    reachable = len(eval(REACH, *GUARD))
    bomb = type(ast.parse("9**9**9", mode="eval").body).__name__
    return {"reachable": reachable, "bomb": bomb}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "advanced_prompting")
    runs = {name: route(ref, replies) for name, replies in ROUTES.items()}
    shapes = {name: (type(run["answer"]).__name__, type(run["trace"]).__name__)
              for name, run in runs.items()}
    return {"runs": runs, **sandbox(), "shapes": sorted(set(shapes.values())),
            "division": eval("48 + 48/2", *GUARD), "max_steps": 5}


def verify(result):
    runs = result["runs"]
    both, tool, noop = runs["both"], runs["tool"], runs["noop"]
    return [
        practice.Check(
            "ANSWER: a reply holding both lines never reaches the calculator",
            all([both["calls"] == 1, both["observations"] == [], both["answer"] == "72"]),
            f"the loop tests `Answer:` before `Action:` and returns on the first hit, so a "
            f"reply with a Thought, an Action and an Answer ends in {both['calls']} call "
            f"with {len(both['observations'])} observations, answer {both['answer']!r}. "
            "The arm labelled tool-grounded ran pure chain-of-thought",
        ),
        practice.Check(
            "FINDING: when the tool does fire it hands back a float",
            all([tool["observations"] == ["Observation: 72.0"], tool["answer"] == "72."]),
            f"`eval('48 + 48/2')` is {result['division']!r} under true division, so the "
            f"observation reads {tool['observations'][0]!r}. The run then answers "
            f"{tool['answer']!r} -- a float echoed back, or a full stop, either graded "
            "wrong against an expected '72' by the comparison Exercise 1 measured",
        ),
        practice.Check(
            "FINDING: the guard removes names, not capability",
            all([result["reachable"] > 500, result["bomb"] == "BinOp"]),
            f"under `eval(expr, {{'__builtins__': {{}}}}, {{}})` -- the lesson's guard "
            f"verbatim -- `{REACH}` evaluates to {result['reachable']} classes, and "
            f"`9**9**9` parses as a valid {result['bomb']} that the guard has no view of. "
            "A calculator needs a restricted grammar, not an emptied namespace",
        ),
        practice.Check(
            "FINDING: a turn with neither Action nor Answer is a silent five-step no-op",
            all([noop["calls"] == result["max_steps"], noop["answer"] == "3"]),
            f"nothing is appended to the conversation, so the loop re-sends the same "
            f"messages {noop['calls']} times and the model repeats. The fallback then runs "
            f"`extract_answer` over the concatenated assistant text and returns "
            f"{noop['answer']!r} -- the last number in the words 'the 3 parts of this'",
        ),
        practice.Check(
            "ANSWER: the three exits are indistinguishable from the return value",
            all([len(result["shapes"]) == 1, not tool["trace_has_observation"]]),
            f"tool-grounded, one-shot and exhausted all return {result['shapes'][0]}, and "
            f"the tool-grounded trace is the final assistant turn ({tool['trace']!r}), "
            "which holds no observation. Whether the calculator ran lives only in the "
            "client's call log, and the client is not returned",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
