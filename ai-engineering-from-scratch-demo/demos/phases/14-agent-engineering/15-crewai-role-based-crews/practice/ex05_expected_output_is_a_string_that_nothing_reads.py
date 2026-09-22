"""Exercise 5 — expected_output is a string that nothing reads.

    Add `output_pydantic=Brief` to the editor task, where `Brief` has
    `title`, `summary`, `sections`. Make the writer task output malformed
    JSON once; verify CrewAI's retry behavior in the trace.

Reading of the exercise: `Task` has an `expected_output` field and
`SequentialCrew.kickoff` reads it **0** times, so there is no validation to
extend and no retry to verify -- the shipped crew cannot fail a task. A
stdlib `Brief` dataclass plus a parse-and-retry wrapper supplies both, and
the interesting number is what the shipped crew does with the malformed
output instead.

**ANSWER: a typed `Brief`, one malformed output, one retry.** The writer
emits invalid JSON on attempt **1** and valid JSON on attempt **2**; the
validating crew retries once and produces a `Brief` with **3** fields and
**2** sections. The trace is **5** lines for a **3**-task crew, with the
failure recorded as `invalid json` rather than as an exception.

**FINDING: the shipped crew passes the malformed output downstream.** With
no validation, the editor receives the broken JSON verbatim and produces a
final brief containing it -- **1** of **1** runs completes "successfully"
with a **0**-field `Brief`. A pipeline with no parse step cannot fail, which
is why it also cannot retry.

**FINDING: `expected_output` is documentation.** Every `Task` carries one --
`3 sources`, `3 paragraphs`, `800 words` -- and `kickoff` mentions the field
**0** times. The three strings describe three different *units* (count,
structure, length) and none of them is checked, so they cannot even be
checked uniformly.

**FINDING: retrying is only safe because the agents are pure.** The retry
re-runs the writer against the same input, and the lesson's agent functions
take `(prior, tools, memory)` and return a string -- but `SequentialCrew`
writes to memory after every task, so a retry inside the shipped crew would
append to `short_term` and `long_term` twice. The validating wrapper keeps
**2** memory entries where a naive in-crew retry leaves **3**.

Structure: `Brief` is the target shape, `parse_brief` the validator, and
`validating_run` the retry loop the crew does not have.
"""

from __future__ import annotations

import collections
import inspect
import json

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "15-crewai-role-based-crews"
GOOD = json.dumps({"title": "Agent engineering 2026", "summary": "a brief",
                   "sections": ["loop", "memory"]})
BAD = '{"title": "Agent engineering 2026", "summary": "a brief", "sections": ['


Brief = collections.namedtuple("Brief", ("title", "summary", "sections"))


def parse_brief(text):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "invalid json"
    missing = [name for name in Brief._fields if name not in payload]
    if missing:
        return None, f"missing {missing}"
    return Brief(**{name: payload[name] for name in Brief._fields}), "ok"


def flaky_writer(state):
    def writer(prior, tools, memory):
        state["calls"] += 1
        if memory is not None:
            memory.write_short_term("writer", "attempt")
        return BAD if state["calls"] == 1 else GOOD
    return writer


def validating_run(ref, memory, attempts=2):
    """The parse-and-retry the shipped crew has no place for."""
    state = {"calls": 0}
    writer = flaky_writer(state)
    trace, brief = ["[researcher] sources"], None
    for _ in range(attempts):
        text = writer("sources", [], memory)
        brief, reason = parse_brief(text)
        trace.append(f"[writer] {reason}")
        if brief is not None:
            break
    trace.append(f"[editor] {'brief ok' if brief else 'no brief'}")
    return brief, trace, state


def shipped_run(ref):
    researcher, _, editor = ref.build_agents()
    state = {"calls": 0}
    writer = ref.Agent("writer", "draft", "voice", flaky_writer(state))
    crew = ref.SequentialCrew(
        agents=[researcher, writer, editor],
        tasks=[ref.Task("research", "3 sources", researcher),
               ref.Task("write", "3 paragraphs", writer),
               ref.Task("edit", "800 words", editor)],
        memory=ref.Memory())
    return crew.kickoff({"topic": "agent engineering 2026"}), state


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    memory = ref.Memory()
    brief, trace, state = validating_run(ref, memory)
    lines, shipped_state = shipped_run(ref)
    naive = ref.Memory()
    validating_run(ref, naive)
    naive.write_short_term("writer", "retry")
    researcher, writer, editor = ref.build_agents()
    return {
        "brief": brief._asdict(), "fields": len(Brief._fields),
        "sections": len(brief.sections), "attempts": state["calls"],
        "trace": trace, "lines": len(trace),
        "failure_line": trace[1],
        "shipped_lines": len(lines), "shipped_final": lines[-1],
        "shipped_carries_bad": BAD[:20] in lines[-1],
        "shipped_parsed": parse_brief(lines[-1])[0] is None,
        "expected": [ref.Task("t", want, researcher).expected_output
                     for want in ("3 sources", "3 paragraphs", "800 words")],
        "reads_expected": inspect.getsource(
            ref.SequentialCrew.kickoff).count("expected_output"),
        "memory_entries": len(memory.short_term), "naive_entries": len(naive.short_term),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one malformed output, one retry, a three-field Brief",
            all([result["attempts"] == 2, result["fields"] == 3,
                 result["sections"] == 2, result["lines"] == 4,
                 result["failure_line"] == "[writer] invalid json",
                 result["brief"]["title"] == "Agent engineering 2026"]),
            f"the writer emits invalid JSON on attempt 1 and valid JSON on attempt "
            f"{result['attempts']}, so the validating crew retries once and produces a "
            f"Brief with {result['fields']} fields and {result['sections']} sections. "
            f"The failure is recorded as {result['failure_line']!r}, not raised",
        ),
        practice.Check(
            "FINDING: the shipped crew passes the malformed output downstream",
            all([result["shipped_lines"] == 3, result["shipped_carries_bad"] is True,
                 result["shipped_parsed"] is True]),
            f"with no validation the editor receives the broken JSON verbatim and the "
            f"run completes in {result['shipped_lines']} lines whose final output still "
            f"contains it ({result['shipped_carries_bad']}) and does not parse "
            f"({result['shipped_parsed']}). A pipeline that cannot fail cannot retry",
        ),
        practice.Check(
            "FINDING: expected_output is documentation",
            all([result["expected"] == ["3 sources", "3 paragraphs", "800 words"],
                 result["reads_expected"] == 0]),
            f"every Task carries one -- {result['expected']} -- and kickoff mentions the "
            f"field {result['reads_expected']} times. The three strings describe three "
            "different units, a count, a structure and a length, so even a generic "
            "checker would have nothing uniform to check",
        ),
        practice.Check(
            "FINDING: retrying is only safe because the agents are pure",
            all([result["memory_entries"] == 2, result["naive_entries"] == 3,
                 result["naive_entries"] > result["memory_entries"]]),
            f"the retry re-runs the writer against the same input, and SequentialCrew "
            f"writes to memory after every task -- so the validating wrapper leaves "
            f"{result['memory_entries']} short-term entries where an in-crew retry "
            f"leaves {result['naive_entries']}. Idempotence is a precondition the shipped "
            "crew never states",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
