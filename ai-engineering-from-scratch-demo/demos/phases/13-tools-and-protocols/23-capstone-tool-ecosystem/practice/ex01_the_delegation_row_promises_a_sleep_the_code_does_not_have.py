"""Exercise 1 — the delegation row promises a sleep the code does not have.

    Run `code/main.py`. Separate facts proven by the output from production
    claims that still need integration evidence.

Reading of the exercise: a partition is only worth writing if each side is
decided by something other than the author's judgement, so each fact is
re-derived from a fresh run and each claim is tested against the module's
import graph. A boundary that needs a socket, a subprocess or a browser
cannot have been crossed by a module that imports none of them -- which makes
the "claims" column provable as a negative rather than merely asserted.

**ANSWER: 6 facts, all re-derived; 9 production layers, all unevidenced by
construction.** The module imports `hashlib`, `json`, `time`, `uuid`, `copy`
and `datetime` -- **0** of the 10 modules that could open a socket, spawn a
process or touch a database. Every row of the lesson's own
simulation-versus-production table is on the claims side, and no reading of
the output could move one.

**FINDING: the delegation row promises a sleep the code does not have.** The
table describes the delegation layer as "Sleep plus nested span", and
`time.sleep` appears **0** times; `time` is imported for `time_ns` alone. The
`a2a.SendMessage` span lasts under a microsecond, so the stub does not even
simulate the one property -- latency -- that makes delegation worth a timeout
test.

**FINDING: the audit log and the span list are process-global, so a per-run
fact has to be sliced out.** Two orchestrator runs leave **4** audit rows and
**11** spans in one list under **2** trace ids. "Every span in one run shares
one trace id" is true and is not what the printed list shows.

**FINDING: opacity here is the absence of a field rather than a boundary.**
The `a2a.SendMessage` span carries **2** attributes, peer and skill, and no
content -- but the writer is the same process, and the HTML it "returns" is
built inline by the caller. Nothing was withheld, because nothing was ever
separate.

Structure: `facts()` re-derives each printed claim from a fresh run;
`io_modules()` is the negative test that decides the other column.
"""

from __future__ import annotations

import ast
import pathlib

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "23-capstone-tool-ecosystem"
IO_CAPABLE = frozenset({"socket", "ssl", "http", "urllib", "subprocess",
                        "asyncio", "sqlite3", "os", "selectors", "webbrowser"})
LAYERS = ["Discovery", "Authentication", "Authorization", "Search", "Tasks",
          "Delegation", "App", "Telemetry", "Sandbox"]


def imports_of(source):
    """Top-level module names the reference imports, from its AST."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return sorted(names)


def spans_of(ref, trace_id):
    return [sp for sp in ref.SPANS if sp["traceId"] == trace_id]


def parented(spans):
    """Every span but the root names a parent that is a span in the same run."""
    ids = {sp["spanId"] for sp in spans}
    roots = [sp for sp in spans if sp["parentSpanId"] is None]
    return len(roots) == 1 and all(sp["parentSpanId"] in ids for sp in spans[1:])


def facts(ref, alice, bob):
    run = spans_of(ref, alice["trace_id"])
    result = alice["task"]["result"]
    kinds = [item["type"] for item in result["content"]]
    a2a = next(sp for sp in run if sp["name"] == "a2a.SendMessage")
    discovery = ref.server_discover(ref.request_meta())
    return {
        "discovery advertises 2026-07-28 and the tasks extension":
            discovery["supportedVersions"] == [ref.PROTOCOL_VERSION]
            and ref.TASK_EXTENSION in discovery["capabilities"]["extensions"],
        "alice may write and bob may not":
            alice["report"]["resultType"] == "task"
            and bob["report"] == {"error": "insufficient_scope",
                                  "scope": "research:write"},
        "one run, one trace id, every child parented": parented(run),
        "the handle becomes a completed task through tasks/get":
            alice["report"]["status"] == "working"
            and alice["task"]["resultType"] == "complete"
            and alice["task"]["status"] == "completed",
        "the final result carries text and a ui:// reference":
            kinds == ["text", "ui_resource"]
            and result["ui"]["resourceUri"].startswith("ui://"),
        "the writer boundary records no writer content":
            sorted(a2a["attrs"]) == ["a2a.peer", "a2a.skill"],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = pathlib.Path(ref.__file__).read_text(encoding="utf-8")
    alice = ref.orchestrator("tok_alice", "summarize the 2026 arXiv papers")
    bob = ref.orchestrator("tok_bob", "generate a report")
    proven = facts(ref, alice, bob)
    run = spans_of(ref, alice["trace_id"])
    a2a = next(sp for sp in run if sp["name"] == "a2a.SendMessage")
    return {
        "facts": proven, "facts_held": sorted(k for k, v in proven.items() if v),
        "imports": imports_of(source),
        "io_imports": sorted(set(imports_of(source)) & IO_CAPABLE),
        "layers": LAYERS, "sleeps": source.count("time.sleep("),
        "a2a_us": (a2a["end"] - a2a["start"]) / 1_000,
        "a2a_attrs": sorted(a2a["attrs"]),
        "html_built_by": "research_generate_report" in source
        and "html = (" in source,
        "audit_rows": len(ref.AUDIT), "spans": len(ref.SPANS),
        "traces": len({sp["traceId"] for sp in ref.SPANS}),
        "run_spans": len(run),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: six facts re-derived, nine production layers unevidenced by import",
            all([len(result["facts_held"]) == 6, result["io_imports"] == [],
                 len(result["layers"]) == 9]),
            f"every one of the {len(result['facts_held'])} printed facts survives being "
            f"re-derived from a fresh run, while the module imports {result['imports']} -- "
            f"{len(result['io_imports'])} of the {len(IO_CAPABLE)} modules that could open a "
            f"socket, spawn a process or reach a database. All {len(result['layers'])} rows "
            "of the lesson's own table are on the claims side, provably",
        ),
        practice.Check(
            "FINDING: the delegation row promises a sleep the code does not have",
            all([result["sleeps"] == 0, result["a2a_us"] < 1_000,
                 "time" in result["imports"]]),
            f"the table calls the delegation layer 'Sleep plus nested span' and time.sleep "
            f"appears {result['sleeps']} times; time is imported for time_ns alone. The "
            f"a2a.SendMessage span lasts {result['a2a_us']:.1f}us, so the stub does not "
            "simulate latency -- the one property that would make a timeout test mean "
            "something",
        ),
        practice.Check(
            "FINDING: the audit log and the span list are process-global",
            all([result["audit_rows"] == 4, result["spans"] == 11,
                 result["traces"] == 2, result["run_spans"] == 7]),
            f"two runs leave {result['audit_rows']} audit rows and {result['spans']} spans in "
            f"one list under {result['traces']} trace ids. 'Every span in one run shares one "
            f"trace id' is true of the {result['run_spans']} sliced out by trace id and is "
            "not what the printed list shows",
        ),
        practice.Check(
            "FINDING: opacity here is the absence of a field rather than a boundary",
            all([result["a2a_attrs"] == ["a2a.peer", "a2a.skill"],
                 result["html_built_by"]]),
            f"the delegation span carries {result['a2a_attrs']} and no content, but the "
            "writer is this process and the HTML it 'returns' is built inline by the caller. "
            "Nothing was withheld because nothing was ever separate -- which is why the "
            "opacity test belongs on the claims side too",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
