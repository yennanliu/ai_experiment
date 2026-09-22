"""Exercise 5 — the blocked plan renders no schedule, in either format.

    Render the plan as Markdown while keeping JSON as the source of truth.

Reading of the exercise: "JSON as the source of truth" is a testable
property, not a slogan. The renderer must add no field the document does not
carry, and every number in the Markdown must be derivable from the JSON. What
the round trip then exposes is what the JSON does not carry.

**ANSWER: the Markdown reproduces all 4 top-level keys and 20 item fields from
the document, adding 0 facts.** Rendering the ready plan gives a table of
**4** items, a wave list of **3** rows and a status line; parsing the ids back
out of the Markdown returns the **4** the JSON holds, in the same order. The
renderer reads `plan_document` and nothing else.

**FINDING: a plan blocked on a missing proof renders with no waves at all.**
`plan_document` sets `"waves": []` whenever `issues` is non-empty, so a plan
whose only fault is one empty proof string loses all **3** of its computable
waves, and the Markdown inherits the hole because the JSON is the source of
truth.

**FINDING: the waves are computed twice and discarded once.** `validate` calls
`execution_waves` to detect a cycle, then `plan_document` calls it again to
report the waves -- **2** traversals, and on a blocked plan the second one is
skipped and the first one's result is thrown away. Keeping it would cost
nothing and would leave the schedule intact.

**FINDING: evidence renders as a string, and nothing resolves it.** The
document carries **4** items with **4** evidence receipts, **0** of which name
a file that exists in the lesson directory -- the same fiction as the previous
lesson's frame. The Markdown makes them look like links; they resolve
**0** of **4** times.

Structure: `render()` is the Markdown view; `parity_check()` compares what the
Markdown says against what the document holds.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "44-plan-from-evidence"


def render(document):
    """Markdown built only from the JSON document -- no second source."""
    lines = [f"# Execution plan ({document['status']})", ""]
    if document["issues"]:
        lines += ["## Blocking issues", ""] + [f"- {issue}" for issue in document["issues"]] + [""]
    lines += ["## Waves", ""]
    lines += [f"{index}. {', '.join(wave)}" for index, wave in enumerate(document["waves"], 1)]
    lines += ["", "## Items", "", "| id | change | evidence | depends on | proof |",
              "|---|---|---|---|---|"]
    for item in document["items"]:
        lines.append(f"| `{item['id']}` | {item['change']} | "
                     f"{', '.join(item['evidence'])} | {', '.join(item['depends_on']) or '-'} "
                     f"| `{item['proof']}` |")
    return "\n".join(lines) + "\n"


def parity_check(document, markdown):
    """Every id in the Markdown, in order, so the two views can be compared."""
    rows = [line for line in markdown.splitlines() if line.startswith("| `")]
    return [re.findall(r"`([^`]+)`", row)[0] for row in rows]


def broken(ref):
    """The lesson's plan with one empty proof and nothing else wrong."""
    return [row if row.id != "integration" else
            ref.WorkItem(row.id, row.change, row.evidence, row.depends_on, "   ")
            for row in ref.example()]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ready = ref.plan_document(ref.example())
    markdown = render(ready)
    blocked = ref.plan_document(broken(ref))
    blocked_md = render(blocked)
    lesson_root = Path(ref.__file__).resolve().parents[1]
    receipts = [receipt for item in ready["items"] for receipt in item["evidence"]]
    return {
        "keys": sorted(ready), "ids": parity_check(ready, markdown),
        "document_ids": [item["id"] for item in ready["items"]],
        "item_fields": sum(len(item) for item in ready["items"]),
        "wave_rows": len([line for line in markdown.splitlines()
                          if re.match(r"^\d+\. ", line)]),
        "blocked_status": blocked["status"], "blocked_issues": blocked["issues"],
        "blocked_waves": blocked["waves"],
        "blocked_wave_rows": len([line for line in blocked_md.splitlines()
                                  if re.match(r"^\d+\. ", line)]),
        "schedulable": len(ref.execution_waves(broken(ref))),
        "traversals": inspect.getsource(ref.validate).count("execution_waves")
        + inspect.getsource(ref.plan_document).count("execution_waves"),
        "receipts": len(receipts),
        "resolving": sum((lesson_root / receipt.rpartition(":")[0]).exists()
                         for receipt in receipts),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the Markdown reproduces the document and adds nothing",
            all([result["keys"] == ["issues", "items", "status", "waves"],
                 result["ids"] == result["document_ids"],
                 result["item_fields"] == 20, result["wave_rows"] == 3]),
            f"the document holds {result['keys']} and {result['item_fields']} item fields; "
            f"the Markdown renders {result['wave_rows']} wave rows and parses back to "
            f"{result['ids']}, the same ids in the same order",
        ),
        practice.Check(
            "FINDING: a plan blocked on a missing proof renders with no waves",
            all([result["blocked_status"] == "blocked", result["blocked_waves"] == [],
                 result["blocked_wave_rows"] == 0, result["schedulable"] == 3,
                 len(result["blocked_issues"]) == 1]),
            f"one empty proof string gives status {result['blocked_status']!r}, "
            f"{len(result['blocked_issues'])} issue and waves {result['blocked_waves']}, so "
            f"the {result['schedulable']} waves that were computable are lost -- and the "
            "Markdown inherits the hole, because the JSON is the source of truth",
        ),
        practice.Check(
            "FINDING: the waves are computed twice and discarded once",
            result["traversals"] == 2,
            f"validate calls execution_waves to find cycles and plan_document calls it "
            f"again to report them -- {result['traversals']} traversals, and on a blocked "
            "plan the first result is thrown away rather than kept",
        ),
        practice.Check(
            "FINDING: evidence renders as a string and nothing resolves it",
            all([result["receipts"] == 4, result["resolving"] == 0]),
            f"{result['receipts']} receipts, {result['resolving']} of which name a file that "
            "exists in the lesson directory; the Markdown makes them look like links",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
