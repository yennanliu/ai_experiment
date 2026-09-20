"""Exercise 6 — Level 1 scales with the catalog and the others with the task.

    Instrument the demo to report Level 1, Level 2, and Level 3 byte counts
    separately.

Reading of the exercise: "separately" is the whole instruction, so the three
counts are never summed into one number, and "byte counts" is taken against
the demo, which reports characters. Reporting them apart immediately shows
they are not three parts of one budget: one of them grows with the number of
skills installed and the other two grow with the task, so their ratio is a
property of the deployment rather than of the skill.

**ANSWER: three counts, in bytes, reported apart and per resource.** With
**3** skills installed the run costs **459** bytes at Level 1, **28** at
Level 2 and **80** at Level 3 across **2** resources. Level 1 is **81%** of
the total, and none of the three is derivable from the others. Level 1 is
measured with the install path folded to its last segment, because the
shipped entry carries an absolute one.

**FINDING: Level 1 scales with the catalog and the others with the task.**
Installing **50** skills instead of 3 leaves Levels 2 and 3 byte-identical
and takes Level 1 to **99%** of the run. Halving every description is the
only lever that touches the first number, and it does nothing for the other
two -- which is why the lesson calls them different budgets.

**FINDING: the per-call cap is not a budget.** `load_reference` bounds each
file at 12000 characters and nothing accumulates across calls, so **2**
references cost **80** bytes against **46** for one, and no code path
notices. A Level
3 budget has to live in the caller, because the loader is stateless by
construction.

**FINDING: the demo reports characters, and reports the smaller of the two
catalog numbers.** `catalog_chars` measures the entries the model sees and
`report_chars` the whole diagnostic including collisions and omissions --
**459** against a larger report here, and both include the absolute install
path -- so the shipped Level 1 figure changes when the tree moves. A CJK
description makes the gap worse in the other direction: **240** characters
is **720** bytes.

Structure: `levels()` returns the three counts for one run, so the 3-skill
and 50-skill cases are the same function called twice.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import tempfile
from dataclasses import asdict

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "24-skill-discovery-and-progressive-disclosure"
DESCRIPTION = "Report evidence for audit number {n} when that audit completes."
REFERENCES = ("references/format.md", "references/schema.md")


def catalog_bytes(entries, absolute=False):
    """Level 1 in bytes, optionally with the absolute install path left in."""
    rows = [asdict(entry) if absolute
            else {**asdict(entry), "directory": entry.directory.rsplit("/", 1)[-1]}
            for entry in entries]
    return len(json.dumps({"entries": rows}, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


def plant(ref, root, count):
    for index in range(count):
        name = "evidence-report" if index == 0 else f"audit-helper-{index}"
        ref._write_skill(root, name, DESCRIPTION.format(n=index), f"# {name}\n\nStep one.")
    skill = root / "evidence-report"
    (skill / "references" / "schema.md").write_text(
        "# Schema\n\nOne object per finding.\n", encoding="utf-8")


def levels(ref, root, count):
    """Level 1, Level 2 and Level 3 for one activation, in bytes, kept apart."""
    plant(ref, root, count)
    candidates = ref.discover_scope(ref.Scope("user", root))
    catalog = ref.build_catalog(candidates, ("user",),
                                ref.CatalogBudget(max_entries=count, max_catalog_chars=80_000))
    entry = next(e for e in catalog.entries if e.name == "evidence-report")
    body = ref.load_skill_body(entry, candidates)
    resources = {name: len(ref.load_reference(entry, name).encode("utf-8"))
                 for name in REFERENCES}
    return {"level_1": catalog_bytes(catalog.entries),
            "level_1_absolute": catalog_bytes(catalog.entries, absolute=True),
            "level_2": len(body.encode("utf-8")),
            "level_3": sum(resources.values()), "resources": resources,
            "entries": len(catalog.entries), "report_chars": catalog.report_chars,
            "catalog_chars": catalog.catalog_chars}


def share(run):
    return run["level_1"] / (run["level_1"] + run["level_2"] + run["level_3"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        base = pathlib.Path(temp)
        small = levels(ref, base / "small", 3)
        large = levels(ref, base / "large", 50)
        wide = "中" * 240
        ref._write_skill(base / "wide", "evidence-report", wide, "# Evidence\n")
        candidates = ref.discover_scope(ref.Scope("user", base / "wide"))
        entry = ref.build_catalog(candidates, ("user",), ref.CatalogBudget()).entries[0]
        return {
            "small": small, "large": large,
            "small_share": round(share(small), 2), "large_share": round(share(large), 2),
            "same_body": small["level_2"] == large["level_2"],
            "same_resources": small["resources"] == large["resources"],
            "resource_count": len(small["resources"]),
            "one_resource": small["resources"][REFERENCES[0]],
            "per_call_cap": inspect.signature(
                ref.load_reference).parameters["max_chars"].default,
            "accumulators": [name for name, value in vars(ref).items()
                             if isinstance(value, (list, dict, set))
                             and not name.startswith("__")],
            "wide_chars": len(entry.description),
            "wide_bytes": len(entry.description.encode("utf-8")),
        }


def verify(result):
    small, large = result["small"], result["large"]
    return [
        practice.Check(
            "ANSWER: three counts, in bytes, reported apart and per resource",
            all([small["entries"] == 3, small["level_1"] == 459, small["level_2"] == 28,
                 small["level_3"] == 80, result["resource_count"] == 2,
                 small["level_3"] == sum(small["resources"].values())]),
            f"with {small['entries']} skills installed the run costs {small['level_1']} "
            f"bytes at Level 1, {small['level_2']} at Level 2 and {small['level_3']} at "
            f"Level 3 across {result['resource_count']} resources "
            f"{small['resources']}. Level 1 is {result['small_share']:.0%} of the total and "
            "none of the three is derivable from the others",
        ),
        practice.Check(
            "FINDING: Level 1 scales with the catalog and the others with the task",
            all([large["entries"] == 50, result["same_body"], result["same_resources"],
                 result["large_share"] > 0.95, result["small_share"] < 0.90]),
            f"installing {large['entries']} skills instead of {small['entries']} leaves "
            f"Levels 2 and 3 byte-identical at {small['level_2']} and {small['level_3']}, "
            f"and takes Level 1 from {result['small_share']:.0%} to "
            f"{result['large_share']:.0%} of the run. Shortening descriptions moves the "
            "first number and nothing else -- two budgets, not two halves of one",
        ),
        practice.Check(
            "FINDING: the per-call cap is not a budget",
            all([result["per_call_cap"] == 12_000, result["accumulators"] == [],
                 small["level_3"] > result["one_resource"],
                 result["one_resource"] == 46]),
            f"load_reference bounds each file at {result['per_call_cap']} characters and "
            f"the module holds {len(result['accumulators'])} mutable module-level "
            f"containers to accumulate in, so {result['resource_count']} references "
            f"cost {small['level_3']} bytes against {result['one_resource']} for one and no "
            "code path notices. A Level 3 budget has to live in the caller, because the "
            "loader is stateless by construction",
        ),
        practice.Check(
            "FINDING: the demo reports characters, and the smaller of the two catalog numbers",
            all([small["report_chars"] > small["catalog_chars"],
                 small["level_1_absolute"] > small["level_1"],
                 result["wide_chars"] == 240, result["wide_bytes"] == 720]),
            f"catalog_chars is {small['catalog_chars']} and report_chars "
            f"{small['report_chars']}, and the demo prints the first -- correct for what "
            f"the model sees, silent about what discovery produced. Both carry the "
            f"absolute install path, so Level 1 reads {small['level_1_absolute']} bytes "
            f"here against {small['level_1']} with the path folded, and the figure moves "
            f"when the tree does. In the other direction {result['wide_chars']} CJK "
            f"characters are {result['wide_bytes']} bytes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
