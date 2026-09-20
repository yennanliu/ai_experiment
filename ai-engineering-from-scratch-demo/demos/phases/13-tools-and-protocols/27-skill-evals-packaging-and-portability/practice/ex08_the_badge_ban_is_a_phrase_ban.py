"""Exercise 8 — the badge ban is a phrase ban.

    Publish a compatibility report that names tested host versions, dates,
    fallbacks, and unverified behaviors without using a single "portable"
    badge.

Reading of the exercise: "without a badge" is a negative requirement, so it
is enforced the way the module already enforces one -- an `ArtifactContract`
with `forbidden_terms` -- and the positive half becomes required headings.
Wiring that up shows the ban is weaker than it reads, and wiring up the
positive half shows the matrix cannot supply three of the four facts.

**ANSWER: a report that passes a contract requiring four sections and
forbidding five badge phrases.** `Tested hosts`, `Fallbacks`, `Not verified`
and `Method` are all present, every host row carries a version and a test
date, and `evaluate_artifact` reports **0** forbidden hits and **0** missing
headings. The word `portable` appears **0** times.

**FINDING: the badge ban is a phrase ban.** `evaluate_artifact` matches each
forbidden term as a whole word with `(?<!\\w)term(?!\\w)`, so
`"guaranteed portable"` catches that exact phrase and misses `"fully
portable"`, `"portable"` and `"portability-verified"`. Banning a claim means
enumerating its spellings; **1** of the **4** variants is caught by the
lesson's own term list.

**FINDING: three of the four required facts are not in the matrix.**
`portability_matrix` rows carry `host`, `status` and `missing`, and
`HostCapabilities` carries a bare `name`. Versions, dates and fallbacks have
to travel beside the matrix in a structure the module does not define, which
is why a compatibility report is authored rather than generated.

**FINDING: "not verified" is the complement of a set nothing enumerates.**
`missing` lists capabilities known to be absent -- **2** entries here -- and
there is no field for a capability nobody tested. The honest section is the
one with no data source, so it is the section most likely to be quietly
dropped.

Structure: `render()` builds the report from rows, and `CONTRACT` is what
decides whether it is a compatibility report or a badge.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "27-skill-evals-packaging-and-portability"
BADGES = ("portable", "guaranteed portable", "fully portable", "portability-verified",
          "works everywhere")
ROWS = (
    {"host": "orchestrator", "version": "4.2.1", "tested": "2026-08-14",
     "status": "native", "fallback": "none needed"},
    {"host": "notebook-runner", "version": "0.9.3", "tested": "2026-08-14",
     "status": "adapter-required", "fallback": "companion files are inlined into the body"},
    {"host": "chat-only", "version": "2026-07", "tested": "2026-08-12",
     "status": "unsupported", "fallback": "the skill is not offered on this host"},
)
UNVERIFIED = ("behaviour under a cold cache",
              "concurrent activation from two sessions",
              "hosts newer than the dates below")
CONTRACT_HEADINGS = ("Tested hosts", "Fallbacks", "Not verified", "Method")


def render(rows, unverified):
    """Four sections, each carrying a fact the reader would otherwise assume."""
    tested = "\n".join(f"- {row['host']} {row['version']}, tested {row['tested']}: "
                       f"{row['status']}" for row in rows)
    fallbacks = "\n".join(f"- {row['host']} {row['version']}: {row['fallback']}"
                          for row in rows)
    unknown = "\n".join(f"- {item}" for item in unverified)
    return (f"# Compatibility\n\n## Tested hosts\n\n{tested}\n\n"
            f"## Fallbacks\n\n{fallbacks}\n\n## Not verified\n\n{unknown}\n\n"
            f"## Method\n\nOne fixture run per host on the date shown; no host was "
            f"retested after its version changed.\n")


def contract(ref):
    return ref.ArtifactContract(required_headings=CONTRACT_HEADINGS,
                                required_terms=("tested", "fallbacks"),
                                forbidden_terms=BADGES)


def badge_hits(ref, text):
    shipped = ref.ArtifactContract(forbidden_terms=("guaranteed portable",))
    return ref.evaluate_artifact(text, shipped)["forbidden_hits"]


def shipped_matrix(ref):
    hosts = (ref.HostCapabilities("orchestrator", True, True, True),
             ref.HostCapabilities("notebook-runner", True, False, True),
             ref.HostCapabilities("chat-only", False, False, False))
    return ref.portability_matrix(
        ref.PackageRequirements(companion_files=True, script_execution=True), hosts)


def variants_of(ref):
    return {phrase: len(badge_hits(ref, f"This bundle is {phrase}."))
            for phrase in ("guaranteed portable", "fully portable", "portable",
                           "portability-verified")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    report = render(ROWS, UNVERIFIED)
    evaluated = ref.evaluate_artifact(report, contract(ref))
    variants, matrix = variants_of(ref), shipped_matrix(ref)
    return {
        "passed": evaluated["passed"], "missing_headings": evaluated["missing_headings"],
        "missing_terms": evaluated["missing_terms"],
        "forbidden_hits": evaluated["forbidden_hits"], "headings": list(CONTRACT_HEADINGS),
        "portable_count": report.lower().count("portable"),
        "dated": sum(row["tested"] in report for row in ROWS),
        "versioned": sum(f"{row['host']} {row['version']}" in report for row in ROWS),
        "unverified": len(UNVERIFIED),
        "variants": variants,
        "caught": sorted(phrase for phrase, hits in variants.items() if hits),
        "row_keys": sorted(matrix[0]), "host_fields":
            list(vars(ref.HostCapabilities)["__dataclass_fields__"]),
        "missing_capabilities": sorted(
            item for row in matrix for item in row["missing"]),
        "date_fields": [name for name in vars(ref.HostCapabilities)["__dataclass_fields__"]
                        if "date" in name or "version" in name],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a report passing a contract that requires four sections and bans badges",
            all([result["passed"], result["missing_headings"] == [],
                 result["missing_terms"] == [],
                 result["forbidden_hits"] == [], result["portable_count"] == 0,
                 result["dated"] == 3, result["versioned"] == 3,
                 result["unverified"] == 3]),
            f"{result['headings']} are all present, {result['versioned']} host rows carry a "
            f"version and {result['dated']} a test date, {result['unverified']} behaviours "
            f"are listed as unverified, and evaluate_artifact reports "
            f"{len(result['forbidden_hits'])} forbidden hits. The word 'portable' appears "
            f"{result['portable_count']} times",
        ),
        practice.Check(
            "FINDING: the badge ban is a phrase ban",
            all([result["caught"] == ["guaranteed portable"],
                 result["variants"]["fully portable"] == 0,
                 result["variants"]["portable"] == 0,
                 result["variants"]["portability-verified"] == 0]),
            f"evaluate_artifact matches each forbidden term as a whole word, so the "
            f"lesson's own ('guaranteed portable',) catches {result['caught']} and misses "
            f"the other three: {sorted(result['variants'])}. Banning a claim means "
            "enumerating its spellings, which is why this report bans five phrases and "
            "still cannot ban the idea",
        ),
        practice.Check(
            "FINDING: three of the four required facts are not in the matrix",
            all([result["row_keys"] == ["host", "missing", "status"],
                 result["host_fields"][0] == "name", result["date_fields"] == []]),
            f"portability_matrix rows carry {result['row_keys']} and HostCapabilities "
            f"carries {result['host_fields']} -- {len(result['date_fields'])} of them a "
            "version or a date. Versions, dates and fallbacks travel beside the matrix in "
            "a structure the module does not define, which is why a compatibility report "
            "is authored rather than generated",
        ),
        practice.Check(
            "FINDING: not verified is the complement of a set nothing enumerates",
            all([result["missing_capabilities"] == ["companion-files", "core-skill-loader"],
                 result["unverified"] == 3,
                 "unverified" not in " ".join(result["row_keys"])]),
            f"missing lists capabilities known to be absent -- "
            f"{result['missing_capabilities']} -- and there is no field for a capability "
            f"nobody tested. The {result['unverified']} entries in the Not verified "
            "section have no data source at all, which makes it the section most likely "
            "to be quietly dropped",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
